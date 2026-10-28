# Anjo Sleep dev host and Glimmer serving (Task 4)

This runbook names settings, never their values. No AWS key exists anywhere: the dev host's AWS
calls run as an IAM role it assumes with its own Google identity. Modal's secret (for the fallback)
and the dev host's environment file hold the rest, and nothing is in the repo. Everything here that
starts paid GPU time or touches DNS is run by the coordinator or Kevin, on Kevin's go.

There are three parts:

1. **Glimmer on SageMaker** (`anjo/sleep/serving/sagemaker/`), with Modal as the fallback
   (`anjo/sleep/serving/modal_app.py`): it only serves. Training runs on Modal's own training app,
   with Colab as the fallback (§1.7, Task 5).
2. **The dev host**: a small CPU-only GCP VM that runs Anjo in production mode with `ANJO_SLEEP_HOST=1`.
3. **The dev app**: the existing mobile app, built in Xcode with its own bundle id and
   `EXPO_PUBLIC_API_URL` pointing at the dev host. Nothing in `mobile/` changes for it.

## 1. Glimmer on SageMaker (Modal is the fallback)

**What runs.** vLLM's OpenAI-compatible server serves `meta-models/Muse-Glimmer-30B` (Apache 2.0)
at the pinned commit, on one L40S in AWS SageMaker, paid by AWS credits (Kevin's call, 2026-10-27).
LoRA is on, and each person's adapter `p-<hash>-n<night>` is loaded before the first request that
names it, so the base model and every personal arm share one GPU. Modal (§1.6) stays as the fallback, one
setting away: `ANJO_SLEEP_BACKEND=modal`.

| Setting | Value |
|---|---|
| Region, instance | `us-east-1`, `ml.g6e.xlarge` (one L40S 48 GB, 4 vCPU, 32 GiB) or `ml.g6e.2xlarge` (the same GPU, 8 vCPU, 64 GiB); the account's endpoint quota is 1 |
| Precision | FP8, quantized when vLLM loads the weights (`config.SERVING = FP8_L40S`); the same on Modal |
| vLLM | `0.28.0` (`config.DEFAULT_VLLM_VERSION`), the first release whose registry has Glimmer's `MuseGlimmerForConditionalGeneration` (vLLM #51655); 0.11.0 can't load it. The image is `FROM vllm/vllm-openai:v0.28.0` |
| Names | endpoint `anjo-sleep-glimmer`, config `anjo-sleep-glimmer-config`, inference component `anjo-sleep-glimmer-ic`, models `anjo-sleep-glimmer-<image tag>`, ECR repository `anjo-sleep-glimmer` (`serving/sagemaker/settings.py`) |
| Bucket | `anjo-sleep-sagemaker-<suffix>` (the operator's policy reaches only buckets named `*sagemaker*`): `weights/<model>/<revision>/` and `adapters/<name>/` |
| Roles | `anjo-sleep-sagemaker-exec` (the container) and `anjo-sleep-devhost` (the VM), from `deploy/aws/` |
| Scale to zero | the component's copy count is 0 between uses, and managed instance scaling (min 0, max 1) then removes the instance. Waking sets it to 1 |
| Context | 32,768 tokens per request (`MAX_MODEL_LEN`), 4,096 output |
| LoRA | rank 32 at most, 8 adapters on the GPU at once, 16 kept on the CPU (see the sizing below) |
| Reply deadline | the dev host waits 20 s for the reply and for each read after (`runtime.REPLY_DEADLINE_S`), then the frontier talker answers |
| Cost | about **$2.61 an hour on g6e.xlarge and $2.81 on g6e.2xlarge while a copy is up**, nothing while it is at 0 (the bucket and the image cost cents). When g6e.xlarge has no capacity (`InsufficientInstanceCapacity`), use g6e.2xlarge: `deploy up --instance-type ml.g6e.2xlarge` (also on `status` and `stop`), or `ANJO_SLEEP_INSTANCE_TYPE`; the option wins, a type outside the two is refused, and a config that exists keeps its type until `stop --delete-endpoint`. Set the fourth value of `ANJO_SLEEP_PRICES` to the chosen type's rate so the budget ledger's `modal_usd` pool holds the right rate (the pool keeps its name) |

Every arm serves at the same precision: the base and the adapters run on the same server.

### 1.1 How it fits together

- **The container** (`serving/sagemaker/Dockerfile`, `shim.py`): vLLM on loopback behind a small
  shim on port 8080. `GET /ping` answers 200 only once vLLM's `/health` does. `POST /invocations`
  takes the envelope `{"path", "body", "stream"}` and forwards it to vLLM, for the four paths the
  dev host uses, streaming the reply back (chunked whenever the envelope asks for a stream). The
  output limit is held to 4,096 under either name, `max_tokens` or `max_completion_tokens`. The
  weights come from the bucket (SageMaker copies
  `weights/<model>/<revision>/` into `/opt/ml/model`), never from Hugging Face. The shim logs
  request lines, never a body.
- **Adapters.** On the first request that names `p-<hash>-n<night>`, the shim fetches exactly
  `adapter_config.json` and `adapter_model.safetensors` from `adapters/<name>/`, holds them to the
  same checks as Modal's upload (`config.check_adapter`: a PEFT LoRA of Glimmer, rank 32 or less,
  nothing else in the folder), and loads them through vLLM's runtime `/v1/load_lora_adapter`, once
  per name. A refused adapter gets a 422 and is never loaded.
- **The dev host's calls** are SigV4-signed boto3 calls with the `anjo-sleep-devhost` role:
  `invoke_endpoint_with_response_stream` for turns (read timeout 20 s, one attempt) and for every
  call of the night's client (300 s, one attempt: `invoke_endpoint` gives up after 60 s, which a
  long reply outlasts; a chat streams and is put back together as one completion),
  `invoke_endpoint` only for the wake poll (10 s), and the component's copy count to wake and
  sleep. A component with no copy up refuses with "no capacity": the dev host reads that as a 503,
  transient, so the frontier talker answers the turn. No bearer token exists on this backend.
- **Uploads** (`backend.from_env().upload_adapter`, which the night's trainer uses): checked first,
  refused if `adapters/<name>/` holds anything, then written with `IfNoneMatch="*"`, the weights
  first and the config last, so a night's adapter is never overwritten, even by two uploads racing.

### 1.2 Waking and sleeping

- **Wake** (`SageMakerServing.wake`): if the copy count is 0 it sets it to 1, then polls
  `GET /v1/models` through the endpoint with a backoff, under a hard total of
  `settings.WAKE_TOTAL_TIMEOUT_S`, 1,200 s (a cold start provisions the instance, pulls the image,
  copies the weights and loads them). A wake that runs out fails with `glimmer_not_ready`, exactly
  as on Modal. If the copy count drops back to 0 mid-wake, the wake sets it again, three times at most.
- **Sleep** (`sleep_now`): sets the copy count to 0. **Every phase ends with it**, well or not: the
  simulated day, the night batch, the morning run and a keep-warm window each run inside
  `backend.phase`, which calls it on the way out. The daily night phase and an accelerated run
  nest their phases, so Glimmer stays up from the night into the morning and sleeps once at the end.
- **A turn never wakes SageMaker.** While the copy count is 0 the endpoint refuses at once, and the
  frontier talker answers. Days, nights, mornings and demos wake it first.
- **The idle net** (AWS side, made by `deploy up`): the alarm `anjo-sleep-idle` reads 1 for each
  minute in which a copy is up (its memory metric exists) and nobody invoked it, and after 30 such
  minutes in a row its step policy `anjo-sleep-idle-to-zero` sets the copy count to exactly 0. A
  sleeping component never breaches it. A copy still loading looks the same as an idle one (no
  metric tells them apart), so the window is the longest wake (1,200 s) plus ten minutes: the net
  never takes down a copy a wake is still waiting for. A missed sleep costs at most 30 minutes of
  the instance, about $1.30. It covers a process that died, a sleep call that failed, and a night's
  training (nothing pings during training).

### 1.3 Stand-up, in order (Kevin runs every paid step)

Run these from Kevin's machine with the operator's AWS profile, except step 6 on the VM:

```bash
export ANJO_SLEEP_GLIMMER_REVISION=cec94fcfd6512f2cfe708138c32533b346ea4f69   # the pin
export ANJO_SLEEP_S3_BUCKET=anjo-sleep-sagemaker-<suffix>

# 1. IAM (console): the two roles from anjo/sleep/deploy/aws/ (README.md there says how).
# 2. The bucket (private, SSE-S3 by default; no bucket policy, versioning or lifecycle rule).
aws s3api create-bucket --bucket "$ANJO_SLEEP_S3_BUCKET" --region us-east-1
# 3. The weights, Hugging Face to S3 one file at a time (about 60 GB; a rerun resumes).
python -m anjo.sleep.serving.sagemaker.deploy stage-weights --scratch <dir with 10 GB free>
# 4. The image: writes a build context with only the shim's files, prints the commands to run.
python -m anjo.sleep.serving.sagemaker.deploy image --context <empty dir>
#    -> aws ecr create-repository ...; aws ecr get-login-password ... | docker login ...;
#       docker buildx build --platform linux/amd64 ... --provenance=false --sbom=false
#         --output type=image,name=<uri>,push=true,oci-mediatypes=false <dir>
#    SageMaker refuses an OCI image index (buildx's default push). An image already pushed that way
#    can be re-tagged as Docker v2 without a rebuild: `aws ecr batch-get-image` the amd64 manifest
#    digest, rewrite the manifest to `application/vnd.docker.distribution.manifest.v2+json`, the config
#    to `application/vnd.docker.container.image.v1+json` and the layers to
#    `application/vnd.docker.image.rootfs.diff.tar.gzip`, then `aws ecr put-image` with
#    `--image-manifest-media-type application/vnd.docker.distribution.manifest.v2+json`.
# 5. The endpoint (paid, 20 to 40 minutes): model, config, endpoint, component, the idle net.
#    It waits for the first copy, prints first_copy_seconds, and ends at 0 copies.
python -m anjo.sleep.serving.sagemaker.deploy up --image-uri <the uri step 4 printed>
python -m anjo.sleep.serving.sagemaker.deploy status
# 6. The VM: the env file names below, and the token refresher units (§2).
```

`stop` sets the copy count to 0 at once; `upload-adapter --folder <dir> --name p-<hash>-n<night>`
uploads by hand. **If `up` fails** at any step (or is interrupted), it cleans up before it exits:
a component an earlier run made goes back to 0 copies; otherwise it deletes the component it made,
the endpoint and its config (an endpoint with no component would bill an idle instance), and the
model it made, then shows the error. Run `up` again after the fix. If the cleanup itself fails,
the error says so; then run `stop --delete-endpoint`, which deletes the component (if any), the
endpoint and its config, so no instance is left, and `status` (all null) to confirm.

The account's AWS policies are AmazonSageMakerFullAccess, AmazonEC2ContainerRegistryFullAccess and
billing read. The first `up` creates Application Auto Scaling's service-linked role for SageMaker;
if that is refused, it needs Kevin in the IAM console once.

### 1.4 The dev host's AWS identity (no keys)

The VM has no AWS key. A systemd timer writes a fresh Google ID token for the VM's service account
every 10 minutes (`python -m anjo.sleep.serving.sagemaker.token_refresh`, audience
`anjo-sleep-devhost-aws`), and boto3 trades it for one-hour credentials with
`AssumeRoleWithWebIdentity` against Google's built-in provider, re-reading the file each time it
refreshes. The env file names:

| Name | Value |
|---|---|
| `ANJO_SLEEP_BACKEND` | `sagemaker` (the default; `modal` for the fallback) |
| `ANJO_SLEEP_S3_BUCKET` | `anjo-sleep-sagemaker-<suffix>` |
| `AWS_ROLE_ARN` | the `anjo-sleep-devhost` role's ARN |
| `AWS_WEB_IDENTITY_TOKEN_FILE` | `/var/lib/anjo-sleep-aws/token` |
| `AWS_ROLE_SESSION_NAME` | `anjo-sleep-devhost` |
| `AWS_DEFAULT_REGION` | `us-east-1` |

The units are in §2. The token file is mode 600, and the token is never printed.

### 1.5 LoRA sizing

About 6 to 8 adapters are live at once: Kevin, Teo, three lives and the donors for arm 3x. So
`--max-loras 8` and `--max-cpu-loras 16`. vLLM queues a request for a ninth adapter rather than
failing it, and the CPU cache swaps it in. Task 5 trains rank 16, and Meta's recipe goes up to 32,
so `--max-lora-rank 32`.

vLLM sets aside each GPU slot at the full rank for every LoRA-capable projection (qkv, o, gate_up,
down), whatever the adapter targets. With Glimmer's shape (52 layers, hidden 6,656, 2 KV heads of
208, so a KV width of 416, intermediate 19,968), one slot at rank r holds, per layer:

| Projection | A (r × in) + B (out × r) | Per layer |
|---|---|---|
| qkv | 3 × 6,656 + (6,656 + 416 + 416) | 27,456 r |
| o | 6,656 + 6,656 | 13,312 r |
| gate_up | 2 × 6,656 + 2 × 19,968 | 53,248 r |
| down | 19,968 + 6,656 | 26,624 r |
| **all** | | **120,640 r** |

That is 52 × 120,640 × 32 ≈ 201M values per slot, about 0.40 GB in bf16 (LoRA runs in the model
dtype, bf16, at either precision). 8 slots take about 3.2 GB of GPU memory; rank 64 would have
taken 6.4 GB. The CPU cache holds adapters as trained (q, k, v and o at rank 16): about 68 MB
each, about 1.1 GB for 16.

What remains for the KV cache, at `GPU_MEMORY_UTILIZATION` 0.92 (estimates; the Oct 28 check reads
vLLM's own "maximum concurrency" line and records it here):

| Server | Usable | Weights | LoRA slots | Activations | KV cache |
|---|---|---|---|---|---|
| L40S, FP8 | 44 GB | about 30 GB | 3.2 GB | about 2 GB | about 9 GB |
| A100-80GB, bf16 | 74 GB | about 60 GB | 3.2 GB | about 3 GB | about 7 GB |

A token's KV is at most 52 × 2 × 416 × 2 bytes, about 86 KB, so 7 to 9 GB holds about 80k to 100k
tokens: two or three full 32k contexts at once, and more, since three layers in four keep only a
2,048-token window. If the check shows less room, `MAX_LORAS` comes down to 6 before the context does.

### 1.6 The Modal fallback

`ANJO_SLEEP_BACKEND=modal` puts everything back on Modal as it ran before (`serving/modal_app.py`):
the app `anjo-sleep-serve`, volumes `anjo-sleep-weights` and `anjo-sleep-adapters`, the secret
`anjo-sleep-serve` holding only `VLLM_API_KEY`, scale to zero after 600 idle seconds, and a
bearer-token https server. The dev host then also needs `ANJO_SLEEP_MODAL_URL` (the `Serve.web`
URL the deploy prints) and `ANJO_SLEEP_MODAL_TOKEN` (the same value as `VLLM_API_KEY`). The
deploy-time settings are the same names as above (`ANJO_SLEEP_GLIMMER_REVISION`,
`ANJO_SLEEP_TOOL_CALL_PARSER`, `ANJO_SLEEP_VLLM_VERSION`), plus `ANJO_SLEEP_GPU_SNAPSHOT=1` to try
Modal's GPU snapshot and `ANJO_SLEEP_HF_SECRET` (leave it unset: the repo isn't gated).

```bash
modal secret create anjo-sleep-serve VLLM_API_KEY=...                        # once
modal run anjo/sleep/serving/modal_app.py::download_weights
modal deploy anjo/sleep/serving/modal_app.py
modal run anjo/sleep/serving/modal_app.py::upload_adapter --folder <dir> --name p-<hash>-n<night>
modal app stop anjo-sleep-serve                                              # the RF3 drill
```

On Modal, sleeping is Modal's own idle window, and the L40S costs about $2 an hour while up. The
night's client loads each adapter through vLLM's `/v1/load_lora_adapter` from
`/adapters/<name>` before the first request that names it, once per client, and loads it again
once if a restarted container answers 404 (`serving/client.py`). vLLM's filesystem resolver isn't
used: in vLLM 0.28.0's source it serves an adapter only when its `base_model_name_or_path` equals the
path vLLM was started with (`/weights/...`, not the served name), which a trained adapter never names,
so it answered 404. A running container
mounts the adapters volume as it was at its start, and nothing reloads it: on night 1c (2026-10-28) the
container reloaded the volume every 20 s, the gates' first load of the new adapter never answered, and
the server stopped answering anything, `/v1/models` included. So when vLLM can't find an adapter
under `/adapters` (committed since the container started), the night's client calls `Serve.stage`
(`backend.stage_on_modal`, through the modal package, at most `backend.STAGE_TIMEOUT_S`). It copies
the adapter through the volume's API, not the mount, onto the container's own disk
(`config.STAGED_ADAPTERS_DIR`), checks it as an upload is checked, and vLLM loads it from there. An
adapter found nowhere is asked for again for up to 45 s (`client.LOAD_WAIT_S`). Only the night, the
morning and the demo's pre-generated pairs wait so. A chat turn (`app/runtime.py`) and the live demo
page load once, with no wait, inside the turn's 20 s deadline: a missing adapter or a failed load ends
the turn, and the frontier talker answers. A chat turn never stages: by then the night has loaded the
adapter into the running container, or a container started since mounts it. `Serve` runs one container
at most (`config.MAX_CONTAINERS`): every arm shares one GPU, and on 2026-10-28 one wake started two.

A wedged server fails the night fast. After a request lost in transport, the client asks
`/v1/models` (which never reaches vLLM's engine) within 15 s (`client.LIVENESS_S`). With no answer,
that request and every request the client sends for the next 60 s (`client.DOWN_HOLD_S`) fail at once
as `ServerUnresponsive`, instead of each gate thread waiting out 300 s per request. To tell a wedge
from a slow server by hand, the probe times each step (it wakes the container, so it costs money; it
prints its plan without `--go`):

```bash
python -m anjo.sleep.tools.serve_probe                                    # the plan; sends nothing
python -m anjo.sleep.tools.serve_probe --go                               # /v1/models, /tokenize, 4 tokens
python -m anjo.sleep.tools.serve_probe --adapter p-<hash>-n<night> --go   # then load (or stage) one adapter
```

### 1.7 Training on Modal (Colab is the fallback)

`ANJO_SLEEP_TRAIN_RUNNER` picks the runner that trains Sleep's nights (`train/runners/choice.py`):
`modal`, the default (Kevin, 2026-10-28: Modal's hackathon credit first), or `colab`. The night
command, the accelerated runs and the synthetic example all read it. MIND's training (Tasks 17 and
17b: replay's thinker and the APD runner) stays on Colab and never reads it.

Every night trains `sft_dpo` (SFT, then a DPO pass on the night's (best, worst) practice pairs) unless `ANJO_SLEEP_OBJECTIVE=sft`; a night whose practice kept no usable pair trains `sft` instead, and its log records the objective used and the pair count, never the words. MIND's thinker keeps `sft` and never reads the setting. The DPO learning rate is 5e-6 (beta 0.1), ten times the spec's full-fine-tune 5e-7, which is inert on LoRA weights; the receipt counts the pairs actually trained.

| Setting | Value |
|---|---|
| `ANJO_SLEEP_TRAIN_RUNNER` | `modal` (the default) or `colab`; any other value stops the night before it starts |
| `ANJO_SLEEP_OBJECTIVE` | `sft_dpo` (the default) or `sft`; any other value stops the night, the accelerated run and the example before they start |
| App, function | `anjo-sleep-train`, `train` (`train/modal_app.py`); the runner calls the deployed function by name |
| GPU, limits | one A100-80GB; Modal's hard timeout of 2 h (§12.1's training cap), a 15-minute container start limit, no retries; the training script gets the job's timeout minus 21 minutes (the start, the return and a cancel's margin) |
| Volumes | `anjo-sleep-weights` mounted read-only (Glimmer at the pinned revision), `anjo-sleep-adapters` read-write |
| Image | Debian slim, Python 3.12, exactly `colab_train.TORCH` and `TRAIN_PACKAGES`, the list the Colab runner installs |
| Secrets | none: no Hugging Face token and nothing from `.env`; Hugging Face runs offline in the container |
| Rate, pool | $2.50 an hour (`A100_USD_PER_HOUR`, Modal's A100-80GB list price, checked 2026-10-27) on `modal_usd` |
| Call records | `<data dir>/sleep/modal_calls/<adapter>.json`: an adapter name and a Modal call id, nothing else |

Each job holds its whole hard timeout at that rate ($5.00 for the 2 h job) before the call starts,
then settles the seconds the call ran, never more than the hold: the runner waits only for the
time left less a minute's margin to cancel, so a call that times out never leaves an overage that
would stop `modal_usd`, serving included. Modal also bills the container's CPU and memory beside the
GPU, which the hold doesn't count, so compare the first nights on Modal's usage page with the ledger.
In the container Glimmer loads from the weights volume at the revision the app was deployed with (a
recipe without that revision is refused), the adapter is held to the upload's checks and its
rendering, and it lands on the adapters volume under its name, so nothing downloads or uploads. A
call that may still run is cancelled when its job fails, times out or is interrupted; a cancel that
fails settles the whole hold and stops the night. A run killed hard leaves its call record, and the
next night's sweep, on either runner, cancels that call before it settles any hold. A delete cancels the person's running
call and, whenever this host has trained on Modal, removes their adapters from the adapters volume
(RF2).

Deploy once, with the Python that has `pip install -e ".[modal]"`, from the repo root. The weights
volume must hold the pinned revision even while Glimmer serves from SageMaker (§1.6's
`download_weights`, skipped if it ran already):

```bash
ANJO_SLEEP_GLIMMER_REVISION=<commit> modal run anjo/sleep/serving/modal_app.py::download_weights
ANJO_SLEEP_GLIMMER_REVISION=<commit> modal deploy anjo/sleep/train/modal_app.py
ANJO_DATA_DIR=<data dir> python -m anjo.sleep.train.smoke --runner modal         # the plan: spends nothing
ANJO_DATA_DIR=<data dir> ANJO_SLEEP_GLIMMER_REVISION=<commit> \
    python -m anjo.sleep.train.smoke --runner modal --run                         # paid, about $2.50 held
modal volume rm anjo-sleep-adapters /<smoke adapter> --recursive                 # the line the smoke prints
```

The smoke trains a fresh synthetic smoke person on the smoke dataset with the night's own recipe
(`sft_dpo` from cold) under a one-hour timeout, so it holds $2.50. Without `--run` it prints the
job and the hold, and holds, starts and spends nothing.

Known limits, accepted (review of 2026-10-28):

- The call record is written once the spawn returns, so a run killed between the two leaves a call
  no sweep can see; Modal's own 2 h timeout ends it, and the sweep settles its hold at the full bound.
- The synthetic example keeps its ledger and call records in a fresh temporary folder, so no night
  sweeps a call a killed example run left; Modal's timeout ends that one too.
- Holds count the A100 only, not the CPU and memory Modal bills beside it (above).

**The Colab fallback.** `ANJO_SLEEP_TRAIN_RUNNER=colab` puts training back on Colab as it ran
before (`runners/colab.py`, held on `colab_cu`), and the adapter is uploaded to the serving backend
again. Switch between nights, not during one. Every night sweeps both runners' leftovers before it
settles their holds, whichever runner it trains on: Modal's recorded calls are cancelled, and Sleep's
Colab sessions are stopped wherever the Colab CLI is installed.

### The Oct 28 checks (record the results here)

0. **Text-only serve (seen live on Modal).** vLLM 0.28.0 with `--enable-lora` crashed at start-up in
   `lora_shrink` (`assert token_lora_mapping.size(0) == M`): the profile run's dummy image ran
   Glimmer's vision tower, which had LoRA wrappers. `serving.config.vllm_command` now passes
   `--language-model-only`, so Modal, the SageMaker shim and the APD Colab session all skip the tower.
   The next start-up failure was FlashInfer's sampler JIT (`Could not find nvcc`) in warm-up; `vllm_env()`
   sets `VLLM_USE_FLASHINFER_SAMPLER=0` (also in the shim's env) so the native sampler runs.

1. **Glimmer loads on the L40S in FP8, with vLLM 0.28.0.** `deploy up` reaching InService proves the
   load; record vLLM's "maximum concurrency" line from the container log in §1.5. If FP8 doesn't
   fit, the fallback is BF16 on an A100-80GB, which on SageMaker is a larger instance type (and a
   quota request), so try Modal's `BF16_A100` first.
2. **The adapter loads on the FP8 server.** Load one QLoRA adapter that Task 5 trained, reloaded
   with bf16 dtype and `bnb_4bit_quant_storage` bf16 (Meta's post-training page), and confirm that
   applying it changes the output. If it doesn't load, serve merged checkpoints (`merge_and_unload()`)
   at the same precision, and record that in the night log and the README.
   **Result (Modal, 2026-10-28):** the adapter loads on the FP8 server via explicit load
   (`/v1/load_lora_adapter`) and changes the output (same seed). Asked for by name with the
   filesystem resolver on, it was a 404 every time, so the client now loads each adapter itself (§1.6).
3. **FP8 against bf16 on the general set.** Run Task 5's general gate set on both precisions. If
   FP8 loses more than the gate's own margin, switch `SERVING` to `BF16_A100`.
4. **Tool calls.** Chat turns run without tools: the runtime writes the reply's header itself
   (`/tokenize`, then `/v1/completions`) so the tokens served are the tokens trained. A tool-call
   parser (`ANJO_SLEEP_TOOL_CALL_PARSER`) only changes the server's chat route, which no turn uses.
5. **Control tokens.** Read Glimmer's special tokens from the pinned `tokenizer_config.json`, and
   fix `config.NAMED_SPECIAL_TOKENS` if they differ; `serving.framing` breaks every one of them in
   what Glimmer is sent or trained on, and the train smoke checks a hostile row end to end.
6. **Wake time**, from 0 copies to the first answer (`python -m anjo.sleep.serving.wake --wake`
   prints it), and its parts from the container log and `status`: instance provisioning, image
   pull, weights copy, vLLM load. Nothing of this is measured yet.
7. **The idle net.** Wake, then leave it idle: `status` should show the alarm in ALARM and the copy
   count at 0 within about 32 minutes, and a wake right after should not be undone. If the metric
   math misbehaves, replace the alarm with a target-tracking policy on
   `SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution` and record it here.
8. **No words in the logs.** The container's CloudWatch log holds request lines and vLLM's own
   lines; confirm vLLM logs no prompts at 0.28.0's defaults.

| Run | Backend | Wake time | Parts |
|---|---|---|---|
| | SageMaker | | |
| | Modal | | |

The wake time sets the live page's copy (Task 13), which is `COPY-PENDING(kevin)`.

## 2. The dev host

**Machine.** An e2-medium in asia-east1, CPU only, paid by GCP credits. It runs no model
weights. Kevin adds the dev subdomain's A record at Porkbun, pointing at the VM's static IP.
Caddy gets and renews the TLS certificate for that name.

**Anjo.** Anjo runs in production mode with its own data dir and its own secret, never the live
server's. The app listens on loopback, and only Caddy is public. Flagged turns go to Glimmer on
SageMaker (or Modal, the fallback). Everyone else, and every fallback, uses the existing talker on
Vertex (the VM's service account provides the credentials; the same service account's ID token is
the VM's AWS identity, §1.4).

The environment file is `/etc/anjo-sleep/env` (mode 600, owned by the service user). It holds
these names; their values never enter the repo:

| Name | Meaning |
|---|---|
| `ANJO_ENV` | `production` |
| `ANJO_DATA_DIR` | this host's own data dir |
| `ANJO_SECRET` | this host's own secret, at least 32 characters |
| `ANJO_BIND_HOST`, `PORT` | `127.0.0.1` and the port Caddy proxies to |
| `ANJO_SLEEP_HOST` | `1`: every sleep code path checks it first |
| `ANJO_SLEEP_ENABLED`, `ANJO_SLEEP_FORCED_USER_IDS` | the `sleep` capability's switch, and the flagged accounts (Kevin's dev account, the synthetic people) |
| `ANJO_CLOUD_TALKER_CHOICE_ENABLED` | `1`, so the open-model talker can be chosen |
| `ANJO_ANJO_MODEL_ENABLED` | `1`: the open-model route's kill switch. Its gates (hosts, consent, budget) run before the override, so `0` stops every flagged turn reaching Glimmer |
| `ANJO_SLEEP_BACKEND`, `ANJO_SLEEP_S3_BUCKET`, `AWS_ROLE_ARN`, `AWS_WEB_IDENTITY_TOKEN_FILE`, `AWS_ROLE_SESSION_NAME`, `AWS_DEFAULT_REGION` | Glimmer on SageMaker and the VM's keyless AWS identity (§1.4) |
| `ANJO_SLEEP_MODAL_URL`, `ANJO_SLEEP_MODAL_TOKEN` | only with `ANJO_SLEEP_BACKEND=modal`: the `Serve.web` URL, and the same value as the Modal secret's `VLLM_API_KEY` |
| `ANJO_SLEEP_GLIMMER_REVISION`, `ANJO_SLEEP_TOOL_CALL_PARSER` | the same values as the deploy (the revision is the pin above), so the pin matches what is served |
| `ANJO_SLEEP_INSTANCE_TYPE` | `ml.g6e.xlarge` (default) or `ml.g6e.2xlarge`; read by `deploy` only, which the `--instance-type` option overrides |
| `ANJO_SLEEP_PRICES` | `spark_in,spark_out,train_gpu_hour,serve_gpu_hour`; the first two are Muse Spark 1.3's standard list prices, `1.25,4.25` (Meta's pricing page, checked 2026-10-28; Wonder's `4.00/20.00` is fixed in `homework/agent.py`); the last follows the chosen type: `2.61` on g6e.xlarge, `2.81` on g6e.2xlarge |
| `VERTEX_MIN_CONSENT_VERSION` | `6`: the AI consent version the Gemma talker's gate asks for. Without it the consent gate fails closed and a flagged turn never reaches the talker |
| `ANJO_SLEEP_CONCURRENCY` | optional: how many of a night step's model calls run at once in practice, the collapse and safety gates and the morning run, 1 to 16; `8` when unset, `1` for one at a time as before Oct 28 (`anjo/sleep/parallel.py`). The general gate, log-probabilities only, keeps the server's 16. |
| `GCP_PROJECT`, `GCP_LOCATION` | the frontier talker's Vertex project |
| `RESEND_API_KEY` | needed to create password accounts: production mode refuses password registration without email |

**Accounts.**

- Kevin's dev account is created, flagged through `ANJO_SLEEP_FORCED_USER_IDS`, and set to the
  open-model talker (`PUT /api/talker {"choice": "gemma"}`). It is offered to flagged accounts
  only.
- Each synthetic person's account is flagged the same way and then marked synthetic, once:

  ```bash
  ANJO_SLEEP_HOST=1 python -m anjo.sleep.app.synthetic <user_id> ...
  ```

  Only the operator can do this. No account can mark itself synthetic over HTTP.
- A fresh account (Kevin's or a synthetic person's) needs two more steps before it can reach the
  Gemma talker. Do them as that account, with its own signed-in token, and keep the token out of the
  repo and the shell history:
  1. Accept AI consent version 6: `POST /api/ai-consent {"version": 6}`. The host must run with
     `VERTEX_MIN_CONSENT_VERSION=6` (the env table above); without it the consent gate fails closed.
  2. Choose the open-model talker: `PUT /api/talker {"choice": "gemma"}` (a synthetic person needs this
     too, not only Kevin's account).
  Restart the service after changing `VERTEX_MIN_CONSENT_VERSION`.

**The dev API** is mounted only on this host:

- `POST /api/sleep/dev/sessions {"cleared": true}` opens the account's live session and marks it
  cleared. Only a flagged, synthetic account may do this; anyone else gets a 403.
- The simulator then posts to `/api/chat/<session_id>/message` and ends the session with the
  existing route.
- `GET /api/sleep/dev/sessions/<id>` reports whether a session is cleared.

**Units** (systemd):

```ini
# /etc/systemd/system/anjo-sleep.service
[Unit]
Description=Anjo (sleep dev host)
After=network-online.target
[Service]
User=anjo
EnvironmentFile=/etc/anjo-sleep/env
WorkingDirectory=/opt/anjo-sleep
ExecStart=/opt/anjo-sleep/.venv/bin/anjo-dashboard
Restart=on-failure
[Install]
WantedBy=multi-user.target

# /etc/systemd/system/anjo-sleep-day.service
[Unit]
Description=Anjo Sleep: Teo's simulated day (wakes Glimmer first)
[Service]
Type=oneshot
User=anjo
EnvironmentFile=/etc/anjo-sleep/env
WorkingDirectory=/opt/anjo-sleep
ExecStart=/opt/anjo-sleep/.venv/bin/python -m anjo.sleep.night.daily --phase day --persona film_founder
TimeoutStartSec=1h

# /etc/systemd/system/anjo-sleep-day.timer
[Unit]
Description=Anjo Sleep day at 14:00 Taipei
[Timer]
OnCalendar=*-*-* 14:00:00 Asia/Taipei
[Install]
WantedBy=timers.target

# /etc/systemd/system/anjo-sleep-night.service
[Unit]
Description=Anjo Sleep nightly run, then the morning probe run
[Service]
Type=oneshot
User=anjo
EnvironmentFile=/etc/anjo-sleep/env
WorkingDirectory=/opt/anjo-sleep
ExecStart=/opt/anjo-sleep/.venv/bin/python -m anjo.sleep.night.daily --phase night --persona film_founder
TimeoutStartSec=5h

# /etc/systemd/system/anjo-sleep-night.timer
[Unit]
Description=Anjo Sleep at 02:00 Taipei
[Timer]
OnCalendar=*-*-* 02:00:00 Asia/Taipei
Persistent=true
[Install]
WantedBy=timers.target

# /etc/systemd/system/anjo-sleep-aws-token.service
[Unit]
Description=Anjo Sleep: the VM's Google ID token for its AWS role (no AWS key)
After=network-online.target
[Service]
Type=oneshot
User=anjo
StateDirectory=anjo-sleep-aws
StateDirectoryMode=0700
WorkingDirectory=/opt/anjo-sleep
ExecStart=/opt/anjo-sleep/.venv/bin/python -m anjo.sleep.serving.sagemaker.token_refresh --out /var/lib/anjo-sleep-aws/token
TimeoutStartSec=1min

# /etc/systemd/system/anjo-sleep-aws-token.timer
[Unit]
Description=Refresh the AWS web identity token every 10 minutes (each lasts an hour)
[Timer]
OnBootSec=30s
OnUnitActiveSec=10min
[Install]
WantedBy=timers.target
```

Enable the token timer first (`systemctl enable --now anjo-sleep-aws-token.timer`), and add
`After=anjo-sleep-aws-token.service` to the app, day and night services. A free check that the role
works, from the VM: `python -c "import boto3; print(boto3.client('sagemaker').describe_inference_component(InferenceComponentName='anjo-sleep-glimmer-ic')['RuntimeConfig'])"`.

The night runs as its own process, so a long night can't take down the server.

**The real-time schedule** (`anjo/sleep/night/daily.py`). Teo (`film_founder`, the demo's hero and the default `--persona`) lives in real time from
2026-10-28: one simulated day per real day, then the night, then the morning probe run.

- `--phase day` (14:00 Taipei) plays that day's index (days since 2026-10-28) through the dev API,
  waking Glimmer first. If Glimmer can't be woken the day does not start and the unit fails with
  `glimmer_not_ready`; it is never counted as a short day.
  The day timer has no `Persistent=`: a day that was missed stays missed, and the night records it.
  After the day, the phase writes a marker `$ANJO_SLEEP_DATA_DIR/sleep/daily/<persona>-<date>.json`
  (persona, the date the day was meant for, day index, status `ok`/`short`/`failed`, the failure
  code, failed turns, and `failure_detail`, the code of the turn that ended a short day) and exits 1 unless the day ran whole, a mid-day Glimmer death included.
  `--for-date YYYY-MM-DD` runs it by hand for another date (default: today, Taipei); the marker
  carries that date.
- `--phase night` (02:00 Taipei) runs the night for the previous Taipei calendar day (or, with
  `--for-date D`, for day D: not in the future, not before the life starts), with MIND's inquiry
  only under `--mind` or `ANJO_SLEEP_DAILY_MIND=1` in the env file (default off), as its own
  virtual date (the date it was lived on). It takes the night lock, sweeps, wakes Glimmer and trains
  exactly as `python -m anjo.sleep.night` does, and then runs the morning probe run on the
  checkpoint that serves, in the same process. It first reads the marker of the day it closes: a
  missing, short or failed day fails the night at `simulate` (`day_missing`, `short_day` or the
  day's failure code), and nothing is scored. A failed night is never scored. This timer keeps
  `Persistent=true`: the night lock makes a catch-up safe.
- The account id is never in the code: `ANJO_SLEEP_DAILY_USERS=film_founder=<account id>` in the env file.
  The env file also holds `ANJO_SLEEP_HOST=1`, the backend's names (§1.4, or Modal's URL and token),
  `ANJO_SLEEP_DEV_URL`, `ANJO_SLEEP_SIM_TOKEN` and the data dir.
- Each phase ends with Glimmer asleep (`backend.phase`): the day sets the copy count to 0 when it
  ends, and the night phase once, after the morning run.

By hand, one night is `python -m anjo.sleep.night --wiring anjo.sleep.app.night:build --night <date>
--user <id> --persona <name>`, with `--simulate anjo.sleep.app.sim_transport:simulate --day <i>` to
play a day first. An accelerated run adds `--morning anjo.sleep.app.morning:build` (see
`sim/accelerated.py`).

## Waking Glimmer, and keep-warm windows

Glimmer sits at 0 copies on SageMaker between uses (on Modal it scales to zero after 600 idle
seconds), and a cold start of the 30B model takes many minutes, while the dev host waits
`runtime.REPLY_DEADLINE_S` (20 s) for a reply. In a cleared session only Glimmer answers, so the
first turn after an idle spell would fail. So everything that needs Glimmer wakes it first
(`backend.from_env().wake()`: on SageMaker it sets one copy and polls `GET /v1/models` through the
endpoint under a hard total of 1,200 s, §1.2; on Modal it polls with the bearer token under 600 s),
and puts it back to sleep when it ends:

- **A simulated day** wakes before its first turn. If the wake fails the day does not start, and
  the night records a `glimmer_not_ready` failure at `simulate`: an error, not a short day.
- **A night** wakes before its first serving step (practice) and again right before the gates,
  since training outlasts the idle window (Modal's 600 s, SageMaker's 30-minute idle net;
  nothing pings during training: that would hold a $2.61/h ($2.81/h on g6e.2xlarge) GPU for hours). A wake that fails or raises fails that person with `glimmer_not_ready` at
  `practice` or `gates`. `summary.json` records `wake_seconds` and `gates_wake_seconds`. **A morning
  run** wakes before its first arm call and records `wake_seconds` in its score file.
- **Demos:** run `python -m anjo.sleep.serving.wake --wake` a few minutes before, and it prints how
  many seconds the wake took. It leaves Glimmer up for the demo (unless the wake failed); on
  SageMaker the idle net takes it down 30 minutes after the last call, or run
  `python -m anjo.sleep.serving.sagemaker.deploy stop` (operator) right after.

**Keep-warm windows** are for live demos and judging, where a cold start in front of people is not
acceptable:

```bash
python -m anjo.sleep.serving.wake --keep-warm-minutes 90   # wakes, then pings every 240 s
```

Run it as the service user with the environment file loaded, as the app runs:
`sudo -u <service user> bash -c 'set -a; . /etc/anjo-sleep/env; set +a; exec python -m anjo.sleep.serving.wake --keep-warm-minutes 90'`.
Each answered ping writes `$ANJO_DATA_DIR/sleep/serving/warm_window.json`, which the live demo page
reads to answer inside the window on either backend (on Modal it never pings, since a request would
start a container) and to charge none of it to its own warm-time cap. Without `ANJO_DATA_DIR` the
window refuses to start, before anything wakes.

It uses the backend `ANJO_SLEEP_BACKEND` names and never prints a credential. A window is capped at
180 minutes, and on SageMaker it ends by setting the copy count to 0, after an error or Ctrl-C too.
If the process dies, the pings stop and the idle net (SageMaker, 30 minutes) or Modal's own window
(600 s) takes Glimmer down, so nothing stays up unbounded. The L40S costs about **$2.61 an hour on
SageMaker** ($2.81 on g6e.2xlarge) (about $2 on Modal) while it is up, so **each window needs Kevin's OK**.

**Caddy** (`/etc/caddy/Caddyfile`):

```
<dev-subdomain> {
    reverse_proxy 127.0.0.1:<PORT>
}
```

**Checks after setup:**

- `systemctl list-timers 'anjo-sleep-*.timer'` shows the day at 14:00 and the night at 02:00 Taipei.
- The dev app signs in over HTTPS.
- A flagged turn is served by Glimmer.
- With Glimmer asleep (`python -m anjo.sleep.serving.sagemaker.deploy stop`, or
  `modal app stop anjo-sleep-serve` on the fallback), the same turn is answered by the frontier
  talker, and its sidecar record says `fallback`.
- `systemctl list-timers anjo-sleep-aws-token.timer` shows the token refresher, and the token file
  is mode 600.

## What a flagged turn does

1. The chat route's crisis floor runs first. A severe or harm message gets its fixed protocol,
   and no model sees it.
2. The open-model route's gates run first: the `anjo_model` kill switch, the pinned hosts, the
   person's processor consent and the budget. Only a route that passes them all is changed: the
   route override (`runtime.sleep_route`) then pins a flagged person who chose the open-model
   talker to `sleep_modal` (the host keeps its first name, whichever backend serves). Consent is
   checked on the existing Vertex and OpenRouter entries, since the consent registry names no
   Modal or SageMaker processor. Everyone else's route is unchanged.
3. The prompt hook (`prompt.sleep_turn_hook`) refuses any turn with a crisis tier or in a session
   in distress, so the frontier talker answers it. Otherwise it adds context C. In a cleared
   session it builds the cleared context instead.
4. `SleepRuntime` refuses a request whose latest words trip the floor. It sends the adapter
   name, or the base model, to Glimmer (SageMaker's endpoint, or Modal), and refuses an adapter
   trained under another rendering (`serving.framing.rendering`). It records the reply as served
   once Glimmer's first word is shown.
5. Any error before that (asleep, stopped, refused, broken off, or no reply within the 20 s
   deadline) falls back to the frontier talker (RF3), and the sidecar says `fallback`.

Each turn's context ancestry, and whether Glimmer served it, are kept in
`data/users/<id>/sleep/reply_ancestry.jsonl` (encrypted, exported, erased with the account).

## MIND's experiment on Fireworks (Tasks 17 and 17b)

MIND's experiment (`python -m anjo.sleep.mind.apd` and `python -m anjo.sleep.mind.replay`) calls
Spark Open through one provider per experiment (`anjo/sleep/mind/spark_provider.py`). It runs
from the operator's machine, not this host, so the dev host's environment file needs neither name
below.

| Name | Meaning |
|---|---|
| `ANJO_FIREWORKS_API_KEY` | Fireworks' API key, read at each call (`anjo/sleep/fireworks_api.py`); never uploaded to the Colab VM |
| `ANJO_SLEEP_MIND_SPARK` | `fireworks` (the default) or `baseten`; any other value stops the command before it starts |

- **The model.** `accounts/fireworks/models/muse-spark-open`, at
  `https://api.fireworks.ai/inference/v1/chat/completions`. Its exact id is cleared for training
  beside Baseten's Spark Open id (`provenance.SPARK_OPEN_IDS`); hosted Muse Spark stays uncleared.
- **The prices.** Fireworks' list prices for Spark Open, checked 2026-10-28: $1.25 per million
  input tokens, $4.25 per million output tokens, $0.15 per million cached input tokens. `--prices`
  takes Spark Open's rates as its first two values, the same for either provider:
  `--prices 1.25,4.25,<train_gpu_hour>,<serve_gpu_hour>`. The ledger prices every input token at
  the full rate, cached or not, so it never counts less than Fireworks bills.
- **The budget.** Every Spark Open call of the experiment on Fireworks, the screen's included, is
  held first on `fireworks_mind_usd`, capped at $90 ($10 under the $100 credit). Fireworks shares
  no credit with Sleep, so a MIND night there waits only on other MIND nights. With `baseten`,
  the calls hold `baseten_mind_usd` as before. `--live` prints the provider, the model and the
  pool before anything runs.
- **The run's ceiling.** `generate`, `run`, `timing` and `judge` take `--max-usd` (default $20).
  Before each new generation attempt, life-arm run or judge call, the command sums what its own
  holds have settled (not the pools' totals), and once that reaches the ceiling it starts nothing
  more and exits with code 5, saying how far it got and what it spent. A unit already running
  finishes under its own hold, so a run can pass the ceiling by the units in flight (with
  `--parallel N`, up to N life-arms). After each life and each life-arm it prints what it has
  settled per life and per arm-day: run `--only 1 generate` first and extrapolate (`--lives`
  sets only the held-out draw).
- **One provider per pilot.** The data dir's `serving_pin.json` records `spark_provider` and
  `spark_model` at the first command that calls Spark Open (`generate`, `run`, `timing`,
  `intervene`, `kill-resume`), and refuses a later command on the other provider. It also
  refuses a data dir whose lives another model wrote before the pin existed. To change provider,
  start a new data dir.
- **Sleep's night stays on Baseten.** Practice, extraction, the brief, the simulator and the
  night's own MIND step never read `ANJO_SLEEP_MIND_SPARK`.

# Anjo Sleep dev host and Glimmer serving (Task 4)

This runbook names settings, never their values. Secrets live in the Modal secret and in the dev
host's environment file, and nowhere in the repo. Everything here that starts paid GPU time or
touches DNS is run by the coordinator or Kevin, on Kevin's go.

There are three parts:

1. **Glimmer on Modal** (`anjo/sleep/serving/modal_app.py`): it only serves. Training runs on Colab (Task 5).
2. **The dev host**: a small CPU-only GCP VM that runs Anjo in production mode with `ANJO_SLEEP_HOST=1`.
3. **The dev app**: the existing mobile app, built in Xcode with its own bundle id and
   `EXPO_PUBLIC_API_URL` pointing at the dev host. Nothing in `mobile/` changes for it.

## 1. Glimmer on Modal

**What runs.** vLLM's OpenAI-compatible server serves `meta-models/Muse-Glimmer-30B` (Apache 2.0)
at the pinned commit. The weights sit on a Modal volume. LoRA is on, and each person's adapter
`p-<hash>-n<night>` loads by name on its first request, so the base model and every personal arm
share one GPU.

| Setting | Value |
|---|---|
| App | `anjo-sleep-serve` |
| GPU and precision | one L40S, with weights quantized to FP8 when vLLM loads them (`config.SERVING = FP8_L40S`) |
| Fallback | one A100-80GB in bf16: set `SERVING = BF16_A100` in `serving/config.py` and redeploy |
| Volumes | `anjo-sleep-weights` (`/weights`), `anjo-sleep-adapters` (`/adapters`) |
| Secrets | `anjo-sleep-serve`, holding only `VLLM_API_KEY` (the dev host's bearer token), for `Serve` only. `download_weights` holds no secret: the Glimmer repo isn't gated. If a token is ever needed, put `HF_TOKEN` in a second secret and name it in `ANJO_SLEEP_HF_SECRET`; only `download_weights` gets it |
| Idle | scales to zero after 600 s; each container serves for at most 3 h; every function has a timeout |
| Context | 32,768 tokens per request (`MAX_MODEL_LEN`), 4,096 output |
| LoRA | rank 32 at most, 8 adapters on the GPU at once, 16 kept on the CPU (see the sizing below) |
| Reply deadline | the dev host waits 20 s for Modal's reply and for each read after (`runtime.REPLY_DEADLINE_S`), then the frontier talker answers |

Every arm serves at the same precision: the base and the adapters run on the same server.

**Deploy-time settings** (read by `ServeConfig.from_env` and baked into the image):

- `ANJO_SLEEP_GLIMMER_REVISION`: the 40-character Glimmer commit, required. The pin is
  `cec94fcfd6512f2cfe708138c32533b346ea4f69` (`meta-models/Muse-Glimmer-30B`, Apache-2.0).
  Loading `modal_app.py` without it raises, so no deploy or run can start unpinned.
- `ANJO_SLEEP_HF_SECRET`: the Modal secret holding `HF_TOKEN`, for the download only. Leave it
  unset: the repo isn't gated.
- `ANJO_SLEEP_TOOL_CALL_PARSER`: vLLM's tool-call parser for Glimmer. If it's unset, flagged turns run without tools.
- `ANJO_SLEEP_GPU_SNAPSHOT=1`: try Modal's GPU memory snapshot to cut the cold start.
- `ANJO_SLEEP_VLLM_VERSION`: the vLLM release, if not the pinned default.

**Commands** (paid; the coordinator runs them on Kevin's go):

```bash
export ANJO_SLEEP_GLIMMER_REVISION=cec94fcfd6512f2cfe708138c32533b346ea4f69   # the pin
modal secret create anjo-sleep-serve VLLM_API_KEY=...                        # once
modal run anjo/sleep/serving/modal_app.py::download_weights
modal deploy anjo/sleep/serving/modal_app.py
modal run anjo/sleep/serving/modal_app.py::upload_adapter --folder <dir> --name p-<hash>-n<night>
modal app stop anjo-sleep-serve                                              # the RF3 drill
```

The deploy prints the `Serve.web` URL. That URL becomes the dev host's `ANJO_SLEEP_MODAL_URL`.

Before upload, `upload_adapter` checks the folder:

- it holds exactly `adapter_config.json` and `adapter_model.safetensors`;
- it is a PEFT LoRA of `meta-models/Muse-Glimmer-30B` with rank 32 or less;
- its name is not already on the volume. An adapter is never overwritten, so a pinned night's
  adapter can't change; Modal's upload also runs with `force=False`.

A container that is already running sees a new adapter only after its next start. That happens
after 10 idle minutes, or after `modal app stop`.

**LoRA sizing.** About 6 to 8 adapters are live at once: Kevin, Tessa, three lives and the
donors for arm 3x. So `--max-loras 8` and `--max-cpu-loras 16`. vLLM queues a request for a
ninth adapter rather than failing it, and the CPU cache swaps it in without reading the volume.
Task 5 trains rank 16, and Meta's recipe goes up to 32, so `--max-lora-rank 32`.

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

### The Oct 28 checks (record the results here)

1. **The adapter loads on the FP8 server.** Load one QLoRA adapter that Task 5 trained, reloaded
   with bf16 dtype and `bnb_4bit_quant_storage` bf16 (Meta's post-training page), and confirm that
   applying it changes the output. If it doesn't load, serve merged checkpoints (`merge_and_unload()`)
   at the same precision, and record that in the night log and the README.
2. **FP8 against bf16 on the general set.** Run Task 5's general gate set on both precisions. If
   FP8 loses more than the gate's own margin, switch `SERVING` to `BF16_A100`.
3. **Tool calls.** Find the parser that reads Glimmer's tool calls and set
   `ANJO_SLEEP_TOOL_CALL_PARSER` on both Modal and the dev host. If no parser reads them, leave it
   unset; turns then run without tools, and the README says so.
4. **Control tokens.** Read Glimmer's special tokens from the pinned `tokenizer_config.json`, and
   fix `runtime.CONTROL_TOKENS` if they differ.
5. **Wake time.** Measure the cold start from a stopped app to the first token:

| Run | GPU snapshot | Wake time |
|---|---|---|
| | off | |
| | on | |

The wake time sets the live page's copy (Task 13), which is `COPY-PENDING(kevin)`.

## 2. The dev host

**Machine.** An e2-medium in asia-east1, CPU only, paid by GCP credits. It runs no model
weights. Kevin adds the dev subdomain's A record at Porkbun, pointing at the VM's static IP.
Caddy gets and renews the TLS certificate for that name.

**Anjo.** Anjo runs in production mode with its own data dir and its own secret, never the live
server's. The app listens on loopback, and only Caddy is public. Flagged turns go to Glimmer on
Modal. Everyone else, and every fallback, uses the existing talker on Vertex (the VM's service
account provides the credentials).

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
| `ANJO_SLEEP_MODAL_URL`, `ANJO_SLEEP_MODAL_TOKEN` | the `Serve.web` URL, and the same value as the Modal secret's `VLLM_API_KEY` |
| `ANJO_SLEEP_GLIMMER_REVISION`, `ANJO_SLEEP_TOOL_CALL_PARSER` | the same values as the Modal deploy (the revision is the pin above), so the pin matches what Modal serves |
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

# /etc/systemd/system/anjo-sleep-night.service
[Unit]
Description=Anjo Sleep nightly run
[Service]
Type=oneshot
User=anjo
EnvironmentFile=/etc/anjo-sleep/env
WorkingDirectory=/opt/anjo-sleep
ExecStart=/opt/anjo-sleep/.venv/bin/python -m anjo.sleep.night
TimeoutStartSec=4h

# /etc/systemd/system/anjo-sleep-night.timer
[Unit]
Description=Anjo Sleep at 02:00 Taipei
[Timer]
OnCalendar=*-*-* 02:00:00 Asia/Taipei
Persistent=true
[Install]
WantedBy=timers.target
```

The night runs as its own process, so a long night can't take down the server.
`python -m anjo.sleep.night` is Task 5's.

**Caddy** (`/etc/caddy/Caddyfile`):

```
<dev-subdomain> {
    reverse_proxy 127.0.0.1:<PORT>
}
```

**Checks after setup:**

- `systemctl list-timers anjo-sleep-night.timer` shows the next run at 02:00 Taipei.
- The dev app signs in over HTTPS.
- A flagged turn is served by Glimmer.
- After `modal app stop anjo-sleep-serve`, the same turn is answered by the frontier talker, and
  its sidecar record says `fallback`.

## What a flagged turn does

1. The chat route's crisis floor runs first. A severe or harm message gets its fixed protocol,
   and no model sees it.
2. The open-model route's gates run first: the `anjo_model` kill switch, the pinned hosts, the
   person's processor consent and the budget. Only a route that passes them all is changed: the
   route override (`runtime.sleep_route`) then pins a flagged person who chose the open-model
   talker to `sleep_modal`. Consent is checked on the existing Vertex and OpenRouter entries,
   since the consent registry names no Modal processor. Everyone else's route is unchanged.
3. The prompt hook (`prompt.sleep_turn_hook`) refuses any turn with a crisis tier or in a session
   in distress, so the frontier talker answers it. Otherwise it adds context C. In a cleared
   session it builds the cleared context instead.
4. `SleepRuntime` refuses a request whose latest words trip the floor. It sends the adapter
   name, or the base model, to Modal. It records the reply as served once Glimmer's first word
   is shown, or, in a tool round whose words are held back, once the round ends cleanly.
5. Any error before that (cold, stopped, refused, broken off, or no reply within the 20 s
   deadline) falls back to the frontier talker (RF3), and the sidecar says `fallback`.

Each turn's context ancestry, and whether Glimmer served it, are kept in
`data/users/<id>/sleep/reply_ancestry.jsonl` (encrypted, exported, erased with the account).

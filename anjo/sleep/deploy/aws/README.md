# The two IAM roles for Glimmer on SageMaker

The operator account has no IAM write, so Kevin creates these two roles in the IAM console, once.
Each role is one trust policy and one inline permissions policy, pasted as they are after the
placeholders are filled in. The code only names the roles (`serving/sagemaker/settings.py`) and
never creates one.

| Placeholder | What goes there |
|---|---|
| `<ACCOUNT_ID>` | the 12-digit AWS account id |
| `<BUCKET_SUFFIX>` | the suffix chosen for the bucket `anjo-sleep-sagemaker-<BUCKET_SUFFIX>` |
| `<VM_SERVICE_ACCOUNT_UNIQUE_ID>` | the dev host VM's service account's numeric unique id (`gcloud iam service-accounts describe <email> --format='value(uniqueId)'`) |

Fill them in the console only. Nothing with a real id goes back into the repo.

## `anjo-sleep-sagemaker-exec`: what SageMaker runs the container as

- Trust: `sagemaker-exec-trust.json` (SageMaker only).
- Permissions: `sagemaker-exec-permissions.json`. It reads the staged weights and the adapters,
  pulls our image, and writes its own logs and metrics. It can't write to the bucket or invoke
  anything.

## `anjo-sleep-devhost`: what the dev host VM calls AWS as

- Trust: `devhost-trust.json`. Google's built-in provider `accounts.google.com`, so no OIDC
  provider is created. Only an ID token for the VM's own service account (`sub` and `aud`, which
  AWS maps from the token's `azp`) with our audience (`oaud`, `token_refresh.AUDIENCE`) can assume it.
- Permissions: `devhost-permissions.json`. It can invoke our endpoint's one component, read and set
  that component's copy count (wake and sleep), and list, read and write under `adapters/` in our
  bucket. It can't touch the weights, the endpoint's config, the scaling or anything else.

## The two `*` resources

AWS accepts no narrower resource for these two actions:

- `EcrSignIn`: `ecr:GetAuthorizationToken` signs in to the registry as a whole. The image pull
  itself (`PullOurImageOnly`) is limited to our repository.
- `SageMakerMetrics`: `cloudwatch:PutMetricData` takes no resource. The condition limits it to
  SageMaker's own namespaces.

## In the console

1. IAM, Roles, Create role, Custom trust policy: paste the trust file. Skip the managed policies.
   Name the role exactly as above and create it.
2. Open the role, Add permissions, Create inline policy, JSON: paste the permissions file. Name it
   after the role (for example `anjo-sleep-devhost-permissions`).
3. Leave the maximum session at 1 hour.

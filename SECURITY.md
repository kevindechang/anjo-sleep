# Security

## Reporting

Report a vulnerability privately with GitHub's **Report a vulnerability** button on this repo's
Security tab. Please don't open a public issue for it.

In scope: this repo's code, and the judge site at sleep.anjo.nexus (only within its normal use: no
load testing, and no access to anyone else's session). The private Anjo app is out of scope here.

## How the repo handles secrets

- Every key comes from the environment (`MODAL_TOKEN_ID`, `BASETEN_API_KEY`, `ANJO_META_API_KEY`, ...).
  `.env` files, keys, weights and run data are git-ignored.
- A run with a missing key stops before anything is created or paid for, and names the variable.
- Tests are offline by construction: `anjo/sleep/tests/offline.py` refuses non-loopback traffic.
- The repo is generated from a private trunk by `anjo/sleep/tools/publish.py`. It refuses to push
  when the export crosses the published folders, and it scans the full history for secrets first.

## Supported versions

Only the latest commit on `main`.

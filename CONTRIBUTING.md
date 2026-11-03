# Contributing

Thanks for looking. A few things are unusual about this repo.

**It is generated.** `main` is exported from a private trunk by `anjo/sleep/tools/publish.py`, and
public history only fast-forwards. A pull request here is read, and an accepted change lands
through the trunk with credit, so your commit may arrive under a new hash.

**Some files are locked by hash.** Their sha256 is in the pre-registration ledger (the `prereg`
branch), so changing them breaks a registered claim:

- the files the freeze registered: probe and judge prompts, `anjo/sleep/prereg/thresholds.json`,
  `anjo/sleep/sets/general.jsonl` and `collapse.jsonl`, the draw and final report code
- `anjo/sleep/results/*/scores/final.json`
- everything under `anjo/sleep/mind/cog/heldout/`

`python -m anjo.sleep.tools.verify_claims` (after `git fetch origin prereg`) fails if any of them changed.
A new experiment gets a new registration, not an edit to an old one.

**Before a pull request:**

```bash
pip install -e ".[test]" ruff
ruff check anjo
pytest
python -m anjo.sleep.examples.synthetic --fake
```

Tests must stay offline: no keys, no network. Use the fakes in `anjo/sleep/tests/`.

**Results need simulated people.** Never add a real person's conversations, notes or adapters.

By contributing you agree that your contribution is licensed under Apache-2.0 (LICENSE).

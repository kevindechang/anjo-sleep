# Anjo Sleep

<!-- COPY-PENDING(kevin): every paragraph of this README is a first draft. -->

**It sleeps on it.** Anjo Sleep gives each person their own copy of an open model, Muse Glimmer
30B, and teaches it that person every night. During the day you just talk with Anjo. At night
her state decides what to work on: she follows up on a plan you dated, or studies a topic you
left open. She practises the kind of conversation you have, a personal adapter trains on that
practice, and three gates check the result: general skills, collapse into your field, and
safety. Only a night that passes every gate is promoted. After a night that fails or is
rejected, the last good model keeps serving.

**State decides, weights learn.** What Anjo knows about you stays in her state: the evidence
store, the brief, her targets and the receipts for every choice she made. Facts stay word for
word, so they stay exact and delete cleanly. The weights learn the language of your field and
how to talk with you, and we test whether they do with no notes in the prompt at all, against
the base model and against another person's trained weights. MIND, her inquiry process, notices
problems nobody assigned, works on them across nights, and changes a belief only on evidence.

Built for the Meta Global AI Developer Hackathon, October 27 to November 4, 2026.

## Architecture

<!-- COPY-PENDING(kevin): node labels. -->

```mermaid
flowchart LR
    subgraph existing["Existing Anjo (private, before the hackathon)"]
        appUI["Anjo app"]
        server["Anjo server: chat, memory, history"]
        crisis["Crisis check"]
        outreach["Outreach and push"]
    end

    subgraph sleep["Anjo Sleep (this repo, built during the hackathon)"]
        route["Route override"]
        glimmer["Her own Glimmer 30B on SageMaker (vLLM + LoRA)"]
        evidence["Evidence store and targets"]
        chooser["Her chooser: wants, choices, receipts"]
        mind["MIND: questions, hypotheses, findings"]
        homework["Homework (Muse Wonder)"]
        practice["Practice (Muse Spark Open)"]
        train["Nightly training (QLoRA on Modal)"]
        gates["Gates: general, collapse, safety"]
        note["Held note"]
        sealed["Sealed tests (Muse Spark judge)"]
    end

    appUI --> server
    server --> crisis
    server --> route --> glimmer
    server -- "the day's turns" --> evidence
    evidence --> chooser
    chooser --> homework
    chooser --> mind
    evidence --> practice
    homework --> practice
    practice --> train --> gates
    gates -- "safety cases" --> crisis
    gates -- "promote or reject" --> glimmer
    chooser --> note --> outreach --> appUI
    sealed -. "scores" .-> glimmer

    classDef old fill:#d9d9d9,stroke:#8c8c8c,color:#404040
    classDef new fill:#ffe082,stroke:#ef6c00,color:#1a1a1a,stroke-width:2px
    class appUI,server,crisis,outreach old
    class route,glimmer,evidence,chooser,mind,homework,practice,train,gates,note,sealed new
```

## What's new and what's existing

Everything in this repository was built during the hackathon, from October 27, 2026 on. Its
history starts there, with real commit dates.

Existing Anjo is the private app that was live before the event (tagged `pre-hackathon`). It
appears here only in grey: in the diagram, and as **existing** wherever a night log, a result
or the video shows a step that calls it, for example "existing: crisis check" or "existing:
history store". Its code isn't in this repo. The adapters in `anjo/sleep/app/` are the only
code here that calls into it, and they run only inside the private app.

The engine core, which is everything else in `anjo/sleep/`, imports no private code. It reaches
the app only through the protocols in `anjo/sleep/ports.py`, so it runs on its own.

## Run it yourself

A standalone example runs one night for one invented person, Calla, who restores wooden boats.
It needs no private Anjo code: it loads her persona and one fixed day into plain files, seals
and hashes the probes, extracts the day on Spark Open and folds it into her brief, lets her
chooser pick the night's target and prints its receipt, practises, trains a LoRA on Modal, runs
the arms on the probes and prints a score table, then serves base plus adapter and opens a chat.

**It has no crisis gate**, because Anjo's crisis case lists stay private. It says so on every
run, runs no gate of any kind and never marks a checkpoint as promoted.

First, with no keys and no cost. Everything is faked, so the replies and scores mean nothing; it
only shows the flow:

```bash
pip install -e ".[test]"
python -m anjo.sleep.examples.synthetic --fake
```

Then live, with your own keys. This pays for Baseten, the Meta API and Modal GPU time:

```bash
pip install -e ".[modal]"
export MODAL_TOKEN_ID=...              # your Modal account
export MODAL_TOKEN_SECRET=...
export ANJO_META_API_KEY=...           # the Meta Model API (the judge)
export BASETEN_API_KEY=...             # Muse Spark Open on Baseten
export ANJO_SLEEP_GLIMMER_REVISION=... # the pinned commit of meta-models/Muse-Glimmer-30B
# Glimmer's server and its training app on your Modal account, deployed once:
modal secret create anjo-sleep-serve VLLM_API_KEY=<a token you choose>
modal run anjo/sleep/serving/modal_app.py::download_weights
modal deploy anjo/sleep/serving/modal_app.py
modal deploy anjo/sleep/train/modal_app.py
export ANJO_SLEEP_MODAL_URL=...        # the web URL modal deploy prints
export ANJO_SLEEP_MODAL_TOKEN=...      # the token you chose above
python -m anjo.sleep.examples.synthetic --persona persona.json --day day1.jsonl
```

A missing key stops the run before anything is created or paid for, and names the variable. The
two file names resolve to the bundled `anjo/sleep/examples/synthetic/persona.json` and
`day1.jsonl` unless you point them at your own. `--donor <adapter>` adds arm 3x (an adapter that
another run trained), `--adapter <adapter>` reuses an adapter already on the volume and skips
practice and training (a new adapter loads only when the server container starts, so after a
run stops with "the server doesn't serve adapter ...", run `modal app stop anjo-sleep-serve`,
deploy again and re-run with `--adapter`), `--no-chat` skips the chat and `--workdir <new folder>` keeps the run's
files. The example trains and serves Glimmer with the engine's own rows, prompt framing and pins, and an
adapter is served only under the rendering it was trained under. The training and serving parts were written against Modal's documented API and run only
with your keys.

## Running the tests

```bash
pip install -e ".[test]"
pytest
```

## Licence

Apache 2.0 (see `LICENSE`). Built with Muse Spark Open; the Muse Model License and Usage Policy
are linked in `NOTICE`.

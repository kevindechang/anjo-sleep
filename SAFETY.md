# Safety and limitations

Anjo Sleep v0.1.0 is a research preview built in nine days. This page says what protects a person
today, what is switched off, and what the results do not show.

## What is switched off for the hackathon

- **The nightly safety gate.** `ANJO_SLEEP_SAFETY_GATE` is off. The gate sends no case to a model and
  runs only its deterministic floor. Every night page and log shows it as **skipped**, never as passed.
- **The standalone example has no crisis gate.** Anjo's crisis case lists stay private, so
  `python -m anjo.sleep.examples.synthetic` says so on every run, runs no gate of any kind and never
  marks a checkpoint as promoted.

## What still guards a person

- On the live site, existing Anjo's reply check guards every reply a person sees, including the
  medical output guard. It is private code and not part of this repo.
- A night is promoted only if every gate it runs passes (general skills and collapse). After a night
  that fails or is rejected, the last good model keeps serving.
- What Anjo knows about a person stays in its state, word for word, so it can be corrected and
  deleted (`anjo/sleep/ownership/`).
- The judge site has hard spend caps per night, per day and in total, and one night at a time.

## What the results do not show

- Every person in the results is simulated. None of the numbers is evidence about real people.
- MIND was scored on held-out people from the simulator it was trained on. On a differently written
  world it did worse than a coin flip until retrained.
- The pairwise judge is a model (DeepSeek V4.1 Flash), scored blind in both orders. A model judge
  can share blind spots with the models it judges.
- The adapter alone has not beaten the base model yet.

## Don't

- Don't use this for clinical, legal, financial or crisis decisions.
- Don't put personal information into the judge site. It is a demo host.
- Don't serve a trained adapter to anyone without the safety gate and a crisis check in front of it.
  An adapter trained on Muse Spark Open outputs carries the Muse Model License and its Usage Policy
  (NOTICE).

Security issues: see SECURITY.md.

"""The live demo page's engine (Task 13a, spec §2.8): the minimal step.

A visitor reads the film student's brief, confirms they are 18 or older, and compares two replies
to her, blind: arm 4 (her own weights, with context C) against the baseline set for the demo (2C
or 2F). They vote left, right, tie or neither, and only then see which side was which.

- ``words``: every word a visitor reads, and the starter scenes, in one place;
- ``limits``: the abuse limits, after the removed public demo's design;
- ``visits``: the signed visit token (18+ confirmed) and the keyed client bucket;
- ``pairs``: the blind side order, held on the server until the vote;
- ``starters``: the starter scenes' pre-generated pairs;
- ``service``: one visit's flow over those parts and the ports the app fills.

Nothing here imports private Anjo. The crisis checks, the arms' wiring and the routes are in
``anjo.sleep.app.demo_live`` and ``anjo.sleep.app.demo_api``.
"""

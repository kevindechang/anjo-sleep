"""The comparison's conventional setup: base model plus a scheduled agent (arm 2A).

The comparison asks whether MIND's notes beat an ordinary nightly agent's, so this package is that
agent and nothing more. ``AgentNight`` runs one scripted pass a night over the person's evidence and
current brief, with the inquiry's own two tools (``evidence recall``, ``research``) and its per-night
token cap, on Spark Open. It has no workspace, targets, questions or log of its own, and it carries
nothing between nights: what it reads comes from the evidence store and the brief, and what it
leaves is a note in that store which no other reader of it sees (``agent_note`` items with
``agent-`` ids, read only by the morning arm 2A).

``night.on`` is the switch: the night runs the arm only with ``ANJO_SLEEP_AGENT_BASELINE=1``, and
its paid calls hold their own worst case on ``baseten_sleep_usd``, which the ledger remaps when
Sleep's Spark Open is switched to its other provider.
"""

"""Serving her weights: Glimmer and each person's adapter behind vLLM on Modal (Task 4).

``config`` is plain data and imports only the standard library, so the Modal container can load
it without the rest of Anjo. ``modal_app`` is the deployable app; it needs the ``modal`` package,
which only the machine that deploys has.
"""

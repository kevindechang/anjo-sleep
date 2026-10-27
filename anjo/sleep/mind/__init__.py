"""Her chooser: wants, choices and receipts (spec §3.1, Task 7a).

Engine core, and pure: no clock, database, model call or environment read. ``choose.StateChooser``
is her chooser; ``rule_chooser.RuleChooser`` stays behind the same ``Chooser`` port as the labeled
fallback.
"""

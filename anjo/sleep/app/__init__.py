"""The bridge to private Anjo: the only package in ``anjo/sleep`` that imports other ``anjo.*``
modules (spec §12.1).

The adapters here implement the protocols in ``anjo.sleep.ports`` on private Anjo: the history
reader and evidence extraction, the encrypted stores, the route override and runtime, flags,
delivery, the export and delete hooks, the safety gate, decontamination, the API router, the
context builder and the chooser's policy and intentions. The engine core never imports this
package; it reaches these adapters only through the ports, so it runs without private Anjo
(the standalone example, Task 15). Their tests live in ``anjo/sleep/tests/app`` and run only in
the private repo. An import-linter contract in the private ``pyproject.toml`` pins all of this.
"""

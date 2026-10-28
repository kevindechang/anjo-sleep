"""Replay (Task 17b): Sleep trains how she investigates, on synthetic development lives only.

``sample`` finds the start states and runs the branches, ``resolve`` labels each start state once
on Spark Open, ``score`` scores the branches, ``dataset`` builds thinker-1's and thinker-shuffled's
rows, and ``compare`` runs the 17b test on the held-out lives. ``python -m anjo.sleep.mind.replay
--fake`` runs the whole path on offline fakes. Nothing here imports a submodule, so importing the
package loads no training code.
"""

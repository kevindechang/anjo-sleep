"""Anchor-protected LoRA, for ``sft_dpo_seed``'s job (``sleep_job.py``) with ``ANJO_SLEEP_PROTECT`` (2026-10-29).

Tonight's slot trains as ``A_eff = A (I - Q Q^T)``, so its update is zero on ``Q``: at every LoRA-targeted
module, a basis of the activations the forgetting anchors produce there. An applied form of null-space
projection for continual learning (AlphaEdit's projected edits; O-LoRA's and InfLoRA's orthogonal LoRA
subspaces), here on a small per-night anchor basis.

- **The basis** (``bases``): each anchor's prompt and then each of its choices after it, as the general
  gate scores an item, run through the model as the night began (eval, no gradient); the inputs every
  LoRA module's ``lora_A`` receives are kept (a prompt's rows once), and ``Q`` is their top ``RANK``
  right singular vectors, of the module's input width. The anchors are ``sets/anchors.jsonl``'s, which
  no gate scores: training never sees a gate item, and the general gate stays the measure of forgetting.
  The sampling stops at ``MAX_MINUTES`` (the receipt says how many sequences it took and how long).
- ``random`` (the ablation): the same sampling, then per module a seeded random orthonormal basis of
  the same shape in place of the anchors': the same rank and dimension, no anchor in it.
- **The projection** (``attach``): a forward hook on each ``lora_A`` takes ``(x Q)(A_slot Q)^T`` off the
  slot's outputs, the ranks past ``lead``; the leading ranks (the fold's long-term adapter) are untouched.
  ``materialize`` then writes ``A_slot (I - Q Q^T)`` into the weights, so the saved adapter is a plain
  LoRA that computes what the hooked one did, with no hook.
- **The fold** (``sleep_fold.fold`` with the bases) keeps it: the folded adapter acts on ``span(Q)`` as
  the long-term one did, and is the best rank-R cut otherwise.

Standalone, as the training script is: the standard library at the top, torch inside the functions;
Modal's image carries it beside ``sleep_job.py``, which loads it by its path.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

KINDS: Final = ("anchors", "random")
RANK: Final = 16  # the night's own rank: the fold's rank-32 cut keeps 16 for tonight beside the anchors' part
MAX_MINUTES: Final = 6.0  # the sampling's cap
EIGEN_FLOOR: Final = 1e-10  # a direction under this share of the largest holds none of the anchors' rows
ADAPTER: Final = "default"


def lora_modules(model: Any, adapter: str = ADAPTER) -> dict[str, Any]:
    """Every LoRA layer by the name its saved tensors carry (``<name>.lora_A.weight``)."""
    return {name: module for name, module in model.named_modules() if adapter in (getattr(module, "lora_A", None) or {})}


def _shared(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def sequences(anchors: Sequence[Any]) -> list[tuple[tuple[int, ...], int]]:
    """Each anchor's prompt with each choice after it, as the general gate scores them, and the position
    its new rows start at: past the longest prefix an earlier sequence ran, whose rows a causal model
    gives again unchanged. The chat template's preamble and each prompt are kept once."""
    out: list[tuple[tuple[int, ...], int]] = []
    for ids in (tuple(a.prompt) + tuple(choice) for a in anchors for choice in a.choices):
        out.append((ids, max((_shared(ids, seen) for seen, _ in out), default=0)))
    return out


def _rows(model: Any, anchors: Sequence[Any], adapter: str, cap_s: float,
          timer: Callable[[], float]) -> tuple[dict[str, list[Any]], int]:
    """Each LoRA module's input rows over the anchors, and how many sequences ran before the cap."""
    import torch

    modules = lora_modules(model, adapter)
    if not modules:
        raise ValueError("the model has no LoRA module to protect")
    rows: dict[str, list[Any]] = {name: [] for name in modules}
    start = [0]

    def keep(name: str) -> Callable[[Any, Any], None]:
        def hook(module: Any, args: Any) -> None:
            x = args[0].detach()
            x = x.reshape(-1, x.shape[-1])[start[0]:]
            rows[name].append(x.to(torch.bfloat16) if x.is_cuda else x)  # every module's rows wait at once
        return hook

    handles = [module.lora_A[adapter].register_forward_pre_hook(keep(name)) for name, module in modules.items()]
    device, was, begun, seen = model.get_input_embeddings().weight.device, model.training, timer(), 0
    model.eval()
    try:
        with torch.no_grad():
            for ids, new_from in sequences(anchors):
                if seen and timer() - begun > cap_s:
                    break
                start[0] = new_from
                model(input_ids=torch.tensor([ids], device=device))
                seen += 1
    finally:
        for handle in handles:
            handle.remove()
        model.train(was)
    return rows, seen


def top_basis(x: Any, rank: int) -> tuple[Any, float]:
    """The ``rank`` directions holding most of the rows ``x`` (orthonormal columns, fewer if the rows span
    fewer), and the share of their squared norm those directions hold."""
    import torch

    x = x.to(torch.float64 if x.device.type == "cpu" else torch.float32)
    n, d = x.shape
    values, vectors = torch.linalg.eigh(x.T @ x if d <= n else x @ x.T)
    held = int((values > EIGEN_FLOOR * values.max()).sum()) if float(values.max()) > 0 else 0
    take = min(rank, held)
    top = vectors[:, values.argsort(descending=True)[:take]]
    q, _ = torch.linalg.qr(top if d <= n else x.T @ top)
    total = float(values.clamp(min=0).sum())
    return q, (float(values.sort(descending=True).values[:take].clamp(min=0).sum()) / total if total else 0.0)


def random_basis(width: int, rank: int, seed: int, name: str) -> Any:
    """A seeded random orthonormal ``(width, rank)`` basis for module ``name``: the ablation's."""
    import torch

    gen = torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}:protect:{name}".encode()).hexdigest()[:15], 16))
    q, _ = torch.linalg.qr(torch.randn(width, rank, generator=gen, dtype=torch.float64))
    return q


def bases(model: Any, anchors: Sequence[Any], kind: str, *, seed: int, rank: int = RANK, adapter: str = ADAPTER,
          cap_minutes: float = MAX_MINUTES, timer: Callable[[], float] = time.monotonic) -> tuple[dict[str, Any], dict[str, Any]]:
    """Each LoRA module's ``Q`` (in its ``lora_A``'s dtype and device) and the receipt: the basis, its rank,
    the modules, the sequences sampled, the share of the anchors' rows ``Q`` holds (mean over modules), the
    minutes."""
    import torch

    if kind not in KINDS:
        raise ValueError(f"the basis is one of {', '.join(KINDS)}")
    if not anchors:
        raise ValueError("the protected slot needs the anchor items its basis is read from")
    begun = timer()
    rows, seen = _rows(model, anchors, adapter, cap_minutes * 60.0, timer)
    found: dict[str, Any] = {}
    shares: list[float] = []
    for name, module in lora_modules(model, adapter).items():
        x = torch.cat(rows.pop(name))
        q, share = top_basis(x, rank)
        if kind == "random":
            q = random_basis(x.shape[1], q.shape[1], seed, name).to(q)
            share = float((x.to(q) @ q).pow(2).sum() / x.to(q).pow(2).sum())
        weight = module.lora_A[adapter].weight
        found[name] = q.to(device=weight.device, dtype=weight.dtype)
        shares.append(share)
    return found, {"basis": kind, "rank": rank, "rank_min": min(q.shape[1] for q in found.values()),
                   "modules": len(found), "sequences": seen, "sequences_asked": len(sequences(anchors)),
                   "energy": round(sum(shares) / len(shares), 4), "minutes": round((timer() - begun) / 60.0, 4)}


def _projector(q: Any, lead: int) -> Callable[[Any, Any, Any], Any]:
    def project(module: Any, args: Any, out: Any) -> Any:
        import torch

        correction = (args[0] @ q) @ (module.weight[lead:] @ q).T
        return out - correction if lead == 0 else torch.cat([out[..., :lead], out[..., lead:] - correction], dim=-1)
    return project


def attach(model: Any, found: Mapping[str, Any], lead: int, adapter: str = ADAPTER) -> list[Any]:
    """Hooks that make each ``lora_A`` compute ``A_slot (I - Q Q^T)`` on the ranks past ``lead``."""
    modules = lora_modules(model, adapter)
    if set(modules) != set(found):
        raise ValueError("a basis for every LoRA module, and only for those")
    return [module.lora_A[adapter].register_forward_hook(_projector(found[name], lead)) for name, module in modules.items()]


def materialize(model: Any, found: Mapping[str, Any], lead: int, adapter: str = ADAPTER) -> float:
    """``A_slot (I - Q Q^T)`` written into each ``lora_A``; the largest share of a slot left on its ``Q``
    after (rounding only). Remove the hooks after: the weights now compute what they did."""
    import torch

    leak = 0.0
    with torch.no_grad():
        for name, module in lora_modules(model, adapter).items():
            weight, q = module.lora_A[adapter].weight, found[name]
            weight[lead:] -= (weight[lead:] @ q) @ q.T
            size = float(weight[lead:].abs().max()) if weight.shape[0] > lead else 0.0
            if size:
                leak = max(leak, float((weight[lead:] @ q).abs().max()) / size)
    return leak

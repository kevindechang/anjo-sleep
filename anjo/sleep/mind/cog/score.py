"""Scoring of MIND's cognition sprint: T1-T4 over ``LifeRecord``s, paired intervals, one report.

- **T1** mean log loss (p clipped to [1e-4, 1 - 1e-4]) and Brier, pooled over queries, per family,
  per week of ``asked_day`` (the learning curve), with the oracle's and the true state's losses.
- **T2** ECE over 15 equal-width bins; AURC, abstaining on the highest ``u`` first (ties share
  their losses, so a constant ``u`` scores like random order), with the optimal (abstain on the
  highest loss itself) and random (the mean loss) references; Spearman of ``u`` with the excess loss
  over the oracle. Skipped (None) when any of a run's ``u`` is None.
- **T3** per life, paired with the same predictor run without the inquirer: the queries asked in
  the ``window`` days starting on each ask's day (the union, so overlapping asks count a query
  once); loss reduction = sum over them of (loss without asks - loss with asks), in nats;
  utility = loss reduction - total cost of the asks. Never asking is worth exactly 0. Also the
  share of asks whose subject is one of the life's topics (distractors).
- **T4** per shift ``s`` (up to the next shift or the life's end): recovery is the smallest ``k``
  such that the mean of (loss - oracle loss) over the queries asked in the full week
  ``s + k`` .. ``s + k + 6`` is <= ``margin``; censored at the number of full weeks that fit (and
  None when not even one does). Weeks lie wholly after the shift, so a good pre-shift record
  cannot hide the dip, and a week of queries keeps one lucky day from counting as recovery.
  ``gap14`` is the mean excess over the 14 days from ``s``; on subjects the shift did not touch,
  the excess after the shift against the 14 days before it (``forgetting``).

Intervals are percentile bootstraps over lives, paired, with a seeded RNG: a pooled mean is a
ratio of per-life sums, resampled together. Everything returned is plain JSON.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from anjo.sleep.mind.cog.run import LifeRecord, QueryRecord

EPS = 1e-4
N_BINS = 15
ROLLING_DAYS = 7
GAP_DAYS = 14
DEFAULT_MARGIN = 0.05
DEFAULT_WINDOW = 7
DEFAULT_BOOT = 2000


# -- per-query scores --
def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=bool)
    out: np.ndarray = -np.where(y, np.log(p), np.log1p(-p))
    return out


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    out: np.ndarray = (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2
    return out


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        sel = bins == b
        total += sel.sum() * abs(y[sel].mean() - p[sel].mean())
    return float(total / len(p)) if len(p) else 0.0


def aurc(u: np.ndarray, loss: np.ndarray) -> float:
    """Mean risk over coverages 1/n .. n/n, keeping the lowest ``u`` first."""
    u, loss = np.asarray(u, dtype=float), np.asarray(loss, dtype=float)
    order = np.argsort(u, kind="stable")
    su, sl = u[order], loss[order].copy()
    start = 0
    for end in range(1, len(su) + 1):  # ties share their mean loss: the expected random tie-break
        if end == len(su) or su[end] != su[start]:
            sl[start:end] = sl[start:end].mean()
            start = end
    risks = np.cumsum(sl) / np.arange(1, len(sl) + 1)
    return float(risks.mean()) if len(risks) else 0.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    sx = x[order]
    start = 0
    for end in range(1, len(sx) + 1):
        if end == len(sx) or sx[end] != sx[start]:
            ranks[order[start:end]] = (start + end + 1) / 2.0
            start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    rx, ry = _rankdata(np.asarray(x, dtype=float)), _rankdata(np.asarray(y, dtype=float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return None if denom == 0.0 else float((rx * ry).sum() / denom)


# -- query tables --
def _arrays(queries: Sequence[QueryRecord]) -> dict[str, np.ndarray]:
    p = np.array([q.p for q in queries], dtype=float)
    y = np.array([q.outcome for q in queries], dtype=bool)
    oracle = np.array([q.oracle_p for q in queries], dtype=float)
    state = np.array([q.state_p for q in queries], dtype=float)
    ll, oll = log_loss(p, y), log_loss(oracle, y)
    return {"p": p, "y": y, "oracle": oracle, "loss": ll, "oracle_loss": oll,
            "state_loss": log_loss(state, y), "excess": ll - oll, "brier": brier(p, y)}


def _mean(x: np.ndarray) -> float | None:
    return float(x.mean()) if len(x) else None


def summarize_run(records: Sequence[LifeRecord]) -> dict[str, Any]:
    """T1 and T2 over one run's lives, pooled over queries."""
    queries = [q for r in records for q in r.queries]
    a = _arrays(queries)
    out: dict[str, Any] = {
        "n_lives": len(records), "n_queries": len(queries),
        "log_loss": _mean(a["loss"]), "brier": _mean(a["brier"]),
        "oracle_log_loss": _mean(a["oracle_loss"]), "state_log_loss": _mean(a["state_loss"]),
        "excess_over_oracle": _mean(a["excess"]),
        "mean_abs_gap_to_oracle": _mean(np.abs(a["p"] - a["oracle"])),
        "ece": ece(a["p"], a["y"]) if queries else None,
    }
    families: dict[str, Any] = {}
    for fam in sorted({q.family for q in queries}):
        sel = np.array([q.family == fam for q in queries])
        families[fam] = {"n": int(sel.sum()), "log_loss": _mean(a["loss"][sel]),
                         "brier": _mean(a["brier"][sel]), "oracle_log_loss": _mean(a["oracle_loss"][sel]),
                         "ece": ece(a["p"][sel], a["y"][sel])}
    out["by_family"] = families
    weeks = np.array([q.asked_day // 7 for q in queries], dtype=int)
    out["learning_curve"] = [
        {"week": int(w), "n": int((weeks == w).sum()), "log_loss": _mean(a["loss"][weeks == w]),
         "oracle_log_loss": _mean(a["oracle_loss"][weeks == w])} for w in np.unique(weeks)]
    has_u = bool(queries) and all(q.u is not None for q in queries)
    if has_u:
        u = np.array([q.u for q in queries], dtype=float)
        out["aurc"] = {"aurc": aurc(u, a["loss"]), "optimal": aurc(a["loss"], a["loss"]),
                       "random": _mean(a["loss"])}
        out["spearman_u_excess"] = spearman(u, a["excess"])
    else:
        out["aurc"], out["spearman_u_excess"] = None, None
    usage: dict[str, float] = {}
    for r in records:
        for k, v in r.usage.items():
            usage[k] = usage.get(k, 0.0) + float(v)
    out["usage"] = usage
    asks = [ask for r in records for ask in r.asks]
    out["asks"] = {"n": len(asks), "per_life": len(asks) / len(records) if records else 0.0,
                   "cost": float(sum(ask.cost for ask in asks)),
                   "distractor_share": (sum(ask.is_distractor for ask in asks) / len(asks)) if asks else None}
    return out


# -- bootstrap --
def _ratio_ci(num: np.ndarray, den: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    """sum(num) / sum(den) with a percentile interval from resampling lives (rows) together."""
    if len(num) == 0 or den.sum() == 0:
        return {"mean": None, "ci": None}
    est = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    dens = den[idx].sum(axis=1)
    ok = dens > 0
    boots = num[idx].sum(axis=1)[ok] / dens[ok]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": est, "ci": [float(lo), float(hi)]}


def _mean_ci(values: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    return _ratio_ci(np.asarray(values, dtype=float), np.ones(len(values)), n_boot, seed)


def _paired(a: Sequence[LifeRecord], b: Sequence[LifeRecord]) -> list[tuple[LifeRecord, LifeRecord]]:
    by_key = {r.key: r for r in b}
    return [(ra, by_key[ra.key]) for ra in a if ra.key in by_key]


def _common(ra: LifeRecord, rb: LifeRecord) -> tuple[list[QueryRecord], list[QueryRecord]]:
    qb = {q.qid: q for q in rb.queries}
    qa = [q for q in ra.queries if q.qid in qb]
    return qa, [qb[q.qid] for q in qa]


# -- T3 --
def _window_days(record: LifeRecord, window: int) -> set[int]:
    return {d for ask in record.asks for d in range(ask.day, ask.day + window)}


def _t3_life(ra: LifeRecord, rb: LifeRecord, window: int) -> tuple[float, float, int]:
    """(loss reduction, ask cost, queries in the windows) for one life, run with vs without asks."""
    days = _window_days(ra, window)
    qa, qb = _common(ra, rb)
    sel = [i for i, q in enumerate(qa) if q.asked_day in days]
    if not sel:
        return 0.0, float(sum(ask.cost for ask in ra.asks)), 0
    la = _arrays([qa[i] for i in sel])["loss"]
    lb = _arrays([qb[i] for i in sel])["loss"]
    return float((lb - la).sum()), float(sum(ask.cost for ask in ra.asks)), len(sel)


def t3_utility(with_asks: Sequence[LifeRecord], without: Sequence[LifeRecord], *,
               window: int = DEFAULT_WINDOW, n_boot: int = DEFAULT_BOOT, seed: int = 0) -> dict[str, Any]:
    pairs = _paired(with_asks, without)
    if not pairs:
        return {"n_lives": 0, "window": window, "n_window_queries": 0, "utility_per_life": None,
                "utility_ci": None, "loss_reduction_per_life": None, "cost_per_life": None,
                "asks_per_life": None, "utility_per_ask": None, "distractor_share": None}
    lives = [_t3_life(ra, rb, window) for ra, rb in pairs]
    red = np.array([life[0] for life in lives])
    cost = np.array([life[1] for life in lives])
    utility = red - cost
    asks = [ask for ra, _ in pairs for ask in ra.asks]
    return {
        "n_lives": len(pairs), "window": window, "n_window_queries": sum(life[2] for life in lives),
        "utility_per_life": float(utility.mean()), "utility_ci": _mean_ci(utility, n_boot, seed)["ci"],
        "loss_reduction_per_life": float(red.mean()), "cost_per_life": float(cost.mean()),
        "asks_per_life": len(asks) / len(pairs),
        "utility_per_ask": float(utility.sum() / len(asks)) if asks else None,
        "distractor_share": (sum(a.is_distractor for a in asks) / len(asks)) if asks else None,
    }


def _t3_per_life(with_asks: Sequence[LifeRecord], without: Sequence[LifeRecord],
                 window: int) -> dict[tuple[int | None, str], float]:
    out = {}
    for ra, rb in _paired(with_asks, without):
        reduction, cost, _ = _t3_life(ra, rb, window)
        out[ra.key] = reduction - cost
    return out


# -- T4 --
def _shift_windows(record: LifeRecord) -> list[tuple[int, int, int]]:
    """(previous boundary, shift day, end) for each shift, ends at the next shift or life's end."""
    shifts = sorted(record.shift_days)
    bounds = [*shifts[1:], record.n_days]
    prevs = [0, *shifts[:-1]]
    return [(prev, s, end) for prev, s, end in zip(prevs, shifts, bounds) if s < end]


def _one_shift(record: LifeRecord, prev: int, s: int, end: int, margin: float) -> dict[str, Any]:
    qs = [q for q in record.queries if s <= q.asked_day < end]
    excess = _arrays(qs)["excess"] if qs else np.zeros(0)
    days = np.array([q.asked_day for q in qs], dtype=int)
    starts = range(s, end - ROLLING_DAYS + 1)  # every full week after the shift, before ``end``
    recovery: int | None = len(starts) if len(starts) else None
    recovered = False
    for start in starts:
        sel = (days >= start) & (days < start + ROLLING_DAYS)
        if sel.any() and excess[sel].mean() <= margin:
            recovery, recovered = start - s, True
            break
    early = days < s + GAP_DAYS
    out: dict[str, Any] = {"recovery_days": recovery, "recovered": recovered,
                           "gap14": float(excess[early].mean()) if early.any() else None,
                           "untouched": None}
    if record.shift_subjects is not None:
        touched = set(record.shift_subjects)
        post = [q for q in qs if q.subject not in touched]
        pre = [q for q in record.queries
               if max(prev, s - GAP_DAYS) <= q.asked_day < s and q.subject not in touched]
        if post:
            a_post = _arrays(post)
            pre_excess = _mean(_arrays(pre)["excess"]) if pre else None
            post_excess = float(a_post["excess"].mean())
            out["untouched"] = {
                "post_log_loss": float(a_post["loss"].mean()), "post_excess": post_excess,
                "pre_excess": pre_excess,
                "forgetting": None if pre_excess is None else post_excess - pre_excess}
    return out


def t4_per_life(records: Sequence[LifeRecord], margin: float = DEFAULT_MARGIN) -> dict[tuple[int | None, str], list[dict[str, Any]]]:
    return {r.key: [_one_shift(r, prev, s, end, margin) for prev, s, end in _shift_windows(r)]
            for r in records if r.shift_days}


def t4_recovery(records: Sequence[LifeRecord], *, margin: float = DEFAULT_MARGIN) -> dict[str, Any] | None:
    per_life = t4_per_life(records, margin)
    shifts = [s for life in per_life.values() for s in life]
    if not shifts:
        return None
    timed = [s for s in shifts if s["recovery_days"] is not None]
    days = np.array([s["recovery_days"] for s in timed], dtype=float)
    gaps = np.array([s["gap14"] for s in shifts if s["gap14"] is not None], dtype=float)
    untouched = [s["untouched"] for s in shifts if s["untouched"] is not None]

    def avg(key: str) -> float | None:
        vals = [u[key] for u in untouched if u[key] is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "margin": margin, "n_lives": len(per_life), "n_shifts": len(shifts), "n_recovery": len(timed),
        "recovered_share": float(np.mean([s["recovered"] for s in timed])) if timed else None,
        "recovery_days_mean": _mean(days),
        "recovery_days_median": float(np.median(days)) if len(days) else None,
        "gap14_mean": _mean(gaps),
        "untouched": {k: avg(k) for k in ("post_log_loss", "post_excess", "pre_excess", "forgetting")}
        if untouched else None,
    }


# -- paired comparisons --
def compare(a: Sequence[LifeRecord], b: Sequence[LifeRecord], *, margin: float = DEFAULT_MARGIN,
            n_boot: int = DEFAULT_BOOT, seed: int = 0) -> dict[str, Any]:
    """Run a minus run b, on the queries both answered, lives paired; negative = a is better."""
    pairs = _paired(a, b)
    ll_num, br_num, den = [], [], []
    week_rows: dict[int, list[tuple[float, float]]] = {}
    for ra, rb in pairs:
        qa, qb = _common(ra, rb)
        if not qa:
            continue
        xa, xb = _arrays(qa), _arrays(qb)
        dl = xa["loss"] - xb["loss"]
        ll_num.append(float(dl.sum()))
        br_num.append(float((xa["brier"] - xb["brier"]).sum()))
        den.append(float(len(qa)))
        weeks = np.array([q.asked_day // 7 for q in qa])
        for w in np.unique(weeks):
            week_rows.setdefault(int(w), []).append((float(dl[weeks == w].sum()), float((weeks == w).sum())))
    num_ll, num_br, d = np.array(ll_num), np.array(br_num), np.array(den)
    out: dict[str, Any] = {
        "n_lives": len(pairs), "n_queries": int(d.sum()),
        "log_loss_diff": _ratio_ci(num_ll, d, n_boot, seed),
        "brier_diff": _ratio_ci(num_br, d, n_boot, seed),
        "by_week": [{"week": w, **_ratio_ci(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
                                           n_boot, seed)} for w, rows in sorted(week_rows.items())],
    }
    out["t4"] = _compare_t4(a, b, margin, n_boot, seed)
    return out


def _compare_t4(a: Sequence[LifeRecord], b: Sequence[LifeRecord], margin: float, n_boot: int,
                seed: int) -> dict[str, Any] | None:
    ta, tb = t4_per_life(a, margin), t4_per_life(b, margin)
    keys = [k for k in ta if k in tb]
    if not keys:
        return None

    def life_mean(shifts: list[dict[str, Any]], key: str) -> float | None:
        vals = [s[key] for s in shifts if s[key] is not None]
        return float(np.mean(vals)) if vals else None

    out: dict[str, Any] = {"n_lives": len(keys)}
    for key in ("recovery_days", "gap14"):
        diffs = [(life_mean(ta[k], key), life_mean(tb[k], key)) for k in keys]
        vals = np.array([x - y for x, y in diffs if x is not None and y is not None])
        out[f"{key}_diff"] = _mean_ci(vals, n_boot, seed)
    return out


# -- the report --
def group_runs(records: Iterable[LifeRecord]) -> dict[str, tuple[LifeRecord, ...]]:
    groups: dict[str, list[LifeRecord]] = {}
    for r in records:
        groups.setdefault(r.run, []).append(r)
    return {k: tuple(v) for k, v in groups.items()}


def _base_run(name: str, runs: Mapping[str, Sequence[LifeRecord]]) -> str | None:
    recs = runs[name]
    if not recs or recs[0].inquirer is None:
        return None
    predictor = recs[0].predictor
    bases = [n for n, rs in runs.items() if rs and rs[0].inquirer is None and rs[0].predictor == predictor]
    return predictor if predictor in bases else (bases[0] if bases else None)


def build_report(runs: Mapping[str, Sequence[LifeRecord]], *, reference: str | None = None,
                 margin: float = DEFAULT_MARGIN, window: int = DEFAULT_WINDOW,
                 n_boot: int = DEFAULT_BOOT, seed: int = 0) -> dict[str, Any]:
    """Every run's T1-T4, paired comparisons (each against ``reference``, else every pair), and
    T3 differences between inquirers sharing a base run."""
    if reference is not None and reference not in runs:
        raise ValueError(f"reference {reference!r} is not among the runs {sorted(runs)}")
    summaries: dict[str, Any] = {}
    bases: dict[str, str | None] = {}
    for name, recs in runs.items():
        summary = summarize_run(recs)
        base = bases[name] = _base_run(name, runs)
        summary["t3"] = None if base is None else {
            "base_run": base, **t3_utility(recs, runs[base], window=window, n_boot=n_boot, seed=seed)}
        summary["t4"] = t4_recovery(recs, margin=margin)
        summaries[name] = summary
    names = list(runs)
    pairs = ([(n, reference) for n in names if n != reference] if reference is not None
             else list(itertools.combinations(names, 2)))
    comparisons = [{"a": x, "b": y, **compare(runs[x], runs[y], margin=margin, n_boot=n_boot, seed=seed)}
                   for x, y in pairs]
    t3_comparisons = []
    for x, y in itertools.combinations([n for n in names if bases[n] is not None], 2):
        base = bases[x]
        if base is None or base != bases[y]:
            continue
        ux = _t3_per_life(runs[x], runs[base], window)
        uy = _t3_per_life(runs[y], runs[base], window)
        diffs = np.array([ux[k] - uy[k] for k in ux if k in uy])
        t3_comparisons.append({"a": x, "b": y, "base_run": base, "n_lives": len(diffs),
                               "utility_diff": _mean_ci(diffs, n_boot, seed)})
    return {"config": {"margin": margin, "window": window, "n_boot": n_boot, "seed": seed,
                       "reference": reference, "eps": EPS, "n_bins": N_BINS},
            "runs": summaries, "comparisons": comparisons, "t3_comparisons": t3_comparisons}

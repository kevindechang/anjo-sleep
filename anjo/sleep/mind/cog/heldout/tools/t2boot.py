"""T2 endpoints paired over lives, percentile CI from resampling lives (prereg §4). The estimators are
anjo's score.aurc and score.spearman, vectorized with the same tie handling (tied u share their mean
loss; tied values share their mean rank), so the bootstrap and the point estimate agree; the loss is
score.log_loss (clip EPS) and the excess is over the oracle's loss, as in the report. CI bounds unrounded.
usage: t2boot.py p3 A.jsonl B.jsonl [n_boot]   A minus B: AURC(u, loss) and Spearman(u, excess)
       t2boot.py k1 K1.jsonl MIND.jsonl [n_boot]  k1's AURC = its mean loss (no u) minus mind's AURC"""
import json
import sys

import numpy as np

from anjo.sleep.mind.cog import score


def aurc(u: np.ndarray, loss: np.ndarray) -> float:
    order = np.argsort(u, kind="stable")
    su, sl = u[order], loss[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    counts = np.diff(np.r_[starts, len(su)])
    shared = np.repeat(np.add.reduceat(sl, starts) / counts, counts)
    return float((np.cumsum(shared) / np.arange(1, len(shared) + 1)).mean()) if len(shared) else 0.0


def rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    sx = x[order]
    starts = np.flatnonzero(np.r_[True, sx[1:] != sx[:-1]])
    ends = np.r_[starts[1:], len(sx)]
    ranks = np.empty(len(x))
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def load(path: str, with_u: bool) -> dict[int, tuple]:
    out = {}
    for line in open(path):
        r = json.loads(line)
        q = r["queries"]
        y = np.array([x["outcome"] for x in q], dtype=bool)
        loss = score.log_loss(np.array([x["p"] for x in q]), y)
        excess = loss - score.log_loss(np.array([x["oracle_p"] for x in q]), y)
        u = np.array([x["u"] for x in q], dtype=float) if with_u else None
        if r["seed"] in out:
            raise SystemExit(f"{path}: seed {r['seed']} twice")
        out[r["seed"]] = (u, loss, excess)
    return out


def main() -> None:
    mode, a_path, b_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n_boot = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    a, b = load(a_path, mode == "p3"), load(b_path, True)
    if set(a) != set(b):
        raise SystemExit("the two runs cover different lives")
    seeds = sorted(a)
    cat = lambda d, ss, i: np.concatenate([d[s][i] for s in ss])  # noqa: E731

    def stats(ss: list[int]) -> tuple[float, ...]:
        if mode == "k1":
            return (float(cat(a, ss, 1).mean()) - aurc(cat(b, ss, 0), cat(b, ss, 1)),)
        return (aurc(cat(a, ss, 0), cat(a, ss, 1)) - aurc(cat(b, ss, 0), cat(b, ss, 1)),
                spearman(cat(a, ss, 0), cat(a, ss, 2)) - spearman(cat(b, ss, 0), cat(b, ss, 2)))

    point = stats(seeds)
    rng = np.random.default_rng(0)
    boots = np.array([stats([seeds[i] for i in rng.integers(0, len(seeds), len(seeds))]) for _ in range(n_boot)])
    ci = lambda k: [float(np.quantile(boots[:, k], 0.025)), float(np.quantile(boots[:, k], 0.975))]  # noqa: E731
    out = {"mode": mode, "a": a_path, "b": b_path, "n_lives": len(seeds), "n_boot": n_boot}
    if mode == "k1":
        out["k1_aurc_mean_loss"] = float(cat(a, seeds, 1).mean())
        out["mind_aurc"] = aurc(cat(b, seeds, 0), cat(b, seeds, 1))
        out["diff"], out["ci"] = point[0], ci(0)
    else:
        out["aurc"] = {"a": aurc(cat(a, seeds, 0), cat(a, seeds, 1)), "b": aurc(cat(b, seeds, 0), cat(b, seeds, 1)),
                       "diff": point[0], "ci": ci(0)}
        out["spearman_u_excess"] = {"a": spearman(cat(a, seeds, 0), cat(a, seeds, 2)),
                                    "b": spearman(cat(b, seeds, 0), cat(b, seeds, 2)), "diff": point[1], "ci": ci(1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

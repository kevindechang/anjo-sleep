"""The cog world's references, exact: a forward filter over the hidden state and a forward pass
over the future. Pure numpy.

``oracle_p`` starts from the filter's posterior of the hidden state on the asked day (every
non-answer observation up to then, the true parameters); ``state_p`` starts from the true state.
Both then roll the hidden chain forward to the target day and, where the activity has a streak or a
trigger, the joint of the chain with the activity's own streak and its trigger source's outcome
yesterday, branching on each day's unseen outcomes. Plans not made by the asked day enter through
their exact chance of being made in time. Nothing is sampled, so the references are exact up to
floating point.

What the filter sees and why the rest drops out: every activity outcome on a day it occurs (its
likelihood depends on the state), the late-message count, the mood level and the hints. Plan
creation, daytime messages and distractor mentions are drawn independently of the hidden state, so
their likelihoods cancel.
"""
from __future__ import annotations

import numpy as np

from anjo.sleep.mind.cog.world_prior import MAX_STREAK, PLAN_LEAD_MAX, Person


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out: np.ndarray = 1.0 / (1.0 + np.exp(-x))
    return out


def log_emissions(person: Person, y: np.ndarray, occ: np.ndarray, streak_in: np.ndarray, prev_in: np.ndarray,
                  late: np.ndarray, mood: np.ndarray, hints: np.ndarray) -> np.ndarray:
    """(T, S) log-likelihood of each day's observations under each hidden state."""
    src = np.maximum(person.trigger_src, 0)
    feat = person.streak_w[None, :] * streak_in + person.trigger_w[None, :] * prev_in[:, src]
    logits = person.base.T[:, :, None] + person.effect[None, :, :] + feat[:, :, None]
    signed = np.where(y[:, :, None], -logits, logits)
    out: np.ndarray = -(np.logaddexp(0.0, signed) * occ[:, :, None]).sum(axis=1)
    out += late[:, None] * np.log(person.late_p)[None, :] + (2 - late)[:, None] * np.log1p(-person.late_p)[None, :]
    out += np.log(person.mood_p)[:, mood - 1].T
    out += hints @ np.log(person.hint_p).T + (1 - hints) @ np.log1p(-person.hint_p).T
    return out


def forward(person: Person, loglik: np.ndarray) -> np.ndarray:
    """(T, S): P(hidden state on day d | observations of days 0..d)."""
    alpha: np.ndarray = np.empty_like(loglik)
    prior = person.init
    with np.errstate(divide="ignore"):
        for d in range(len(loglik)):
            if d:
                prior = alpha[d - 1] @ person.chain
            log_w = np.log(prior) + loglik[d]
            w = np.exp(log_w - log_w.max())
            alpha[d] = w / w.sum()
    return alpha


def occurrence_probs(person: Person, plan_made: np.ndarray, n_ask: int, horizon: int) -> np.ndarray:
    """(A, n_ask, horizon): the chance activity a takes place on day t+h, as seen on day t.

    A habit occurs on its slot days. A plan made by day t is known; a plannable day with no plan
    seen by t still gets one with chance r P(lead < D-t) / (1 - r P(lead >= D-t)), the lead uniform
    on 1..min(PLAN_LEAD_MAX, D)."""
    n_days = person.base.shape[1]
    t = np.arange(n_ask)[:, None]
    d = t + np.arange(1, horizon + 1)[None, :]
    inside = d <= n_days - 1
    dc = np.minimum(d, n_days - 1)
    span = np.minimum(PLAN_LEAD_MAX, dc)
    ahead = d - t
    p_late = np.maximum(0, span - ahead + 1) / span  # P(lead >= D - t)
    out = np.zeros((len(person.activities), n_ask, horizon))
    for a in range(len(person.activities)):
        if not person.planned[a]:
            out[a] = person.pattern[a, dc] * inside
            continue
        made = plan_made[a, dc]
        known = (made >= 0) & (made <= t)
        rate = person.plan_rate[a, dc] * person.pattern[a, dc]
        later = rate * (1.0 - p_late) / (1.0 - rate * p_late)
        out[a] = np.where(known, 1.0, later) * inside
    return out


def predictive(person: Person, starts: np.ndarray, rows_t: np.ndarray, occ: np.ndarray,
               streak0: np.ndarray, prev0: np.ndarray, horizon: int) -> np.ndarray:
    """(A, B, horizon): P(activity a done on day t_b + h | it occurs, start distribution b).

    ``starts`` (B, S) is each row's hidden-state distribution on its asked day ``rows_t[b]``;
    ``occ`` (A, B, horizon) the occurrence chances; ``streak0`` / ``prev0`` (B, A) the observed
    streak entering day t+1 and yesterday-outcome of each activity on day t."""
    n_days = person.base.shape[1]
    dc = np.minimum(rows_t[:, None] + np.arange(1, horizon + 1)[None, :], n_days - 1)
    marg = np.empty((len(starts), horizon, starts.shape[1]))
    current = starts
    for h in range(horizon):
        current = current @ person.chain
        marg[:, h] = current
    out = np.empty((len(person.activities), len(starts), horizon))
    for a in range(len(person.activities)):
        if person.streak_w[a] == 0 and person.trigger_src[a] < 0:
            p = _sigmoid(person.base[a, dc][:, :, None] + person.effect[a][None, None, :])
            out[a] = (marg * p).sum(axis=2)
        else:
            out[a] = _joint(person, a, starts, dc, occ, streak0, prev0, horizon)
    return out


def _joint(person: Person, a: int, starts: np.ndarray, dc: np.ndarray, occ: np.ndarray,
           streak0: np.ndarray, prev0: np.ndarray, horizon: int) -> np.ndarray:
    """The forward pass over (streak of a, trigger source done yesterday, hidden state)."""
    src = int(person.trigger_src[a])
    n_s = MAX_STREAK + 1 if person.streak_w[a] != 0 else 1
    n_y = 2 if src >= 0 else 1
    rows = np.arange(len(starts))
    pi = np.zeros((len(starts), n_s, n_y, starts.shape[1]))
    s_idx: np.ndarray | int = streak0[:, a] if n_s > 1 else 0
    y_idx: np.ndarray | int = prev0[:, src].astype(int) if n_y > 1 else 0
    pi[rows, s_idx, y_idx] = starts
    feat = person.streak_w[a] * np.arange(n_s)[:, None] + person.trigger_w[a] * np.arange(n_y)[None, :]
    out = np.empty((len(starts), horizon))
    for h in range(horizon):
        pi = pi @ person.chain
        logit = person.base[a, dc[:, h]][:, None, None, None] + person.effect[a][None, None, None, :]
        p = _sigmoid(logit + feat[None, :, :, None])
        done = pi * p
        out[:, h] = done.sum(axis=(1, 2, 3))
        if h == horizon - 1:
            break
        new = pi
        if n_s > 1:
            o = occ[a, :, h][:, None, None, None]
            done, fail = done * o, (pi - done) * o
            new = pi - done - fail
            new[:, 0] += done.sum(axis=1)
            new[:, 1:] += fail[:, :-1]
            new[:, -1] += fail[:, -1]
        if n_y > 1:
            ps = _sigmoid(person.base[src, dc[:, h]][:, None] + person.effect[src][None, :]) * occ[src, :, h][:, None]
            m = new.sum(axis=2)
            new = np.stack([m * (1.0 - ps)[:, None, :], m * ps[:, None, :]], axis=2)
        pi = new
    return out

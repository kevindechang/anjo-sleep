"""What a life's nights cost, by step, and the person-month against the $10 ceiling (hackathon report).

    python -m anjo.sleep.reports.cost --root <results> --user <id> [--user <id> ...]

**A night.** Each summary's ``cost_parts`` (``night/nightlog.result_record``) holds the night's own
spend: extraction, the brief, practice (Spark Open, a third-party critic and the serving GPU's
time), the training GPU, the gates' serving time and MIND. The night log's ``morning`` step
(``night/daily.morning_receipt``) holds what the summary never does: the morning run's seconds, its
judge's settled dollars and, for calls that reported no usage, how many were estimated at the
hold's per-call bound (``probes.morning.JUDGE_USD_PER_CALL``). The ``dailies`` step's GPU seconds
are folded into the morning line, as ``night.cost`` already does. Wonder's homework
(``wonder_usd``) is shown apart: it is outside both the summary and the ceiling.

The table prints the seven steps the hackathon prices (practice, training GPU, gates, extraction,
brief, MIND, morning) and says which count toward the ceiling (``demo.dailies.CEILING_PARTS``:
every cost part but MIND; the morning sits outside the night's summary and the ceiling too).

**A person-month.** At the observed promote rate, 30 × (share × a promoted night's mean ceiling
cost + the rest × a quiet night's mean), and at the worst case, training every night: 30 × the
trained nights' mean (``demo.dailies._trained``: promoted, rejected, or failed after training began,
since a rejected night trained too). Both count only the summary's ceiling parts, so the morning and
Wonder are reported but not charged against the $10, exactly as spec §5.5 counts it. Nights whose
summary has no cost parts are printed at their ``cost_usd`` and left out of the month: a total with
no split cannot price a step.

**An accelerated life.** Its nights are deep by construction (accelerated runs force deep), so the
observed-rate line is labeled "every night deep (accelerated runs force deep)", not a real schedule;
beside it, the month ``night.cost`` models through the nap-or-deep decision (``month_from_trace`` over
its ``ACTIVE`` and ``QUIET`` months, at the ``CURRENT`` setting priced on the life's last costed night,
under ``SleepPolicy``'s default cap), labeled as modeled. Neither cost model changes here.

**The drivers.** A step cut to zero saves its mean a night, 30 times over at the observed pace; the
top three by mean are listed with that saving. This is a marginal read, not a claim that a step can
be cut without losing what it buys.

Everything here is read from the results tree the reports already use; nothing reads a clock, the
network or the budget ledger (the ledger cannot split a batch's shared GPU session per person, and
the summary's split is the night's own). A rerun folder (``<date>.2``) is its own night's spend, as
``reports.results`` reads it.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from anjo.sleep.demo.dailies import CEILING_PARTS, CEILING_USD, MONTH_NIGHTS, _trained
from anjo.sleep.night import cost as night_cost
from anjo.sleep.night.cost import DEFAULT_PRICES
from anjo.sleep.night.sleep_pressure import SleepPolicy
from anjo.sleep.night.deps import Prices
from anjo.sleep.probes.morning import JUDGE_USD_PER_CALL
from anjo.sleep.reports import results

LOG: Final = "log.jsonl"
SUMMARY: Final = "summary.json"
PROMOTED: Final = "promoted"
# The seven steps the hackathon prices, in the order the night spends them.
STEPS: Final = ("practice", "training", "gates", "extract", "brief", "mind", "morning")
LABELS: Final[Mapping[str, str]] = {
    "practice": "practice", "training": "training GPU", "gates": "gates", "extract": "extraction",
    "brief": "brief", "mind": "MIND", "morning": "morning",
}
MORNING_STEPS: Final = ("morning", "dailies")  # the morning run and an accelerated day's GPU seconds
TOP_DRIVERS: Final = 3
MISSING: Final = "-"


class CostReportError(ValueError):
    """The results tree cannot be costed: a summary or log the report cannot read."""


@dataclass(frozen=True)
class StepCost:
    """One step's dollars a night; ``estimated`` marks a judge call at the hold's bound, not tokens."""

    step: str
    usd: float
    estimated: bool = False


@dataclass(frozen=True)
class NightCost:
    """One night folder: the night's own spend, the morning's, and Wonder's, as far as records allow."""

    user: str
    night: str
    folder: str  # the folder's name: <date>, or <date>.2 on a rerun
    status: str
    promoted: bool
    trained: bool  # a night that started training (``demo.dailies._trained``), not only a promoted one
    cost_usd: float  # the summary's total (its cost parts)
    costed: bool  # the summary carried cost parts, so the month can price its steps
    steps: tuple[StepCost, ...]
    wonder_usd: float
    accelerated: bool = False  # the summary's stamp: an accelerated run's night

    @property
    def ceiling_usd(self) -> float:
        """The night's share of the $10 ceiling: every cost part but MIND (and not the morning)."""
        return math.fsum(step.usd for step in self.steps if step.step in CEILING_PARTS)


@dataclass(frozen=True)
class Month:
    """A person-month at the observed pace, and at training every night, against ``CEILING_USD``."""

    nights: int  # costed nights the month is priced from
    promoted: int
    trained: int  # promoted or rejected: a night that started training
    promote_share: float
    at_pace: float | None
    every_night: float | None


@dataclass(frozen=True)
class Modeled:
    """A month ``night.cost`` models through the nap-or-deep decision (``month_from_trace``)."""

    usage: str  # ``active`` or ``quiet``
    usd: float  # every night's modeled spend, the morning and Wonder included
    deep: int
    naps: int
    cap_usd: float  # the policy's month cap


@dataclass(frozen=True)
class Driver:
    """One step's mean a night and what cutting it to zero would save a person-month."""

    step: str
    mean_usd: float
    month_usd: float


@dataclass(frozen=True)
class UserCost:
    """A life's nights, its month and its biggest steps."""

    user: str
    nights: tuple[NightCost, ...]
    month: Month | None
    drivers: tuple[Driver, ...]
    modeled: tuple[Modeled, ...] = ()  # an accelerated life's modeled schedule; empty otherwise


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value) else None


def _json(path: Path) -> dict[str, Any]:
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CostReportError(f"{path.name} cannot be read ({type(error).__name__})") from None
    if not isinstance(found, dict):
        raise CostReportError(f"{path.name} is not a record")
    return found


def _steps(folder: Path) -> list[Mapping[str, Any]]:
    """The night's step records; a night without a log (or with none yet) has no steps."""
    path = folder / LOG
    if not path.is_file():
        return []
    found: list[Mapping[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CostReportError(f"{path.name} line {number} is not JSON ({error.msg})") from None
        if isinstance(record, dict):
            found.append(record)
    return found


def _latest(steps: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    return next((step for step in reversed(steps) if step.get("step") == name), {})


def _seconds(steps: Sequence[Mapping[str, Any]]) -> float:
    """The morning's GPU seconds: the last morning and dailies records' ``seconds``, as night.cost does."""
    found = [_number(_latest(steps, name).get("seconds")) for name in MORNING_STEPS]
    return math.fsum(value for value in found if value is not None)


def morning_line(steps: Sequence[Mapping[str, Any]], prices: Prices = DEFAULT_PRICES) -> StepCost | None:
    """What the morning run spent: its judge and its serving GPU, or None when no morning ran.

    The judge's settled dollars cover calls at their tokens and calls with no usage at the hold's
    bound; a log from before the judge was metered carries ``judge_calls`` only, priced at the bound
    and marked estimated. A skipped morning (a nap) carries neither, so it gets no line.
    """
    record = _latest(steps, "morning")
    judge = _number(record.get("judge_usd"))
    calls = _number(record.get("judge_calls"))
    estimated = bool(_number(record.get("judge_estimated")))
    if judge is None and calls is not None:  # logged before the judge was metered
        judge, estimated = calls * JUDGE_USD_PER_CALL, True
    seconds = _seconds(steps)
    if judge is None and seconds == 0.0:
        return None
    gpu = seconds * prices.serve_gpu_per_hour / 3600.0
    return StepCost("morning", math.fsum([judge or 0.0, gpu]), estimated)


def _summary_steps(record: Mapping[str, Any]) -> tuple[StepCost, ...]:
    """The summary's cost parts, canonical steps first and any unknown part after, so none is dropped."""
    raw = record.get("cost_parts")
    if not isinstance(raw, Mapping):
        return ()
    found = {str(name): _number(value) for name, value in raw.items()}
    found = {name: value for name, value in found.items() if value is not None}
    ordered = [name for name in STEPS if name in found] + sorted(name for name in found if name not in STEPS)
    return tuple(StepCost(name, found[name]) for name in ordered)


def night(folder: Path, user: str) -> NightCost:
    """One night folder as its summary and log record it."""
    record = _json(folder / SUMMARY)
    steps = _steps(folder)
    parts = _summary_steps(record)
    morning = morning_line(steps)
    status = str(record.get("status") or "?")
    total = _number(record.get("cost_usd"))
    lines = list(parts)
    if morning is not None:
        lines.append(morning)
    return NightCost(
        user=user, night=str(record.get("night") or folder.name), folder=folder.name, status=status,
        promoted=status == PROMOTED, trained=_trained(record),
        costed=bool(parts),
        cost_usd=math.fsum(step.usd for step in parts) if total is None else total,
        steps=tuple(lines), wonder_usd=_number(record.get("wonder_usd")) or 0.0,
        accelerated=record.get("accelerated") is True,
    )


def nights(user: str, root: Path | None = None) -> tuple[NightCost, ...]:
    """Every night folder of ``user``, in folder order (a rerun is its own night)."""
    base = results.results_root(root) / user / "nights"
    folders = sorted(path.parent for path in base.glob(f"*/{SUMMARY}") if path.is_file())
    return tuple(night(folder, user) for folder in folders)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def month(nights_costed: Sequence[NightCost]) -> Month | None:
    """A person-month at the observed promote rate, and training every night; None with no splits.

    The promote rate stays promoted/known, as the task asks. The every-night case prices a night that
    started training, which a rejected night did too (only a promoted one is not the worst case).
    """
    known = [entry for entry in nights_costed if entry.costed]
    if not known:
        return None
    promoted = [entry.ceiling_usd for entry in known if entry.promoted]
    quiet = [entry.ceiling_usd for entry in known if not entry.promoted]
    trained = [entry.ceiling_usd for entry in known if entry.trained]
    share = len(promoted) / len(known)
    deep, nap = _mean(promoted), _mean(quiet)
    return Month(
        nights=len(known), promoted=len(promoted), trained=len(trained), promote_share=share,
        at_pace=MONTH_NIGHTS * (share * deep + (1.0 - share) * nap),
        every_night=MONTH_NIGHTS * _mean(trained) if trained else None,
    )


def _step_usd(entry: NightCost, step: str) -> float:
    return math.fsum(line.usd for line in entry.steps if line.step == step)


def drivers(nights_costed: Sequence[NightCost]) -> tuple[Driver, ...]:
    """The three biggest steps by their mean a night; cutting one to zero saves 30 times that."""
    known = [entry for entry in nights_costed if entry.costed]
    names = [step for step in STEPS if any(_step_usd(entry, step) for entry in known)]
    names += sorted({line.step for entry in known for line in entry.steps} - set(names))
    means = {name: _mean([_step_usd(entry, name) for entry in known]) for name in names}
    found = [Driver(name, mean, MONTH_NIGHTS * mean) for name, mean in means.items()]
    return tuple(sorted((driver for driver in found if driver.mean_usd > 0.0),
                        key=lambda driver: (-driver.mean_usd, driver.step))[:TOP_DRIVERS])


def forced_deep(nights_costed: Sequence[NightCost]) -> bool:
    """The costed nights come from an accelerated run that slept deep each night (no nap among them)."""
    known = [entry for entry in nights_costed if entry.costed]
    return bool(known) and any(entry.accelerated for entry in known) and not any(
        entry.status == "napped" for entry in known)


def modeled(folder: Path) -> tuple[Modeled, ...]:
    """``night.cost``'s modeled months through the nap-or-deep decision, priced on ``folder``'s night."""
    try:
        summary, steps = night_cost.read_night(folder)
    except night_cost.CostError as error:
        raise CostReportError(f"{folder.name}: {error}") from None
    policy = SleepPolicy()
    projection = night_cost.project(night_cost.CURRENT, night_cost.UNITS, DEFAULT_PRICES,
                                    night_cost.measured(summary, steps, DEFAULT_PRICES))
    found = []
    for usage in (night_cost.ACTIVE, night_cost.QUIET):
        trace = night_cost.month_from_trace(usage.days, policy, projection)
        found.append(Modeled(usage.name, trace.usd, trace.deep, trace.naps, policy.month_usd))
    return tuple(found)


def user_cost(user: str, root: Path | None = None) -> UserCost:
    """A life's nights, month and drivers; an accelerated life's modeled schedule too."""
    found = nights(user, root)
    model: tuple[Modeled, ...] = ()
    if forced_deep(found):
        last = [entry for entry in found if entry.costed][-1]
        model = modeled(results.results_root(root) / user / "nights" / last.folder)
    return UserCost(user, found, month(found), drivers(found), model)


def _usd(value: float | None) -> str:
    return MISSING if value is None else f"${value:.4f}"


def _against(value: float | None) -> str:
    if value is None:
        return "cannot be priced"
    side = "under" if value <= CEILING_USD else "over"
    return f"{side} the ${CEILING_USD:g} ceiling by {_usd(abs(value - CEILING_USD))}"


def _night_rows(entry: NightCost) -> list[str]:
    head = f"night {entry.night} ({entry.status}{', rerun' if '.' in entry.folder else ''})"
    if not entry.costed:
        return [f"{head}  night {_usd(entry.cost_usd)} (no cost parts; left out of the month)"]
    rows = [f"{head}  night {_usd(entry.cost_usd)}, ceiling parts {_usd(entry.ceiling_usd)}",
            _row(["step", "usd", "ceiling"], [12, 10, 8])]
    for line in entry.steps:
        rows.append(_row([LABELS.get(line.step, line.step), _usd(line.usd),
                          "yes" if line.step in CEILING_PARTS else "no"], [12, 10, 8])
                    + ("  estimated" if line.estimated else ""))
    if entry.wonder_usd:
        rows.append(_row(["wonder", _usd(entry.wonder_usd), "no"], [12, 10, 8]))
    return rows


def _row(cells: Sequence[str], widths: Sequence[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()


def render(found: UserCost) -> str:
    """One life's report: a night a block, the month, then the drivers."""
    out = [f"cost: {found.user}"]
    if not found.nights:
        return "\n".join([*out, "no nights"])
    for entry in found.nights:
        out += ["", *_night_rows(entry)]
    out.append("")
    if found.month is None:
        out.append(f"person-month: no cost parts in the nights, so the ${CEILING_USD:g} ceiling cannot be priced")
        return "\n".join(out)
    month = found.month
    out.append(f"person-month ({MONTH_NIGHTS} nights) against the ${CEILING_USD:g} ceiling")
    out.append(f"  {month.nights} costed nights, promote rate {month.promote_share:.2f}"
               f" ({month.promoted} promoted, {month.nights - month.promoted} quiet;"
               f" {month.trained} trained)")
    observed = ("every night deep (accelerated runs force deep)" if forced_deep(found.nights)
                else "at the observed rate")
    out.append(f"  {observed}  {_usd(month.at_pace)}  ({_against(month.at_pace)})")
    out.append(f"  training every night  {_usd(month.every_night)}  ({_against(month.every_night)})")
    for entry in found.modeled:
        out.append(f"  modeled nap/deep schedule, {entry.usage} month (night.cost, ${entry.cap_usd:g} cap,"
                   f" morning and Wonder included)  {_usd(entry.usd)}  ({entry.deep} deep, {entry.naps} naps)")
    out.append("")
    if not found.drivers:
        out.append("top cost drivers: none; the nights spent nothing in the priced steps")
    else:
        out.append("top three cost drivers (each cut to zero, at the observed pace)")
        for number, driver in enumerate(found.drivers, 1):
            out.append(f"  {number}. {LABELS.get(driver.step, driver.step):<12} {_usd(driver.mean_usd)} a night"
                       f"  saves {_usd(driver.month_usd)} a person-month")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None, *, root: Path | None = None, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description="A life's night costs and person-month against the ceiling.")
    parser.add_argument("--root", type=Path, default=None, help="results tree (default: ANJO_DATA_DIR/sleep/results)")
    parser.add_argument("--user", action="append", required=True, help="a life's results folder; repeat for more")
    args = parser.parse_args(argv)
    base = args.root or root
    for number, user in enumerate(args.user):
        if number:
            out("")
        out(render(user_cost(user, base)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

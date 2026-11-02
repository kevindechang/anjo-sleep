"""The command line of MIND's cognition sprint: ``python -m anjo.sleep.mind.cog``.

    eval   --predictor NAME --seeds 10000-10049 [--population base] --out DIR [--inquirer NAME]
           [--workers 12] [--train-seeds 0-299] [--n-days 84] [--set key=value ...]
           [--label NAME] [--query-every K] [--max-asks 1]
    report --records A.jsonl B.jsonl ... --out report.json [--reference NAME]

``eval`` writes ``DIR/<run>.jsonl`` and ``DIR/report.json``. With an inquirer it runs the predictor
twice, alone and with the inquirer, so the report has T3's pairing. Rivals that need a population
fit it on ``--train-seeds`` (train seeds only) in this process before the lives run.

``PREDICTORS`` and ``INQUIRERS`` map a name to a builder: ``builder(ctx) -> factory``, where the
factory makes a fresh predictor per life (an inquirer factory takes that life's predictor). A
factory must pickle for ``--workers > 1``. MIND's own predictor registers here as ``mind``.
Settings come from ``--set key=value``; a key no builder reads is an error.

The Gemini builders need ``--set client=module:attribute``, a ``Callable[[str], str]`` that the
operator supplies; this module never makes a client or reads a key. Responses are cached under
``--set cache_dir=...`` (default ``DIR/gemini_cache``); ``--set platt=a,b`` recalibrates.
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, TypeVar, cast

from anjo.sleep.mind.cog import score
from anjo.sleep.mind.cog.gemini import Complete, GeminiInquirer, GeminiPredictor
from anjo.sleep.mind.cog.inquiry_rivals import MaxUncertainty, NeverAsk, RandomAsk
from anjo.sleep.mind.cog.rivals import Counter, OnlineLogistic, PopulationPrior, SurpriseLR
from anjo.sleep.mind.cog.run import InquirerFactory, LifeMaker, PredictorFactory, load_records, run_many, save_records
from anjo.sleep.mind.cog.types import Inquirer, Life, Predictor

T = TypeVar("T")
DEFAULT_TRAIN_SEEDS = "0-299"
DEFAULT_N_DAYS = 84
WORLDS = {"main": "anjo.sleep.mind.cog.world", "indep": "anjo.sleep.mind.cog.world_indep"}


@dataclass(frozen=True)
class BuildContext:
    population: str
    train_seeds: tuple[int, ...]
    n_days: int
    make_life: LifeMaker  # bound to the population and n_days
    out_dir: Path
    options: Mapping[str, str] = field(default_factory=dict)
    used: set[str] = field(default_factory=set)  # option keys some builder read

    def option(self, key: str, default: T, cast: Callable[[str], T]) -> T:
        self.used.add(key)
        if key not in self.options:
            return default
        try:
            return cast(self.options[key])
        except ValueError as exc:
            raise ValueError(f"--set {key}={self.options[key]!r}: {exc}") from exc

    def train_lives(self) -> list[Life]:
        return [self.make_life(s) for s in self.train_seeds]


PredictorBuilder = Callable[[BuildContext], PredictorFactory]
InquirerBuilder = Callable[[BuildContext], InquirerFactory]


# -- predictor builders --
def _population(ctx: BuildContext) -> PredictorFactory:
    table = PopulationPrior.fit_population(ctx.train_lives(), smoothing=ctx.option("smoothing", 10.0, float))
    return partial(PopulationPrior, table)


def _counter(ctx: BuildContext) -> PredictorFactory:
    table = PopulationPrior.fit_population(ctx.train_lives(), smoothing=ctx.option("smoothing", 10.0, float))
    return partial(Counter, table, strength=ctx.option("strength", 6.0, float),
                   count_strength=ctx.option("count_strength", 100.0, float),
                   weekday_strength=ctx.option("weekday_strength", 100.0, float))


def _logistic_population(ctx: BuildContext) -> Any:
    return OnlineLogistic.fit_population(
        ctx.train_lives(), l2=ctx.option("l2", 1.0, float), ask_rate=ctx.option("ask_rate", 0.25, float),
        smoothing=ctx.option("smoothing", 10.0, float), strength=ctx.option("strength", 6.0, float),
        count_strength=ctx.option("count_strength", 100.0, float),
        weekday_strength=ctx.option("weekday_strength", 100.0, float))


def _online_lr(ctx: BuildContext) -> PredictorFactory:
    return partial(OnlineLogistic, _logistic_population(ctx), prior_var=ctx.option("prior_var", 1e-3, float),
                   shared_var=ctx.option("shared_var", 1e-6, float))


def _surprise_lr(ctx: BuildContext) -> PredictorFactory:
    return partial(SurpriseLR, _logistic_population(ctx), hazard=ctx.option("hazard", 0.02, float),
                   gain=ctx.option("gain", 1.0, float), decay=ctx.option("decay", 0.7, float),
                   prior_var=ctx.option("prior_var", 1e-3, float), shared_var=ctx.option("shared_var", 1e-6, float))


def load_client(path: str) -> Complete:
    """``module:attribute`` -> the callable it names."""
    module, _, attr = path.partition(":")
    if not module or not attr:
        raise ValueError(f"client must be module:attribute, got {path!r}")
    client = getattr(importlib.import_module(module), attr)
    if not callable(client):
        raise TypeError(f"{path} is not callable")
    return cast(Complete, client)


def _platt(raw: str) -> tuple[float, float]:
    a, b = (float(x) for x in raw.split(","))
    return a, b


@dataclass(frozen=True)
class GeminiFactory:
    """Makes a GeminiPredictor per life, importing the client in the process that runs it."""
    client: str
    cache_dir: Path
    model_label: str
    stats: bool
    platt: tuple[float, float] | None

    def __call__(self) -> Predictor:
        return GeminiPredictor(load_client(self.client), cache_dir=self.cache_dir,
                               model_label=self.model_label, stats=self.stats, platt=self.platt)


def _gemini_settings(ctx: BuildContext) -> tuple[str, Path, str]:
    client = ctx.option("client", "", str)
    if not client:
        raise ValueError("the Gemini runs need --set client=module:attribute (a Callable[[str], str])")
    return client, ctx.option("cache_dir", ctx.out_dir / "gemini_cache", Path), ctx.option("model", "gemini", str)


def _gemini(stats: bool) -> PredictorBuilder:
    def build(ctx: BuildContext) -> PredictorFactory:
        client, cache, model = _gemini_settings(ctx)
        return GeminiFactory(client, cache, model, stats, ctx.option("platt", None, _platt))
    return build


# -- inquirer builders --
def _make_never(predictor: Predictor) -> Inquirer:
    return NeverAsk()


def _make_random(predictor: Predictor, *, seed: int, p_ask: float) -> Inquirer:
    return RandomAsk(seed, p_ask)


@dataclass(frozen=True)
class GeminiInquirerFactory:
    client: str
    cache_dir: Path
    model_label: str

    def __call__(self, predictor: Predictor) -> Inquirer:
        return GeminiInquirer(load_client(self.client), cache_dir=self.cache_dir, model_label=self.model_label)


def _gemini_inquirer(ctx: BuildContext) -> InquirerFactory:
    return GeminiInquirerFactory(*_gemini_settings(ctx))


PREDICTORS: dict[str, PredictorBuilder] = {
    "population": _population,
    "counter": _counter,
    "online_lr": _online_lr,
    "surprise_lr": _surprise_lr,
    "gemini": _gemini(stats=False),
    "gemini_stats": _gemini(stats=True),
}

INQUIRERS: dict[str, InquirerBuilder] = {
    "never": lambda ctx: _make_never,
    "random": lambda ctx: partial(_make_random, seed=ctx.option("seed", 0, int),
                                  p_ask=ctx.option("p_ask", 0.3, float)),
    "max_uncertainty": lambda ctx: MaxUncertainty,
    "gemini": _gemini_inquirer,
}


# -- arguments --
def parse_seeds(spec: str) -> tuple[int, ...]:
    """``"10000-10049,7"`` -> (10000, ..., 10049, 7)."""
    out: list[int] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        lo, sep, hi = part.partition("-")
        if not sep:
            out.append(int(part))
            continue
        a, b = int(lo), int(hi)
        if b < a:
            raise ValueError(f"seed range {part!r} runs backwards")
        out.extend(range(a, b + 1))
    if not out:
        raise ValueError(f"no seeds in {spec!r}")
    return tuple(out)


def _parse_set(items: Sequence[str]) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set wants key=value, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m anjo.sleep.mind.cog")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="run a predictor (and an inquirer) over lives and score it")
    ev.add_argument("--predictor", required=True, choices=sorted(PREDICTORS))
    ev.add_argument("--inquirer", choices=sorted(INQUIRERS))
    ev.add_argument("--seeds", required=True)
    ev.add_argument("--world", choices=sorted(WORLDS), default="main")
    ev.add_argument("--population", default="base", help="the main world's population (base or ood)")
    ev.add_argument("--train-seeds", default=DEFAULT_TRAIN_SEEDS)
    ev.add_argument("--n-days", type=int, default=DEFAULT_N_DAYS)
    ev.add_argument("--out", required=True, type=Path)
    ev.add_argument("--workers", type=int, default=1)
    ev.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ev.add_argument("--label", help="the run's name in records and the report")
    ev.add_argument("--query-every", type=int, help="predict only every K-th day (days K-1, 2K-1, ...)")
    ev.add_argument("--max-asks", type=int, default=1)
    _report_args(ev)
    rep = sub.add_parser("report", help="score saved records")
    rep.add_argument("--records", required=True, nargs="+", type=Path)
    rep.add_argument("--out", required=True, type=Path)
    rep.add_argument("--reference")
    _report_args(rep)
    return parser


def _report_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-boot", type=int, default=score.DEFAULT_BOOT)
    p.add_argument("--margin", type=float, default=score.DEFAULT_MARGIN)
    p.add_argument("--window", type=int, default=score.DEFAULT_WINDOW)


def _world_maker(args: argparse.Namespace, make_life: Callable[..., Life] | None) -> LifeMaker:
    """``make_life(seed)`` of the chosen world, imported only when a real run needs it (tests
    inject a fake world, which takes ``population`` like the main one). The indep world is a
    black box with the same ``Life`` protocol and no populations."""
    if make_life is not None:
        return partial(make_life, population=args.population, n_days=args.n_days)
    world = importlib.import_module(WORLDS[args.world])
    maker = cast(Callable[..., Life], world.make_life)
    if args.world == "indep":
        if args.population != "base":
            raise ValueError("the indep world has no populations; drop --population")
        return partial(maker, n_days=args.n_days)
    return partial(maker, population=args.population, n_days=args.n_days)


def _write_report(path: Path, runs: Mapping[str, Sequence[Any]], args: argparse.Namespace,
                  reference: str | None = None) -> None:
    report = score.build_report(runs, reference=reference, margin=args.margin, window=args.window,
                                n_boot=args.n_boot)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _eval(args: argparse.Namespace, make_life: Callable[..., Life] | None) -> int:
    maker = _world_maker(args, make_life)
    ctx = BuildContext(population=args.population, train_seeds=parse_seeds(args.train_seeds),
                       n_days=args.n_days, make_life=maker, out_dir=args.out, options=_parse_set(args.set))
    factory = PREDICTORS[args.predictor](ctx)
    inquirer_factory = None if args.inquirer is None else INQUIRERS[args.inquirer](ctx)
    unused = sorted(set(ctx.options) - ctx.used)
    if unused:
        raise ValueError(f"no builder reads --set {', '.join(unused)}")
    seeds = parse_seeds(args.seeds)
    query_days = None if args.query_every is None else range(args.query_every - 1, args.n_days, args.query_every)
    base = args.label or args.predictor
    plan = [(base, None)] + ([] if inquirer_factory is None else [(f"{base}+{args.inquirer}", inquirer_factory)])
    runs = {}
    for run, inq in plan:
        runs[run] = run_many(seeds, factory, inquirer_factory=inq, make_life=maker, workers=args.workers,
                             max_asks_per_day=args.max_asks, query_days=query_days, run=run)
        save_records(args.out / f"{run}.jsonl", runs[run])
    _write_report(args.out / "report.json", runs, args)
    return 0


def _report(args: argparse.Namespace) -> int:
    records = [r for path in args.records for r in load_records(path)]
    _write_report(args.out, score.group_runs(records), args, reference=args.reference)
    return 0


def main(argv: Sequence[str] | None = None, *, make_life: Callable[..., Life] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "eval":
        return _eval(args, make_life)
    return _report(args)

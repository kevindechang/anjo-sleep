"""The results chart (Task 12a): one SVG, a panel per label (development, final), a line per arm
and probe set across the nights (a subset's dashed: scores on different probes are never one
line), and a strip each for general skills and safety. No plotting dependency: the SVG
is written by hand, so the same input gives the same bytes.
"""
from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Final

from anjo.sleep.reports.results import ARMS, FULL_SET, LABELS, OVERALL, Night, dimensions, load, pair_cell

WIDTH: Final = 720
PANEL_H: Final = 220
STRIP_H: Final = 16
PAD: Final = 48
# One hue per arm; arm 3 (the headline) is drawn heaviest. Colors are the chart's, not the site's.
COLORS: Final = {"1": "#6b7280", "2C": "#2563eb", "2F": "#0891b2", "3": "#dc2626", "3x": "#d97706", "4": "#16a34a"}


def _x(i: int, n: int) -> float:
    return PAD + (WIDTH - 2 * PAD) * (0.5 if n == 1 else i / (n - 1))


def _y(score: float) -> float:
    return PANEL_H - PAD / 2 - (PANEL_H - PAD) * max(0.0, min(1.0, score))


def _line(nights: Sequence[Night], arm: str, dim: str) -> str:
    """One polyline a probe set, in the order the sets first appear."""
    found = []
    for probes in dict.fromkeys(n.probes for n in nights):
        points = [(i, n.arms[arm][dim].score) for i, n in enumerate(nights)
                  if n.probes == probes and dim in n.arms.get(arm, {})]
        if not points:
            continue
        coords = " ".join(f"{_x(i, len(nights)):.1f},{_y(v):.1f}" for i, v in points)
        width = 3 if arm == "3" else 1.5
        dashed = "" if probes == FULL_SET else ' stroke-dasharray="5 3"'
        found.append(f'<polyline fill="none" stroke="{COLORS[arm]}" stroke-width="{width}"{dashed} points="{coords}">'
                     f"<title>arm {arm} {escape(dim)} ({escape(probes)})</title></polyline>")
    return "".join(found)


def _strip(nights: Sequence[Night], gate: str, top: float) -> list[str]:
    parts = [f'<text x="4" y="{top + 12:.0f}" font-size="11">{escape(gate)}</text>']
    for i, n in enumerate(nights):
        found = n.gates.get(gate.split()[0])
        skipped = bool(found and found.get("skipped"))  # switched off: amber, never the pass green
        color = "#9ca3af" if not found else "#d97706" if skipped else ("#16a34a" if found.get("passed") else "#dc2626")
        parts.append(f'<rect x="{_x(i, len(nights)) - 6:.1f}" y="{top:.0f}" width="12" height="{STRIP_H}" fill="{color}">'
                     f"<title>{escape(n.night)}{' (skipped)' if skipped else ''}</title></rect>")
    return parts


def _panel(label: str, people: Sequence[str], loaded: dict[str, list[Night]], top: float) -> tuple[list[str], float]:
    parts = [f'<text x="4" y="{top + 14:.0f}" font-size="14" font-weight="bold">{label.upper()}</text>']
    y = top + 24
    for person in people:
        nights = [n for n in loaded[person] if n.label == label]
        if not nights:
            continue
        dims = dimensions(nights) or ("score",)
        parts.append(f'<text x="{PAD}" y="{y + 12:.0f}" font-size="12">{escape(person)} ({escape("/".join(dims))}, '
                     f'{len(nights)} nights)</text>')
        parts.append(f'<g transform="translate(0,{y + 16:.0f})">'
                     f'<rect x="{PAD}" y="{PAD / 2}" width="{WIDTH - 2 * PAD}" height="{PANEL_H - PAD}" fill="none" '
                     f'stroke="#d1d5db"/>')
        parts += [_line(nights, arm, dims[0]) for arm in ARMS]
        parts.append("</g>")
        legend = "  ".join(f'<tspan fill="{COLORS[a]}">arm {a}</tspan>' for a in ARMS)
        parts.append(f'<text x="{PAD}" y="{y + 16 + PANEL_H + 4:.0f}" font-size="11">{legend}</text>')
        strip_top = y + PANEL_H + 24
        parts += _strip(nights, "general skills", strip_top) + _strip(nights, "safety", strip_top + STRIP_H + 6)
        y = strip_top + 2 * STRIP_H + 24
        latest = next((n for n in reversed(nights) if n.pairs), None)
        for name, stats in sorted(latest.pairs.items() if latest else ()):
            stat = stats.get(OVERALL)
            if stat is not None:
                parts.append(f'<text x="{PAD}" y="{y:.0f}" font-size="11">{escape(name)} {escape(pair_cell(stat))}</text>')
                y += 16
    return parts, y


def render(people: Sequence[str], *, out: Path, root: Path | None = None) -> Path:
    """Write the chart for ``people`` to ``out`` and return it."""
    loaded = {person: load(person, root) for person in people}
    parts: list[str] = []
    y = 8.0
    for label in LABELS:
        if any(n.label == label for p in people for n in loaded[p]):
            panel, y = _panel(label, people, loaded, y)
            parts += panel
    height = max(y, 40.0)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height:.0f}" '
           f'viewBox="0 0 {WIDTH} {height:.0f}" font-family="sans-serif">' + "".join(parts) + "</svg>\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    return out

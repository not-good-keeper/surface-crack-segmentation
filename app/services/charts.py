"""Server-rendered SVG charts for the analytics dashboard.

Every screen in this application renders complete on the server and reads with
scripting disabled (see app/routes/pages.py). Charts follow the same rule: they are
static SVG markup built here from plain numbers, not a client-side charting library.
That also keeps them offline - no CDN script, nothing fetched at draw time (NFR-05).

Colour is never the only carrier of meaning here, matching the rule already followed
by the status badges and the result banner (see components.css): every mark also
carries a text value, and every chart embeds an SVG <title> so its value is available
on hover/focus without JavaScript. Corners are square throughout, not rounded, to
match every other surface in this design system (.card, .panel, .button, .status).
"""

from __future__ import annotations

from html import escape as _esc
from typing import Any

Bar = dict[str, Any]  # {label, value, color, value_label?}
Segment = dict[str, Any]  # {label, value, color}
Point = tuple[float, float]  # (x, y) in data units


def _fmt(value: float) -> str:
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.1f}"


def empty_state(message: str, *, width: int = 560, height: int = 140) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{_esc(message)}">'
        f'<text x="{width / 2:.0f}" y="{height / 2:.0f}" text-anchor="middle" '
        f'dominant-baseline="middle" class="chart-empty">{_esc(message)}</text></svg>'
    )


def legend(items: list[dict[str, str]]) -> str:
    """A `<ul>` legend to sit beside/under a chart - the dependable identity channel.

    Not embedded in the SVG: a real list reads better with a screen reader and
    without JS or CSS, matching how the rest of the interface degrades.
    """
    chips = "".join(
        f'<li class="legend__item"><span class="legend__swatch" style="background:{item["color"]}" '
        f'aria-hidden="true"></span>{_esc(item["label"])}</li>'
        for item in items
    )
    return f'<ul class="legend">{chips}</ul>'


def stacked_bar(segments: list[Segment], *, width: int = 560, height: int = 48, title: str = "") -> str:
    """One horizontal bar split proportionally - part-to-whole for a single entity.

    Used for a single session's (or the whole deployment's) outcome mix: clean vs.
    regions found vs. the two failure kinds. A 2px surface gap separates touching
    segments instead of a border, per the mark spec this dashboard follows.
    """
    total = sum(max(0.0, float(s["value"])) for s in segments)
    if total <= 0:
        return empty_state("No runs yet", width=width, height=height)

    present = [s for s in segments if float(s["value"]) > 0]
    gap = 2
    usable = width - gap * max(0, len(present) - 1)
    parts: list[str] = []
    x = 0.0
    for segment in present:
        value = float(segment["value"])
        w = round(usable * value / total)
        pct = value / total * 100
        label = _esc(str(segment["label"]))
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{w}" height="{height}" fill="{segment["color"]}">'
            f"<title>{label}: {_fmt(value)} ({pct:.0f}%)</title></rect>"
        )
        x += w + gap
    aria = _esc(title or "Outcome mix")
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="{aria}">' + "".join(parts) + "</svg>"
    )


def bar_chart(
    bars: list[Bar],
    *,
    width: int = 560,
    height: int = 200,
    max_value: float | None = None,
    unit: str = "",
) -> str:
    """Vertical bars, one category per bar - direct-labeled; the axis names each one.

    Used for small category counts (e.g. crack vs. scratch) and for comparing one
    measure (e.g. clean %) across a handful of sessions.
    """
    if not bars:
        return empty_state("No data yet", width=width, height=height)

    values = [max(0.0, float(b["value"])) for b in bars]
    vmax = max_value if max_value is not None else max(values)
    vmax = vmax or 1.0

    pad_top, pad_bottom, pad_x = 24, 22, 10
    plot_h = height - pad_top - pad_bottom
    n = len(bars)
    slot = (width - pad_x * 2) / n
    bar_w = min(24.0, slot * 0.55)
    baseline = pad_top + plot_h

    parts = [f'<line x1="{pad_x}" y1="{baseline}" x2="{width - pad_x}" y2="{baseline}" class="chart-axis"/>']
    for i, bar in enumerate(bars):
        value = max(0.0, float(bar["value"]))
        h = round(plot_h * value / vmax) if vmax else 0
        h = max(h, 1) if value > 0 else 0
        cx = pad_x + slot * i + slot / 2
        x = cx - bar_w / 2
        y = baseline - h
        color = bar.get("color") or "var(--ink-2)"
        label = _esc(str(bar["label"]))
        value_label = str(bar.get("value_label") or f"{_fmt(value)}{unit}")
        if h > 0:
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h}" fill="{color}">'
                f"<title>{label}: {value_label}</title></rect>"
            )
        text_y = (y - 6) if h > 0 else (baseline - 6)
        parts.append(f'<text x="{cx:.1f}" y="{text_y:.1f}" text-anchor="middle" class="chart-value">{_esc(value_label)}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{baseline + 15}" text-anchor="middle" class="chart-tick">{label}</text>')

    return f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">' + "".join(parts) + "</svg>"


def grouped_bar_chart(
    categories: list[str],
    series: list[dict[str, Any]],
    *,
    width: int = 560,
    height: int = 200,
    unit: str = "",
) -> str:
    """Grouped vertical bars: one group per category, one bar per series within it.

    `series` is `[{"name": "Crack", "color": ..., "values": [...]}]`, aligned to
    `categories` by index. Used to tell crack and scratch counts apart across sessions.
    """
    if not categories or not series:
        return empty_state("No data yet", width=width, height=height)

    vmax = max((v for s in series for v in s["values"]), default=0) or 1.0
    pad_top, pad_bottom, pad_x = 24, 22, 10
    plot_h = height - pad_top - pad_bottom
    baseline = pad_top + plot_h
    n = len(categories)
    slot = (width - pad_x * 2) / n
    group_w = slot * 0.7
    bar_gap = 2
    bar_w = max(3.0, (group_w - bar_gap * (len(series) - 1)) / len(series))

    parts = [f'<line x1="{pad_x}" y1="{baseline}" x2="{width - pad_x}" y2="{baseline}" class="chart-axis"/>']
    for i, cat in enumerate(categories):
        group_x0 = pad_x + slot * i + (slot - group_w) / 2
        for j, s in enumerate(series):
            value = max(0.0, float(s["values"][i]))
            h = round(plot_h * value / vmax) if value else 0
            x = group_x0 + j * (bar_w + bar_gap)
            y = baseline - h
            if h > 0:
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h}" fill="{s["color"]}">'
                    f"<title>{_esc(s['name'])} — {_esc(cat)}: {_fmt(value)}{unit}</title></rect>"
                )
        cx = pad_x + slot * i + slot / 2
        parts.append(f'<text x="{cx:.1f}" y="{baseline + 15}" text-anchor="middle" class="chart-tick">{_esc(cat)}</text>')

    return f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">' + "".join(parts) + "</svg>"


def line_area(
    values: list[float],
    labels: list[str],
    *,
    width: int = 560,
    height: int = 180,
    color: str = "var(--ink-2)",
    unit: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    """A single-series trend line with a light area wash under it.

    Used for a run's cumulative processed count over elapsed time, and for a measure
    (e.g. clean %) trending across recent sessions in chronological order. Only the
    endpoint is direct-labeled, per the "label selectively" rule - a value on every
    point would go unread on anything but the shortest run.
    """
    if len(values) < 2:
        return empty_state("Not enough data yet", width=width, height=height)

    lo = y_min if y_min is not None else min(values)
    hi = y_max if y_max is not None else max(values)
    if hi <= lo:
        hi = lo + 1

    pad_top, pad_bottom, pad_x = 20, 22, 10
    plot_w = width - pad_x * 2
    plot_h = height - pad_top - pad_bottom
    n = len(values)
    step = plot_w / (n - 1)

    def px(i: int) -> float:
        return pad_x + step * i

    def py(v: float) -> float:
        return pad_top + plot_h * (1 - (v - lo) / (hi - lo))

    pts = [(px(i), py(v)) for i, v in enumerate(values)]
    line_d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    area_d = line_d + f" L {pts[-1][0]:.1f} {pad_top + plot_h:.1f} L {pts[0][0]:.1f} {pad_top + plot_h:.1f} Z"

    baseline = pad_top + plot_h
    last_x, last_y = pts[-1]
    first_x, first_y = pts[0]
    end_label = f"{_fmt(values[-1])}{unit}"

    parts = [
        f'<line x1="{pad_x}" y1="{baseline}" x2="{width - pad_x}" y2="{baseline}" class="chart-axis"/>',
        f'<path d="{area_d}" fill="{color}" opacity="0.1" stroke="none"/>',
        f'<path d="{line_d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>',
        f'<circle cx="{first_x:.1f}" cy="{first_y:.1f}" r="4" fill="{color}" stroke="var(--paper)" stroke-width="2">'
        f"<title>{_esc(labels[0])}: {_fmt(values[0])}{unit}</title></circle>",
        f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{color}" stroke="var(--paper)" stroke-width="2">'
        f"<title>{_esc(labels[-1])}: {_fmt(values[-1])}{unit}</title></circle>",
        f'<text x="{last_x:.1f}" y="{last_y - 10:.1f}" text-anchor="end" class="chart-value">{_esc(end_label)}</text>',
        f'<text x="{pad_x}" y="{baseline + 15}" text-anchor="start" class="chart-tick">{_esc(labels[0])}</text>',
        f'<text x="{width - pad_x}" y="{baseline + 15}" text-anchor="end" class="chart-tick">{_esc(labels[-1])}</text>',
    ]
    return f'<svg class="chart" viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">' + "".join(parts) + "</svg>"

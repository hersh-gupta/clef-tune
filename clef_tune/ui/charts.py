"""SVG charts for the UI: label distributions, token lengths, training curves and confusion
matrices. They carry CSS classes, never colours, so style.css themes them in light and dark
mode alike. Every label is escaped: the text comes from uploaded data."""

from __future__ import annotations

import math
from html import escape

from clef_tune.card import top_confusion

LABEL_CHARS = 24
AXIS_LABEL_CHARS = 12
MIN_CELL_OPACITY = 0.06


def _short(text: object, limit: int = LABEL_CHARS) -> str:
    full = str(text)
    return escape(full if len(full) <= limit else full[: limit - 1] + "…")


def _empty(message: str) -> str:
    return f'<div class="ct-empty">{escape(message)}</div>'


def _svg(kind: str, width: float, height: float, label: str, body: str) -> str:
    return (
        f'<svg class="ct-chart ct-{kind}" viewBox="0 0 {width:g} {height:g}" role="img" '
        f'aria-label="{escape(label)}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def bar_chart(counts: dict[str, int], *, missing: int = 0) -> str:
    """Horizontal bars, one per option, with the count and share at the end of each."""
    rows = [(label, count, False) for label, count in counts.items()]
    if missing:
        rows.append(("unlabeled", missing, True))
    if not rows:
        return _empty("No labels yet")
    total = sum(count for _, count, _ in rows) or 1
    top = max(count for _, count, _ in rows) or 1
    left, right, row_height, bar_height = 140.0, 400.0, 28.0, 18.0
    parts = []
    for index, (label, count, muted) in enumerate(rows):
        y = 8 + index * row_height
        width = (right - left) * count / top
        if count:
            width = max(width, 2.0)
        bar_class = "ct-bar ct-bar-muted" if muted else "ct-bar"
        share = f"{count / total:.0%}"
        parts.append(
            f"<g><title>{escape(str(label))}: {count:,} ({share})</title>"
            f'<text class="ct-label" x="{left - 8:g}" y="{y + 13:g}" text-anchor="end">'
            f"{_short(label, 20)}</text>"
            f'<rect class="ct-track" x="{left:g}" y="{y:g}" width="{right - left:g}" '
            f'height="{bar_height:g}" rx="4"/>'
            f'<rect class="{bar_class}" x="{left:g}" y="{y:g}" width="{width:.1f}" '
            f'height="{bar_height:g}" rx="4"/>'
            f'<text class="ct-value" x="{left + width + 6:.1f}" y="{y + 13:g}">'
            f"{count:,} · {share}</text></g>"
        )
    height = 16 + row_height * len(rows)
    return _svg("bars", 480, height, "Label counts per option", "".join(parts))


def histogram(values: list[int], *, cutoff: int | None = None, bins: int = 24) -> str:
    """Token lengths in equal-width bins, with a dashed line at the truncation length."""
    if not values:
        return _empty("No token counts yet")
    top = max(max(values), cutoff or 0, 1)
    width_per_bin = max(1, math.ceil(top / bins))
    counts = [0] * bins
    for value in values:
        counts[min(value // width_per_bin, bins - 1)] += 1
    left, right, plot_top, bottom = 44.0, 468.0, 18.0, 172.0
    peak = max(counts) or 1
    step = (right - left) / bins
    x_of = lambda tokens: left + (right - left) * min(tokens, top) / top  # noqa: E731
    parts = [
        f'<line class="ct-axis" x1="{left:g}" y1="{bottom:g}" x2="{right:g}" y2="{bottom:g}"/>',
        f'<text class="ct-tick" x="{left - 6:g}" y="{plot_top + 4:g}" text-anchor="end">'
        f"{peak:,}</text>",
        f'<text class="ct-tick" x="{left - 6:g}" y="{bottom:g}" text-anchor="end">0</text>',
    ]
    for index, count in enumerate(counts):
        if not count:
            continue
        low, high = index * width_per_bin, (index + 1) * width_per_bin
        height = (bottom - plot_top) * count / peak
        over = cutoff is not None and low >= cutoff
        css = "ct-hbar ct-hbar-over" if over else "ct-hbar"
        parts.append(
            f"<g><title>{low:,}–{high:,} tokens: {count:,} rows</title>"
            f'<rect class="{css}" x="{left + index * step + 1:.1f}" y="{bottom - height:.1f}" '
            f'width="{max(step - 2, 1):.1f}" height="{height:.1f}" rx="2"/></g>'
        )
    for tokens in (0, top // 2, top):
        parts.append(
            f'<text class="ct-tick" x="{x_of(tokens):.1f}" y="{bottom + 16:g}" '
            f'text-anchor="middle">{tokens:,}</text>'
        )
    if cutoff is not None:
        x = x_of(cutoff)
        parts.append(
            f'<line class="ct-cutoff" x1="{x:.1f}" y1="{plot_top - 6:g}" '
            f'x2="{x:.1f}" y2="{bottom:g}"/>'
            f'<text class="ct-cutoff-label" x="{x - 4:.1f}" y="{plot_top + 4:g}" '
            f'text-anchor="end">max_length {cutoff}</text>'
        )
    return _svg("histogram", 480, 200, "Prompt length in tokens", "".join(parts))


def line_chart(
    series: dict[str, list[tuple[float, float]]],
    *,
    reference: dict[str, float] | None = None,
    y_label: str = "",
) -> str:
    """One line per series, plus a dashed line per reference value (such as the baseline)."""
    reference = reference or {}
    points = [point for values in series.values() for point in values]
    if not points:
        return _empty("No points yet")
    xs = [x for x, _ in points]
    ys = [y for _, y in points] + list(reference.values())
    x_lo, x_hi = min(xs), max(xs)
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 1, x_hi + 1
    y_lo = 0.0 if min(ys) >= 0 else min(ys)
    y_hi = max(ys) * 1.08 if max(ys) > 0 else 1.0
    if y_hi <= y_lo:
        y_hi = y_lo + 1
    left, right, plot_top, bottom = 44.0, 464.0, 34.0, 192.0

    def place(x: float, y: float) -> tuple[float, float]:
        return (
            left + (right - left) * (x - x_lo) / (x_hi - x_lo),
            bottom - (bottom - plot_top) * (y - y_lo) / (y_hi - y_lo),
        )

    parts = []
    for i in range(5):
        value = y_lo + (y_hi - y_lo) * i / 4
        _, y = place(x_lo, value)
        parts.append(
            f'<line class="ct-grid" x1="{left:g}" y1="{y:.1f}" x2="{right:g}" y2="{y:.1f}"/>'
            f'<text class="ct-tick" x="{left - 6:g}" y="{y + 4:.1f}" text-anchor="end">'
            f"{_fmt(value)}</text>"
        )
    for value in (x_lo, x_hi):
        x, _ = place(value, y_lo)
        parts.append(
            f'<text class="ct-tick" x="{x:.1f}" y="{bottom + 16:g}" text-anchor="middle">'
            f"{_fmt(value)}</text>"
        )
    legend_x = left
    for index, (name, values) in enumerate(series.items()):
        if values:
            coords = " ".join(f"{px:.1f},{py:.1f}" for px, py in (place(x, y) for x, y in values))
            last_x, last_y = place(*values[-1])
            parts.append(
                f'<polyline class="ct-line ct-series-{index}" points="{coords}" fill="none"/>'
                f'<circle class="ct-dot ct-series-{index}" cx="{last_x:.1f}" cy="{last_y:.1f}" '
                f'r="3"><title>{escape(name)}: {_fmt(values[-1][1])}</title></circle>'
            )
        parts.append(
            f'<rect class="ct-swatch ct-series-{index}" x="{legend_x:g}" y="8" width="10" '
            f'height="10" rx="2"/><text class="ct-legend" x="{legend_x + 14:g}" y="17">'
            f"{_short(name, 20)}</text>"
        )
        legend_x += 28 + 7 * min(len(name), 20)
    for index, (name, value) in enumerate(reference.items()):
        _, y = place(x_lo, value)
        parts.append(
            f'<line class="ct-reference ct-series-{index}" x1="{left:g}" y1="{y:.1f}" '
            f'x2="{right:g}" y2="{y:.1f}"/>'
            f'<text class="ct-reference-label" x="{right:g}" y="{y - 5:.1f}" text-anchor="end">'
            f"{_short(name, 20)} {_fmt(value)}</text>"
        )
    if y_label:
        parts.append(
            f'<text class="ct-legend" x="{right:g}" y="17" text-anchor="end">'
            f"{_short(y_label)}</text>"
        )
    return _svg("lines", 480, 220, y_label or ", ".join(series), "".join(parts))


def _opacity(share: float) -> str:
    if share <= 0:
        return "0"
    return f"{max(share, MIN_CELL_OPACITY):.2f}".rstrip("0").rstrip(".")


def heatmap(labels: list[str], matrix: list[list[int]]) -> str:
    """A confusion matrix: rows are true labels, columns predictions; each cell is shaded by
    its share of the row."""
    names, rows = top_confusion(labels, matrix)
    n = len(names)
    if n == 0:
        return _empty("No predictions yet")
    cell, left, top = 40.0, 80.0, 60.0
    parts = [
        f'<text class="ct-axis-title" x="6" y="14">predicted →</text>'
        f'<text class="ct-axis-title" x="6" y="{top - 6:g}">true ↓</text>'
    ]
    for j, name in enumerate(names):
        x = left + j * cell + cell / 2
        parts.append(
            f'<text class="ct-label" x="{x:g}" y="{top - 8:g}" text-anchor="start" '
            f'transform="rotate(-40 {x:g} {top - 8:g})"><title>{escape(name)}</title>'
            f"{_short(name, AXIS_LABEL_CHARS)}</text>"
        )
    for i, (name, row) in enumerate(zip(names, rows, strict=True)):
        y = top + i * cell
        total = sum(row)
        parts.append(
            f'<text class="ct-label" x="{left - 8:g}" y="{y + cell / 2 + 4:g}" text-anchor="end">'
            f"<title>{escape(name)}</title>{_short(name, AXIS_LABEL_CHARS)}</text>"
        )
        for j, count in enumerate(row):
            x = left + j * cell
            share = count / total if total else 0.0
            strong = " is-strong" if share > 0.55 else ""
            diagonal = " is-diagonal" if i == j else ""
            parts.append(
                f"<g><title>true {escape(name)}, predicted {escape(names[j])}: {count:,} "
                f"({share:.0%} of {escape(name)})</title>"
                f'<rect class="ct-cell-bg{diagonal}" x="{x + 1:g}" y="{y + 1:g}" '
                f'width="{cell - 2:g}" height="{cell - 2:g}" rx="4"/>'
                f'<rect class="ct-cell" x="{x + 1:g}" y="{y + 1:g}" width="{cell - 2:g}" '
                f'height="{cell - 2:g}" rx="4" fill-opacity="{_opacity(share)}"/>'
                f'<text class="ct-cell-value{strong}" x="{x + cell / 2:g}" '
                f'y="{y + cell / 2 + 4:g}" text-anchor="middle">{count:,}</text></g>'
            )
    width, height = left + cell * n + 8, top + cell * n + 8
    return _svg("heatmap", width, height, "Confusion matrix", "".join(parts))

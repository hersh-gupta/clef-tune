"""Step 6: the LinkedIn chart, before -> after dumbbells in the Space's style (1080x1350, 4:5).

    uv run python benchmarks/cfpb/chart.py            # results/chart.html and results/chart.png

The page reads results/metrics.json. With `#play` in its address it animates (the video's
benchmark scene uses that); without, it is drawn still for the PNG.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cfpb.common import DATA, REPO, RESULTS

CSS = REPO / "clef_tune" / "ui" / "static" / "clef-tune.css"
ROWS = [  # (name, before contender, after contender): the frontier LLM first, as the yardstick
    ("Claude Opus 5.5", "claude-zero", "claude-few"),
    ("Clef-Flash", "clef-flash-untrained", "clef-flash-tuned"),
    ("Clef 27B", "clef-27b-untrained", "clef-27b-tuned"),
]
TITLE = "Small decision models tuned on past data matched or beat a frontier LLM"  # results


def money(x: float) -> str:
    return f"${x:,.2f}"


def cost_line(contenders: dict, before: str, after: str) -> str:
    """The cost of the filled dot: the tuned model, or Claude with examples."""
    cost = (contenders.get(after) or {}).get("cost") or {}
    if not cost.get("usd_per_1000"):
        return ""
    kind = "with examples" if after.startswith("claude") else "tuned"
    return f"{money(cost['usd_per_1000'])} per 1,000 complaints, {kind}"


def page(metrics: dict, title: str, css: str | None = None) -> str:
    c = metrics["contenders"]
    rows = []
    for k, (name, before, after) in enumerate(ROWS):
        if before not in c or after not in c:
            continue
        b, a = c[before], c[after]
        bx, ax = 100 * b["all_three_right"], 100 * a["all_three_right"]
        bl, bh = (100 * v for v in b["all_three_right_95"])
        al, ah = (100 * v for v in a["all_three_right_95"])
        lo, hi = min(bx, ax), max(bx, ax)
        origin = "left center" if ax >= bx else "right center"  # the line grows with the dot
        # Labels sit above their dots; when the dots are close, push them apart sideways.
        close = hi - lo < 10
        b_shift = (
            "translateX(calc(-100% + 12px))"
            if close and bx <= ax
            else ("translateX(-12px)" if close else "translateX(-50%)")
        )
        a_shift = (
            "translateX(-12px)"
            if close and bx <= ax
            else ("translateX(calc(-100% + 12px))" if close else "translateX(-50%)")
        )
        rows.append(f"""
      <div class="row" style="--i:{k}">
        <div class="nm">{html.escape(name)}<small>{cost_line(c, before, after)}</small></div>
        <div class="track">
          <i class="grid" style="left:0"></i><i class="grid" style="left:25%"></i>
          <i class="grid" style="left:50%"></i><i class="grid" style="left:75%"></i>
          <i class="grid" style="left:100%"></i>
          <i class="ci b" style="left:{bl:.2f}%;width:{bh - bl:.2f}%"></i>
          <i class="ci a" style="left:{al:.2f}%;width:{ah - al:.2f}%;--from:{bx:.2f}%"></i>
          <i class="ln" style="left:{lo:.2f}%;width:{hi - lo:.2f}%;transform-origin:{origin}"></i>
          <i class="dot b" style="left:{bx:.2f}%"></i>
          <i class="dot a" style="left:{ax:.2f}%;--from:{bx:.2f}%"></i>
          <span class="dl b" style="left:{bx:.2f}%;transform:{b_shift}">{bx:.0f}%</span>
          <span class="dl a"
            style="left:{ax:.2f}%;--from:{bx:.2f}%;transform:{a_shift}">{ax:.0f}%</span>
        </div>
      </div>""")
    first, last = metrics["test_dates"]
    n = metrics["test_rows"]
    training = []
    for name, _, after in ROWS[1:]:
        cost = (c.get(after) or {}).get("cost") or {}
        if cost.get("training_usd") is not None:
            training.append(f"{name} ${cost['training_usd']:.2f}")
    price_date = next(
        (
            (x.get("cost") or {}).get("prices", {}).get("date")
            for x in c.values()
            if (x.get("cost") or {}).get("prices")
        ),
        None,
    )
    month = date.fromisoformat(first).strftime("%B %Y")
    foot = (
        f"{n} complaints to a large US bank from {month}, after Claude's training data, never "
        "used to train or prompt any model here. Shaded bars: 95% ranges."
    )
    if training:
        foot += " One-off training: " + ", ".join(training) + "."
    if price_date:
        foot += f" Prices as of {price_date}."
    foot += " Data: CFPB (CC0)."
    return f"""<!doctype html>
<html lang="en" data-theme="light"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500&display=swap"
  rel="stylesheet">
<link rel="stylesheet" href="{css or CSS.as_uri()}">
<style>
  html, body {{ margin: 0; width: 1080px; height: 1350px; overflow: hidden;
    background: var(--page); }}
  .canvas {{ position: relative; width: 1080px; height: 1350px; padding: 62px 76px 0; }}
  .kicker {{ display: flex; align-items: center; gap: 16px; font-size: 24px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--muted); }}
  .kicker .mark {{ width: 20px; height: 20px; border-width: 3px; }}
  h1 {{ font-weight: 300; font-size: 54px; line-height: 1.18; margin: 30px 0 22px; }}
  .accent {{ width: 110px; height: 3px; background: var(--oak); margin: 0 0 22px; }}
  .sub {{ font-size: 27px; line-height: 1.4; color: var(--muted); margin: 0 0 22px; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 14px 40px; font-size: 25px; color: var(--muted);
    margin: 0 0 26px; }}
  .legend i {{ display: inline-block; width: 24px; height: 24px; border-radius: 50%;
    vertical-align: -4px; margin-right: 12px; box-sizing: border-box; }}
  .legend i.b {{ border: 4px solid var(--oak); }} .legend i.a {{ background: var(--sage); }}
  .row {{ margin: 0 0 18px; }}
  .nm {{ font-size: 34px; font-weight: 400; }}
  .nm small {{ display: block; font-size: 24px; color: var(--muted); margin-top: 2px; }}
  .track {{ position: relative; height: 96px; margin: 0 14px; }}
  .track .grid {{ position: absolute; top: 46px; bottom: 0; width: 1px;
    background: var(--line-soft); }}
  .track::after {{ content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
    background: var(--line); }}
  .ci {{ position: absolute; top: 56px; height: 14px; border-radius: 7px; }}
  .ci.b {{ background: var(--oak); opacity: .18; }}
  .ci.a {{ background: var(--sage); opacity: .22; }}
  .ln {{ position: absolute; top: 60px; height: 6px; background: #8E9C86; border-radius: 3px; }}
  .dot {{ position: absolute; top: 46px; width: 34px; height: 34px; margin-left: -17px;
    border-radius: 50%; box-sizing: border-box; box-shadow: 0 0 0 3px var(--page); }}
  .dot.b {{ border: 5px solid var(--oak); background: var(--page); }}
  .dot.a {{ background: var(--sage); }}
  .dl {{ position: absolute; top: 0; font-size: 31px; white-space: nowrap; }}
  .dl.b {{ color: var(--oak-ink); }} .dl.a {{ color: var(--sage-ink); font-weight: 500; }}
  .axis {{ display: flex; justify-content: space-between; margin: 6px 14px 0; font-size: 22px;
    color: var(--muted); }}
  .foot {{ position: absolute; left: 76px; right: 76px; bottom: 44px; padding-top: 18px;
    border-top: 2px solid var(--line); font-size: 21px; line-height: 1.45; color: var(--muted); }}

  /* the video's build-up: each row rises with its name and cost, then its "after" dot slides
     out from its "before" dot, the line between them growing with it */
  html:not(.play) * {{ animation: none !important; }}
  .play .row {{ animation: rise .9s var(--ease) backwards;
    animation-delay: calc(var(--i) * 1.6s + .3s); }}
  .play .dot.a, .play .dl.a {{ animation: slide 1.2s var(--ease) backwards;
    animation-delay: calc(var(--i) * 1.6s + 1s); }}
  .play .ln {{ animation: grow 1.2s var(--ease) backwards;
    animation-delay: calc(var(--i) * 1.6s + 1s); }}
  .play .ci.a {{ animation: fadein .8s ease backwards;
    animation-delay: calc(var(--i) * 1.6s + 2s); }}
  .play .foot {{ animation: fadein 1s ease backwards; animation-delay: 6s; }}
</style>
<script>if (location.hash === "#play") document.documentElement.classList.add("play");</script>
</head>
<body><div class="canvas">
  <div class="kicker"><span class="mark"></span>clef-tune benchmark</div>
  <h1>{html.escape(title)}</h1>
  <div class="accent"></div>
  <p class="sub">How often all three answers were right: the issue, the product, and the bank's
    outcome</p>
  <div class="legend"><span><i class="b"></i>Out of the box, or instructions only</span>
    <span><i class="a"></i>Tuned, or given 30 examples</span></div>
  {"".join(rows)}
  <div class="axis"><span>0%</span><span>50%</span><span>100%</span></div>
  <div class="foot">{html.escape(foot)}</div>
</div></body></html>
"""


def render_png(html_path: Path, png_path: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        tab = browser.new_page(viewport={"width": 1080, "height": 1350})
        tab.goto(html_path.as_uri())
        tab.evaluate("document.fonts.ready")
        tab.wait_for_timeout(500)
        tab.screenshot(path=str(png_path))
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--title", default=TITLE)
    args = parser.parse_args()
    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    out = DATA / "chart.html"  # git-ignored: it links the stylesheet by a local path
    out.write_text(page(metrics, args.title), encoding="utf-8")
    render_png(out, RESULTS / "chart.png")
    print("->", RESULTS / "chart.png")


if __name__ == "__main__":
    main()

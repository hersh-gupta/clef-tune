"""Step 7: the LinkedIn video (1080x1350 MP4, about 65 s) and the README GIF.

    uv run python benchmarks/cfpb/video/make_video.py fetch-run   # the real run's files
    uv run python benchmarks/cfpb/video/make_video.py video       # results/video.mp4
    uv run python benchmarks/cfpb/video/make_video.py gif         # results/benchmark.gif

The real web UI runs locally against fake services that replay the real Clef-Flash run
(video/services.py), inside a 4:5 stage with a large caption on top, since LinkedIn plays videos
muted. Chromium renders at 2x and its screencast frames are encoded with ffmpeg.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.cfpb.common import DATA, PREPARED, RESULTS, machine_env  # noqa: E402

WIDTH, HEIGHT, SCALE = 540, 675, 2  # CSS pixels; frames are 1080x1350
FPS = 30
LENGTH = 57.0  # seconds
RUN_DIR = DATA / "video" / "run"
EXAMPLE = ROOT / "examples" / "support_tickets.yaml"

CAPTIONS = {
    "data": ("1", "Your data", "Load 3,300 bank complaints from the CFPB's public records"),
    "questions": (
        "2",
        "Your decisions",
        "Lay out the decisions the model should make, in plain words",
    ),
    "train": ("3", "Training", "Configure training, and see the cost and time before you start"),
    "run": ("4", "Training", "It trains in your own private account, so your data stays yours"),
    "results": ("5", "Results", "Measured on complaints it never saw"),
}
ASKS = {
    "issue": "What is this complaint mainly about?",
    "product": "Which product is the complaint about?",
    "outcome": "How did the bank close this complaint?",
}

STAGE = """<!doctype html>
<html lang="en" data-theme="light"><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;500&display=swap"
  rel="stylesheet">
<link rel="stylesheet" href="/static/clef-tune.css">
<style>
  html, body { margin: 0; width: 540px; height: 675px; overflow: hidden; background: var(--page); }
  #cap { position: absolute; left: 0; right: 0; top: 0; height: 150px; padding: 30px 30px 0; }
  #cap .k { display: flex; align-items: center; gap: 9px; font-size: 11.5px; letter-spacing: .14em;
    text-transform: uppercase; color: var(--muted); }
  #cap .k b { color: var(--sage-ink); font-weight: 500; }
  #cap .k .dotsep { color: var(--line); }
  #cap .k .mark { width: 10px; height: 10px; }
  #cap .line { font-size: 25px; font-weight: 300; line-height: 1.25; margin-top: 11px; }
  #cap .sped { display: inline-block; margin-left: 10px; font-size: 10.5px; letter-spacing: .1em;
    padding: 2px 8px; border: 1px solid var(--oak); border-radius: 10px; color: var(--oak-ink); }
  .rise { animation: rise .7s var(--ease) backwards; }
  #frame { position: absolute; left: 22px; right: 22px; top: 158px; bottom: 0;
    border: 1px solid var(--line); border-bottom: 0; border-radius: 14px 14px 0 0; overflow: hidden;
    background: var(--page); box-shadow: 0 -8px 30px -24px var(--shadow); }
  #ui { width: 100%; height: 100%; border: 0; display: block; }
  .card { position: absolute; inset: 0; background: var(--page); opacity: 0;
    transition: opacity .6s ease; pointer-events: none; }
  .card.on { opacity: 1; }
  #title { padding: 150px 38px 0; }
  #title .k, #close .k { display: flex; align-items: center; gap: 9px; font-size: 12px;
    letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
  #title h1 { font-weight: 300; font-size: 43px; line-height: 1.18; margin: 22px 0 18px; }
  #title .accent, #close .accent { height: 2px; width: 0; background: var(--oak);
    transition: width 1.2s var(--ease) .5s; }
  .card.on .accent { width: 64px !important; }
  #title p { font-size: 19px; color: var(--muted); margin: 22px 0 0; }
  #chartcard iframe { position: absolute; left: 0; top: 0; width: 1080px; height: 1350px; border: 0;
    transform: scale(.5); transform-origin: 0 0; }
  #close { padding: 170px 38px 0; }
  #close h1 { font-weight: 300; font-size: 40px; line-height: 1.2; margin: 22px 0 18px; }
  #close .lk { display: block; font-size: 20px; color: var(--sage-ink); margin: 28px 0 0; }
  #close .lk span { border-bottom: 1px solid var(--line); }
  #close .sm { font-size: 14.5px; line-height: 1.55; color: var(--muted); margin-top: 46px;
    max-width: 440px; }
</style></head>
<body>
  <div id="cap"></div>
  <div id="frame"><iframe id="ui" src="about:blank"></iframe></div>
  <div id="title" class="card">
    <div class="k"><span class="mark"></span>clef-tune benchmark</div>
    <h1>Can a small decision model, tuned on real data, beat a frontier LLM?</h1>
    <div class="accent"></div>
    <p>Trained and tested on public bank complaints from the CFPB</p>
  </div>
  <div id="chartcard" class="card"><iframe id="chart" src="about:blank"></iframe></div>
  <div id="close" class="card">
    <div class="k"><span class="mark"></span>clef-tune</div>
    <h1>Try it on your own data</h1>
    <div class="accent"></div>
    <a class="lk"><span>hvgupta1-clef-tune.hf.space</span></a>
    <a class="lk"><span>github.com/hersh-gupta/clef-tune</span></a>
    <p class="sm">The method, data and models are all public, with the limits spelled out in the
      write-up.</p>
  </div>
<script>
  window.caption = (n, label, line, sped) => {
    const cap = document.getElementById("cap");
    cap.innerHTML = `<div class="k rise"><span class="mark"></span>How it works` +
      `<span class="dotsep">·</span><b>${n}</b>${label}</div>` +
      `<div class="line rise" style="animation-delay:.12s">${line}` +
      (sped ? `<span class="sped">SPED UP</span>` : "") + `</div>`;
  };
  window.card = (id) => {
    document.querySelectorAll(".card").forEach((c) => c.classList.toggle("on", c.id === id));
  };
</script>
</body></html>
"""


# ----------------------------------------------------------------------------------- fetch-run
def fetch_run_command() -> None:
    """The real Clef-Flash run: its training log from the jobs bucket, its metrics and config
    from its Hub repo, and the estimate `clef-tune jobs launch` showed before it started."""
    from huggingface_hub import HfApi, hf_hub_download

    from clef_tune.launch.hf_jobs import _work_volume

    token = machine_env("HF_TOKEN")
    api = HfApi(token=token)
    jobs = json.loads((DATA / "jobs.json").read_text(encoding="utf-8"))
    info = api.inspect_job(job_id=jobs["train-flash"]["id"])
    work = _work_volume(info)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    api.download_bucket_files(
        work.source, [(f"{work.path}/out/metrics.jsonl", RUN_DIR / "metrics.jsonl")]
    )
    repo = "hvgupta1/clef-flash-cfpb-complaints"
    shutil.copy(hf_hub_download(repo, "metrics.json", token=token), RUN_DIR / "metrics.json")
    shutil.copy(hf_hub_download(repo, "train_config.yaml", token=token), RUN_DIR / "config.yaml")
    estimate = {  # as shown by `clef-tune jobs launch` for the run (see benchmarks/README.md)
        "flavor": "rtx-pro-6000",
        "price_per_hour": 2.75,
        "hours_low": 0.5,
        "hours_high": 1.0,
        "cost_low": 1.30,
        "cost_high": 2.78,
        "max_cost": 5.50,
        "timeout": "2h",
    }
    (RUN_DIR / "estimate.json").write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    print("Real run files in", RUN_DIR)


# -------------------------------------------------------------------------------------- record
class Screencast:
    """Chromium's screencast frames, written as JPEGs. Each frame keeps Chromium's own timestamp
    and the time it arrived here; `times()` maps the first onto this script's clock."""

    def __init__(self, page, folder: Path) -> None:
        self.cdp = page.context.new_cdp_session(page)
        self.folder = folder
        self.frames: list[tuple[float, float, Path]] = []  # (chromium time, arrival, file)
        self.cdp.on("Page.screencastFrame", self._frame)

    def _frame(self, params: dict) -> None:
        arrived = time.monotonic()
        path = self.folder / f"{len(self.frames):06d}.jpg"
        path.write_bytes(base64.b64decode(params["data"]))
        self.frames.append((params["metadata"]["timestamp"], arrived, path))
        self.cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

    def start(self) -> None:
        self.cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": 95,
                "maxWidth": WIDTH * SCALE,
                "maxHeight": HEIGHT * SCALE,
                "everyNthFrame": 1,
            },
        )

    def stop(self) -> None:
        self.cdp.send("Page.stopScreencast")

    def times(self, t0: float) -> list[tuple[float, Path]]:
        """Seconds since t0 for each frame. A frame arrives some time after Chromium drew it,
        so the smallest gap between the two clocks is the best estimate of their offset."""
        offset = min(arrived - drawn for drawn, arrived, _ in self.frames)
        return [(max(0.0, drawn + offset - t0), path) for drawn, _, path in self.frames]


def encode(frames: list[tuple[float, Path]], end: float, out: Path) -> None:
    """Frames at their times (seconds from the start) -> a constant 30 fps H.264 MP4."""
    lines = []
    for (t, path), (t_next, _) in zip(frames, [*frames[1:], (end, None)], strict=True):
        lines += [f"file '{path.as_posix()}'", f"duration {max(t_next - t, 0.001):.4f}"]
    lines.append(f"file '{frames[-1][1].as_posix()}'")
    listing = frames[0][1].parent / "frames.txt"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-vf",
            f"fps={FPS},scale={WIDTH * SCALE}:{HEIGHT * SCALE}:flags=lanczos,format=yuv420p",
            "-t",
            f"{end:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "17",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )


def serve_ui(run):
    import uvicorn

    import clef_tune.ui.runs as runs
    from benchmarks.cfpb.video.services import video_services
    from clef_tune.ui.server import create_app
    from clef_tune.ui.settings import Settings
    from scripts.screenshots import free_port

    runs.POLL_SECONDS = 0.5
    os.environ["HF_TOKEN"] = "hf_video_only"  # local mode reads a login; nothing is sent
    settings = Settings(space=False, max_upload_mb=200, local_training=False)
    app = create_app(settings, video_services(run, EXAMPLE), secret="video")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server


def upload_file(folder: Path) -> Path:
    """What the real run learned from, as one spreadsheet: the 3,000 training and 300 validation
    complaints, without the date column. The app sets 300 of them aside to test, as the run did."""
    data, columns = [], None
    for split in ("train", "validation"):
        with (PREPARED / f"{split}.csv").open(encoding="utf-8", newline="") as rows:
            reader = csv.DictReader(rows)
            columns = [c for c in reader.fieldnames if c != "date_received"]
            data += [{c: r[c] for c in columns} for r in reader]
    path = folder / "bank_complaints.csv"
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)
    return path


def chart_html(base: str) -> str:
    from benchmarks.cfpb.chart import TITLE, page

    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    return page(metrics, TITLE, css="/static/clef-tune.css")


def video_command(out: Path, keep_frames: bool) -> None:
    from playwright.sync_api import sync_playwright

    from benchmarks.cfpb.video.services import RealRun

    run = RealRun(RUN_DIR)
    if not run.real:
        print("note: the real run isn't downloaded yet; using the test fixture for a draft")
    test_rows = json.loads((PREPARED / "split.json").read_text(encoding="utf-8"))["rows"]["test"]
    base, server = serve_ui(run)
    frames_dir = Path(tempfile.mkdtemp(prefix="clef-video-"))
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(
                viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=SCALE
            )
            page = context.new_page()
            stage = STAGE.replace("__N__", f"{test_rows:,}")
            page.route(f"{base}/__stage", lambda r: r.fulfill(body=stage, content_type="text/html"))
            page.route(
                f"{base}/__chart",
                lambda r: r.fulfill(body=chart_html(base), content_type="text/html"),
            )

            def css(route) -> None:  # the "open in its own tab" link only makes sense in a Space
                response = route.fetch()
                body = response.text() + "\n.own-tab { display: none !important; }\n"
                route.fulfill(response=response, body=body)

            page.route(f"{base}/static/clef-tune.css*", css)
            page.goto(f"{base}/__stage")
            page.evaluate("document.fonts.ready")
            ui = page.frame_locator("#ui")
            frame = lambda: page.frame(url=lambda u: u.startswith(base) and "__" not in u)  # noqa: E731

            def goto(path: str) -> None:
                page.evaluate(f"document.getElementById('ui').src = '{base}{path}'")
                page.wait_for_timeout(250)
                frame().wait_for_load_state("networkidle")

            def caption(key: str, sped: bool = False) -> None:
                n, label, line = CAPTIONS[key]
                page.evaluate("([n, l, t, s]) => caption(n, l, t, s)", [n, label, line, sped])

            def scroll(top: int) -> None:
                page.evaluate(
                    "top => document.getElementById('ui').contentWindow"
                    ".scrollTo({top, behavior: 'smooth'})",
                    top,
                )

            def scroll_to_end_of(selector: str) -> None:
                """Scroll until the element's bottom clears the sticky Continue bar."""
                page.evaluate(
                    """selector => {
                      const win = document.getElementById('ui').contentWindow;
                      const box = win.document.querySelector(selector).getBoundingClientRect();
                      const top = win.scrollY + box.bottom - win.innerHeight + 96;
                      win.scrollTo({top, behavior: 'smooth'});
                    }""",
                    selector,
                )

            def glide_to_end_of(selector: str, seconds: float) -> None:
                """A slow, even scroll (smooth scrolling is too quick to read along)."""
                page.evaluate(
                    """([selector, ms]) => {
                      const win = document.getElementById('ui').contentWindow;
                      const all = win.document.querySelectorAll(selector);
                      const box = all[all.length - 1].getBoundingClientRect();
                      const from = win.scrollY;
                      const to = from + box.bottom - win.innerHeight + 96;
                      const start = win.performance.now();  // the frame's own clock
                      const ease = (t) => t < .5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
                      const step = (now) => {
                        const t = Math.min(1, (now - start) / ms);
                        win.scrollTo(0, from + (to - from) * ease(t));
                        if (t < 1) win.requestAnimationFrame(step);
                      };
                      win.requestAnimationFrame(step);
                    }""",
                    [selector, seconds * 1000],
                )

            def post(path: str, **fields) -> None:
                page.request.post(f"{base}{path}", form=fields, headers={"Origin": base})

            goto("/data")  # behind the title card
            page.evaluate("card('title')")
            cast = Screencast(page, frames_dir)
            cast.start()
            t0 = time.monotonic()

            def at(second: float) -> None:  # wait until this point of the video
                left = second - (time.monotonic() - t0)
                if left < -0.2:
                    print(f"  behind at {second:.1f} s by {-left:.1f} s", flush=True)
                if left > 0:
                    page.wait_for_timeout(left * 1000)

            # 1 · the question, then straight to the answer: the benchmark chart builds up
            at(3.0)
            page.evaluate(f"document.getElementById('chart').src = '{base}/__chart#play'")
            page.wait_for_timeout(400)
            page.evaluate("card('chartcard')")
            # 2 · how it works: the spreadsheet
            at(14.6)
            caption("data")
            page.evaluate("card(null)")
            at(15.6)
            ui.locator(".drop input[type=file]").set_input_files(str(upload_file(frames_dir)))
            frame().wait_for_load_state("networkidle")
            at(19.2)
            scroll_to_end_of(".pager")  # the whole first example: complaint and its 3 answers
            # 3 · the decisions
            at(22.2)
            for qid, ask in ASKS.items():
                if qid != "outcome":
                    post("/questions/ask", qid=qid, ask=ask)
            caption("questions")
            goto("/questions?q=outcome")
            at(23.8)
            box = ui.locator("#ask-box")
            box.click()
            box.press("Control+A")
            box.press("Backspace")
            box.type(ASKS["outcome"], delay=55)
            at(28.8)
            # 4 · configure, and the price before starting
            caption("train")
            post("/settings/flavor", flavor=run.estimate["flavor"])  # the GPU the run used
            post("/settings/holdout", percent="9.0909")  # about 300 of 3,300, as the run did
            # The review lists the settings and the run's real estimate. (The Settings page's own
            # estimate assumes a 90/10 split, so it would show slightly different numbers.)
            goto("/train")
            ui.locator("#review .srows").wait_for()
            at(31.4)
            scroll(170)
            at(34.4)
            # 5 · training, sped up
            caption("run", sped=True)
            ui.get_by_role("button", name="Start training").click()
            frame().wait_for_load_state("networkidle")
            start = time.monotonic()
            while (done := (time.monotonic() - start) / 6.0) < 1:
                run.advance(done)
                page.wait_for_timeout(100)
            run.advance(1.0)
            # 6 · results: the page's own poll swaps them in when the run finishes; no reload
            ui.locator(".dots").wait_for()
            caption("results")
            at(42.6)
            glide_to_end_of(".qres", 6.5)  # the before/after dots, then each question
            # 7 · close
            at(51.4)
            page.evaluate("card('close')")
            at(LENGTH)
            cast.stop()
            browser.close()
        out.parent.mkdir(parents=True, exist_ok=True)
        encode(cast.times(t0), LENGTH, out)
        print(f"{len(cast.frames)} frames -> {out}")
    finally:
        server.should_exit = True
        if not keep_frames:
            shutil.rmtree(frames_dir, ignore_errors=True)
        else:
            print("frames kept in", frames_dir)


def gif_command(out: Path) -> None:
    """The chart building up on a loop, for the GitHub README."""
    from playwright.sync_api import sync_playwright

    from benchmarks.cfpb.chart import TITLE, page

    metrics = json.loads((RESULTS / "metrics.json").read_text(encoding="utf-8"))
    html = page(metrics, TITLE)
    folder = Path(tempfile.mkdtemp(prefix="clef-gif-"))
    try:
        source = folder / "chart.html"
        source.write_text(html, encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            tab = browser.new_page(viewport={"width": 1080, "height": 1350})
            tab.goto(source.as_uri())
            tab.evaluate("document.fonts.ready")
            tab.wait_for_timeout(300)
            cast = Screencast(tab, folder)
            cast.start()
            tab.evaluate("document.documentElement.classList.add('play')")  # starts the build-up
            t0 = time.monotonic()
            tab.wait_for_timeout(9500)
            cast.stop()
            browser.close()
        mp4 = folder / "chart.mp4"
        encode(cast.times(t0), 9.5, mp4)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(mp4),
                "-vf",
                "fps=20,scale=810:-1:flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=128:stats_mode=full[p];"
                "[b][p]paletteuse=dither=none",
                "-loop",
                "0",
                str(out),
            ],
            check=True,
        )
        print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fetch-run")
    v = sub.add_parser("video")
    v.add_argument("--out", type=Path, default=RESULTS / "video.mp4")
    v.add_argument("--keep-frames", action="store_true")
    g = sub.add_parser("gif")
    g.add_argument("--out", type=Path, default=RESULTS / "benchmark.gif")
    args = parser.parse_args()
    if args.command == "fetch-run":
        fetch_run_command()
    elif args.command == "video":
        video_command(args.out, args.keep_frames)
    else:
        gif_command(args.out)


if __name__ == "__main__":
    main()

import re

from clef_tune.data import Issue
from clef_tune.estimate import EstimateRow
from clef_tune.prompt import Piece
from clef_tune.ui import charts, fragments

EVIL = '<img src=x onerror="alert(1)">'


def row(flavor, fits=True, price=1.0, hours=1.0):
    note = None if fits else "needs ~30 GB"
    return EstimateRow(flavor, "L4", 1, price, 20.0, 24.0, fits, fits, note, hours)


def test_every_fragment_escapes_user_text():
    final = {
        "questions": {
            EVIL: {"question_type": "choice", "accuracy": 1.0, "f1": 1.0, "log_loss": 0.1}
        },
        "exact_match": 1.0,
    }
    calibration = {EVIL: {"temperature": 1.2, "ece_before": 0.1, "ece_after": 0.05}}
    outputs = [
        fragments.stat_tiles([(EVIL, EVIL)]),
        fragments.issue_list([Issue("error", EVIL, [2], EVIL)]),
        fragments.prompt_preview([Piece("state", EVIL)]),
        fragments.callout("error", EVIL),
        fragments.stage_badge(EVIL),
        fragments.metrics_table({"questions": {}}, final),
        fragments.calibration_table(calibration),
        charts.bar_chart({EVIL: 3}),
        charts.heatmap([EVIL, "b"], [[1, 0], [0, 1]]),
        charts.line_chart({EVIL: [(1, 0.5), (2, 0.4)]}),
    ]
    for html in outputs:
        assert "<img" not in html and "&lt;img" in html


def test_issue_list_puts_errors_first_and_shortens_rows():
    html = fragments.issue_list(
        [
            Issue("warning", "rare classes", [], "dept"),
            Issue("error", "bad label", list(range(2, 40)), "dept"),
        ]
    )
    assert html.index("bad label") < html.index("rare classes")
    assert "+18 more" in html
    assert "No problems found" in fragments.issue_list([])


def test_prompt_preview_marks_each_kind():
    pieces = [
        Piece("prefix", "SYS"),
        Piece("state", "hello"),
        Piece("question", "Q?", "d"),
        Piece("option", "{}", "d"),
    ]
    html = fragments.prompt_preview(pieces)
    for kind in ("prefix", "state", "question", "option"):
        assert f'class="pp-{kind}"' in html


def test_estimate_table_marks_the_recommended_and_misfits():
    html = fragments.estimate_table([row("l4x1"), row("t4-small", fits=False)], "l4x1")
    assert "ct-recommended" in html and "needs ~30 GB" in html
    assert html.index("l4x1") < html.index("t4-small")


def test_progress_is_clamped_and_can_be_unknown():
    assert "width:100%" in fragments.progress(1.7, "done")
    assert "ct-progress-unknown" in fragments.progress(None, "Starting")


def test_bar_chart_draws_one_bar_per_option_and_unlabeled():
    svg = charts.bar_chart({"a": 3, "b": 0, "c": 7}, missing=2)
    assert len(re.findall(r'class="ct-bar[ "]', svg)) == 4 and "unlabeled" in svg


def test_histogram_marks_the_cutoff_and_handles_no_data():
    svg = charts.histogram([100, 900, 4096, 5000], cutoff=4096)
    assert 'class="ct-cutoff"' in svg and "4096" in svg
    assert "ct-empty" in charts.histogram([])


def test_heatmap_shades_by_row_share_and_merges_beyond_ten():
    svg = charts.heatmap(["a", "b"], [[3, 1], [0, 4]])
    assert svg.count('class="ct-cell"') == 4 and 'fill-opacity="1' in svg
    labels = [f"c{i}" for i in range(12)]
    big = charts.heatmap(labels, [[1 if i == j else 0 for j in range(12)] for i in range(12)])
    assert "other" in big and big.count('class="ct-cell"') == 11 * 11


def test_line_chart_stays_inside_its_view_box():
    svg = charts.line_chart(
        {"loss": [(10, 1.2), (20, 0.6), (30, 0.3)]}, reference={"baseline": 1.0}
    )
    width, height = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    points = re.search(r'<polyline[^>]*points="([^"]+)"', svg).group(1).split()
    for point in points:
        x, y = map(float, point.split(","))
        assert 0 <= x <= width and 0 <= y <= height
    assert 'class="ct-reference' in svg


def test_a_histogram_far_below_the_cutoff_is_scaled_to_the_data():
    svg = charts.histogram([340, 352, 360, 371, 374], cutoff=4096)
    assert 'class="ct-cutoff"' not in svg and "every prompt fits max_length 4096" in svg
    assert len(re.findall(r'class="ct-hbar[ "]', svg)) >= 4  # spread over several bins


def test_heatmap_text_stays_readable_on_its_cell():
    svg = charts.heatmap(["a", "b"], [[7, 3], [1, 9]])  # shares 0.7, 0.3, 0.1, 0.9
    opacities = [float(o) for o in re.findall(r'class="ct-cell"[^>]*fill-opacity="([\d.]+)"', svg)]
    strong = [o for o in opacities if o >= 0.6]
    assert all(o >= 0.92 for o in strong) and all(o <= 0.6 for o in opacities if o < 0.92)
    assert svg.count("is-strong") == 2
    assert re.search(r'<svg[^>]*width="\d+"', svg)  # drawn at its natural size


def test_the_estimate_table_shows_the_five_cheapest_and_folds_the_rest():
    rows = [row(f"flavor-{i}", price=1.0 + i) for i in range(8)] + [row("small", fits=False)]
    html = fragments.estimate_table(rows, "flavor-0")
    main, folded = html.split("<details", 1)
    assert main.count('<th scope="row">') == 5 and "flavor-7" in folded and "small" in folded

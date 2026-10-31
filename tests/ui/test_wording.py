import pytest

from clef_tune.ui import wording as w


@pytest.mark.parametrize(
    ("filename", "noun"),
    [
        ("support_tickets.csv", "ticket"),
        ("claims_2026.csv", "claim"),
        ("Policies-final-export.xlsx", "policy"),
        ("server logs (labelled).tsv", "log"),
        ("addresses.jsonl", "address"),
        ("data.csv", None),
        ("2026_10_export.csv", None),
    ],
)
def test_the_row_noun_is_guessed_from_the_file_name(filename, noun):
    assert w.guess_noun(filename) == noun


@pytest.mark.parametrize(
    ("noun", "plural"),
    [
        ("ticket", "tickets"),
        ("policy", "policies"),
        ("day", "days"),
        ("address", "addresses"),
        ("box", "boxes"),
        ("batch", "batches"),
        ("log line", "log lines"),
    ],
)
def test_plurals(noun, plural):
    assert w.plural(noun) == plural


def test_counts_use_the_noun_and_the_default():
    assert w.noun_or_default(None) == "example" and w.noun_or_default(" claim ") == "claim"
    assert w.counted(1, "ticket") == "1 ticket"
    assert w.counted(1240, "ticket") == "1,240 tickets"


def test_money_and_time():
    assert w.money(0.336) == "$0.34" and w.money(None) == "—"
    assert w.money_range(0.23, 0.5) == "$0.23–$0.50"
    assert w.minutes_range(8.4 / 60, 17.2 / 60) == "8–17 minutes"
    assert w.minutes_range(8.4 / 60, 17.2 / 60, short=True) == "8–17 min"
    assert w.minutes_range(0.2 / 60, 0.4 / 60) == "1 minute"
    assert w.duration(0.42) == "25 minutes"
    assert w.duration(1.0) == "1 hour"
    assert w.duration(1 + 10 / 60) == "1 hour 10 minutes"
    assert w.time_left(12 * 60) == "About 12 minutes left"
    assert w.time_left(50) == "About a minute left"
    assert w.time_left(70 * 60) == "About 1 hour 10 minutes left"
    assert w.timeout_text("30m") == "30 minutes" and w.timeout_text("1h") == "1 hour"
    assert w.timeout_text("1.5h") == "1 hour 30 minutes"


def test_answer_names_use_descriptions_only_for_numbered_scales():
    assert w.answer_names("score", ["0", "1", "2"], ["Can wait", "This week", ""]) == [
        "Can wait",
        "This week",
        "2",
    ]
    assert w.answer_names("score", ["low", "high"], ["Low risk", "High risk"]) == ["low", "high"]
    assert w.answer_names("choice", ["billing", "account"], ["Money", ""]) == [
        "billing",
        "account",
    ]
    assert w.answer_names("noul", ["true", "false"], ["", ""]) == ["yes", "no"]


def bold(phrase):
    return [text for text, strong in phrase if strong]


def test_choice_mistakes_name_the_biggest_mix_ups():
    names = ["account", "billing", "technical"]
    two = w.mistake_summary("choice", names, [[9, 1, 0], [0, 9, 0], [1, 0, 10]], "ticket")
    assert w.plain(two) == (
        "Its 2 mistakes: one account ticket marked billing and one technical ticket marked account."
    )
    assert bold(two) == ["account", "billing", "technical", "account"]
    many = w.mistake_summary("choice", names, [[11, 1, 0], [2, 7, 1], [0, 0, 8]], "example")
    assert w.plain(many) == (
        "Its 4 mistakes: 2 billing examples marked account, one account example marked "
        "billing, and 1 more."
    )
    one = w.mistake_summary("choice", names, [[9, 1, 0], [0, 9, 0], [0, 0, 11]], "claim")
    assert w.plain(one) == "Its one mistake: one account claim marked billing."


def test_scale_mistakes_say_which_way_they_lean():
    names = ["Can wait", "This week", "Today"]
    up = w.mistake_summary("score", names, [[12, 1, 1], [0, 10, 1], [0, 0, 5]], "ticket")
    assert w.plain(up) == (
        "When it's wrong, it leans toward the top of the scale (Today): "
        "all 3 mistakes picked a higher answer."
    )
    assert bold(up) == ["Today"]
    step = w.mistake_summary("score", names, [[10, 2, 0], [0, 9, 1], [0, 0, 8]], "claim")
    assert w.plain(step).endswith(
        "all 3 mistakes picked a higher answer. Every mistake was one step off."
    )
    down = w.mistake_summary("score", names, [[10, 0, 0], [3, 9, 0], [0, 1, 7]], "ticket")
    assert w.plain(down).startswith(
        "When it's wrong, it leans toward the bottom of the scale (Can wait)"
    )
    mixed = w.mistake_summary("score", names, [[10, 2, 0], [1, 9, 0], [0, 0, 8]], "ticket")
    assert w.plain(mixed) == (
        "Its mistakes go both ways: 2 too high and 1 too low. Every mistake was one step off."
    )


def test_yes_no_mistakes_are_spelled_out():
    caught = w.mistake_summary("noul", ["true", "false"], [[5, 0], [3, 22]], "ticket")
    assert w.plain(caught) == (
        "It caught all 5 tickets where the answer was yes. "
        "3 mistakes said yes when the answer was no."
    )
    missed = w.mistake_summary("noul", ["true", "false"], [[16, 2], [1, 11]], "claim")
    assert w.plain(missed) == (
        "It caught 16 of the 18 claims where the answer was yes. "
        "One mistake said yes when the answer was no."
    )
    clean = w.mistake_summary("noul", ["false", "true"], [[19, 0], [2, 9]], "example")
    assert w.plain(clean) == (
        "It caught 9 of the 11 examples where the answer was yes. "
        "It never said yes when the answer was no."
    )


def test_no_mistakes():
    assert w.plain(w.mistake_summary("choice", ["a", "b"], [[3, 0], [0, 4]], "row")) == (
        "No mistakes on this question."
    )


def test_the_headline_fits_the_number_of_questions():
    kwargs = {
        "exact_before": 2,
        "exact_after": 24,
        "n": 30,
        "noun": "ticket",
        "mean_before": 0.41,
        "mean_after": 0.89,
    }
    three = w.headline(questions=3, **kwargs)
    assert w.plain(three) == (
        "It now matches every expected output on 24 of 30 tickets. Before training, it managed 2."
    )
    assert bold(three) == ["24 of 30"]
    assert w.plain(w.headline(questions=1, **kwargs)).startswith(
        "It now matches the expected output on 24 of 30 tickets."
    )
    assert w.plain(w.headline(questions=8, **kwargs)) == (
        "Across its 8 questions, it's now right 89% of the time, up from 41%."
    )
    assert w.dots_caption(3, "ticket") == (
        "Each dot is one test ticket. A filled dot means all three answers matched."
    )
    assert w.dots_caption(2, "claim").endswith("both answers matched.")
    assert w.dots_caption(1, "log line").endswith("the answer matched.")


def test_one_scale_mistake_is_singular():
    names = ["Can wait", "This week", "Today"]
    one = w.mistake_summary("score", names, [[12, 1, 0], [0, 10, 0], [0, 0, 5]], "ticket")
    assert w.plain(one) == (
        "When it's wrong, it leans toward the top of the scale (Today): "
        "the 1 mistake picked a higher answer. Every mistake was one step off."
    )


def test_the_averaged_headline_says_which_way_it_moved():
    kwargs = {"exact_before": 2, "exact_after": 1, "n": 30, "noun": "ticket"}
    fell = w.headline(questions=5, mean_before=0.62, mean_after=0.55, **kwargs)
    assert w.plain(fell) == "Across its 5 questions, it's now right 55% of the time, down from 62%."
    same = w.headline(questions=5, mean_before=0.6, mean_after=0.6, **kwargs)
    assert w.plain(same) == (
        "Across its 5 questions, it's now right 60% of the time, the same as before training."
    )


def test_how_long_a_run_should_take_comes_from_its_own_numbers():
    assert w.usual_time(0.5, "1h") == "Expected to take about 30 minutes"
    assert w.usual_time(None, "1.5h") == "It stops itself after 1 hour 30 minutes at the latest"
    assert w.usual_time(None, None) is None

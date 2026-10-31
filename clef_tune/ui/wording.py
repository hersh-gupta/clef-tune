"""The web UI's generated wording: the row noun, plurals, money and times, and the sentences
that summarize results. Templates only, filled with the person's own words; nothing is
written by a model. A Phrase is a list of (text, bold) pieces, so templates can bold the
person's words without building HTML here."""

from __future__ import annotations

import re
from typing import Literal

Phrase = list[tuple[str, bool]]
QuestionKind = Literal["choice", "score", "noul"]

DEFAULT_NOUN = "example"
FILLER_WORDS = frozenset(
    {
        "data",
        "dataset",
        "export",
        "final",
        "train",
        "training",
        "labeled",
        "labelled",
        "sample",
        "copy",
        "new",
        "all",
    }
)
YES_IDS = ("true", "yes")
_NUMBER_WORDS = {2: "both", 3: "all three"}


# ---------------------------------------------------------------------------- the row noun


def singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if re.search(r"(sses|xes|zes|ches|shes)$", word):
        return word[:-2]
    if re.search(r"(ss|us|is)$", word):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word


def plural(noun: str) -> str:
    """The plural of a noun; for several words, of the last one ("log line" -> "log lines")."""
    if re.search(r"[^aeiou]y$", noun):
        return noun[:-1] + "ies"
    if re.search(r"(s|x|z|ch|sh)$", noun):
        return noun + "es"
    return noun + "s"


def guess_noun(filename: str) -> str | None:
    """What one row is, from a file name: its last word that is not a filler, made singular."""
    stem = re.sub(r"\.[^.]+$", "", filename).lower()
    words = [word for word in re.split(r"[^a-z]+", stem) if word and word not in FILLER_WORDS]
    return singular(words[-1]) if words else None


def noun_or_default(noun: str | None) -> str:
    return (noun or "").strip() or DEFAULT_NOUN


def counted(n: int, noun: str) -> str:
    return f"{n:,} {noun if n == 1 else plural(noun)}"


# ---------------------------------------------------------------------------- money and time


def money(value: float | None) -> str:
    return "—" if value is None else f"${value:,.2f}"


def money_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    return f"{money(low)}–{money(high)}"


def _minutes(hours: float) -> int:
    return max(1, round(hours * 60))


def minutes_range(low_hours: float, high_hours: float, *, short: bool = False) -> str:
    low, high = _minutes(low_hours), _minutes(high_hours)
    unit = "min" if short else "minutes"
    if low == high:
        return f"{low} min" if short else f"{low} minute{'s' if low != 1 else ''}"
    return f"{low}–{high} {unit}"


def _hours_minutes(total_minutes: int) -> str:
    hours, minutes = divmod(total_minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes or not hours:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return " ".join(parts)


def duration(hours: float) -> str:
    return _hours_minutes(_minutes(hours))


def time_left(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes == 1:
        return "About a minute left"
    return f"About {_hours_minutes(minutes)} left"


_TIMEOUT = re.compile(r"^(\d+(?:\.\d+)?)([smhd])$")
_MINUTES_PER_UNIT = {"s": 1 / 60, "m": 1, "h": 60, "d": 1440}


def timeout_text(timeout: str) -> str:
    """An HF Jobs timeout such as 30m or 1.5h, in words."""
    match = _TIMEOUT.match(timeout)
    if match is None:
        return timeout
    return _hours_minutes(round(float(match.group(1)) * _MINUTES_PER_UNIT[match.group(2)]))


# ---------------------------------------------------------------------------- results


def plain(phrase: Phrase) -> str:
    return "".join(text for text, _ in phrase)


def answer_names(kind: QuestionKind, option_ids: list[str], descriptions: list[str]) -> list[str]:
    """How results name each answer: yes and no for a yes/no question, and for a scale whose
    ids are numbers, each answer's description when it has one."""
    if kind == "noul":
        return ["yes", "no"]
    if kind == "score" and all(re.fullmatch(r"-?\d+", oid) for oid in option_ids):
        return [desc.strip() or oid for oid, desc in zip(option_ids, descriptions, strict=True)]
    return list(option_ids)


def _num(n: int) -> str:
    return "one" if n == 1 else str(n)


def _mistakes(total: int) -> str:
    return "one mistake" if total == 1 else f"{total} mistakes"


def _choice(names: list[str], errors: list[tuple[int, int, int]], noun: str) -> Phrase:
    total = sum(count for _, _, count in errors)
    ranked = sorted(errors, key=lambda e: (-e[2], e[0], e[1]))
    parts: list[Phrase] = [
        [
            (f"{_num(count)} ", False),
            (names[i], True),
            (f" {noun if count == 1 else plural(noun)} marked ", False),
            (names[j], True),
        ]
        for i, j, count in ranked[:2]
    ]
    rest = sum(count for _, _, count in ranked[2:])
    if rest:
        parts.append([(f"{rest} more", False)])
    phrase: Phrase = [(f"Its {_mistakes(total)}: ", False)]
    for k, part in enumerate(parts):
        if k:
            last = k == len(parts) - 1
            phrase.append(
                (", and " if last and len(parts) > 2 else " and " if last else ", ", False)
            )
        phrase.extend(part)
    phrase.append((".", False))
    return phrase


def _score(names: list[str], errors: list[tuple[int, int, int]]) -> Phrase:
    total = sum(count for _, _, count in errors)
    up = sum(count for i, j, count in errors if j > i)
    down = total - up
    one_step = all(abs(i - j) == 1 for i, j, _ in errors)
    tail = [(" Every mistake was one step off.", False)] if one_step else []
    if not down or not up:
        end, name, way = ("top", names[-1], "higher") if not down else ("bottom", names[0], "lower")
        return [
            (f"When it's wrong, it leans toward the {end} of the scale (", False),
            (name, True),
            (f"): all {total} mistakes picked a {way} answer.", False),
            *tail,
        ]
    return [(f"Its mistakes go both ways: {up} too high and {down} too low.", False), *tail]


def _yes_no(labels: list[str], matrix: list[list[int]], noun: str) -> Phrase:
    yes = next((k for k, label in enumerate(labels) if label.lower() in YES_IDS), 0)
    no = 1 - yes
    caught, missed, false_yes = matrix[yes][yes], matrix[yes][no], matrix[no][yes]
    total_yes = caught + missed
    phrase: Phrase = []
    if total_yes:
        which = plural(noun) if total_yes != 1 else noun
        lead = (
            f"It caught all {total_yes} {which}"
            if not missed
            else (f"It caught {caught} of the {total_yes} {which}")
        )
        phrase += [(f"{lead} where the answer was ", False), ("yes", True), (". ", False)]
    if false_yes:
        start = "One mistake" if false_yes == 1 else f"{false_yes} mistakes"
        phrase.append((f"{start} said yes when the answer was no.", False))
    else:
        phrase.append(("It never said yes when the answer was no.", False))
    return phrase


def mistake_summary(
    kind: QuestionKind, labels: list[str], matrix: list[list[int]], noun: str
) -> Phrase:
    """One sentence about a question's mistakes, from its confusion matrix (rows are the
    expected output, columns the model's answer). `labels` are the answer names, or for a
    yes/no question the option ids, in matrix order."""
    errors = [
        (i, j, count)
        for i, row in enumerate(matrix)
        for j, count in enumerate(row)
        if i != j and count
    ]
    if not errors:
        return [("No mistakes on this question.", False)]
    if kind == "choice":
        return _choice(labels, errors, noun)
    if kind == "score":
        return _score(labels, errors)
    return _yes_no(labels, matrix, noun)


def headline(
    *,
    questions: int,
    exact_before: int,
    exact_after: int,
    n: int,
    noun: str,
    mean_before: float,
    mean_after: float,
) -> Phrase:
    """The results headline: rows where every answer matched for up to three questions, and
    the average share right for more, where matching all of them is too harsh a bar."""
    if questions >= 4:
        return [
            (f"Across its {questions} questions, it's now right ", False),
            (f"{round(mean_after * 100)}%", True),
            (f" of the time, up from {round(mean_before * 100)}%.", False),
        ]
    what = "the expected output" if questions == 1 else "every expected output"
    return [
        (f"It now matches {what} on ", False),
        (f"{exact_after} of {n}", True),
        (
            f" {noun if n == 1 else plural(noun)}. Before training, it managed {exact_before}.",
            False,
        ),
    ]


def dots_caption(questions: int, noun: str) -> str:
    if questions == 1:
        filled = "the answer matched"
    else:
        filled = f"{_NUMBER_WORDS.get(questions, f'all {questions}')} answers matched"
    return f"Each dot is one test {noun}. A filled dot means {filled}."

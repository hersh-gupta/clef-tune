"""A shortcut check for the trace-judge pilot: how far do word counts alone get?

For each question, fit a bag-of-words model (TF-IDF over one- and two-word terms, logistic
regression with balanced classes) on the training traces that label it, then score every test
trace. If this gets close to Clef, the test sets can be told apart by vocabulary and Clef's
score says less about judging behaviour.

Writes data/scores/bow-<set>.jsonl, which score.py reports as the "bow" arm.

Needs scikit-learn, which the project doesn't install:
  uv run --no-project --with scikit-learn python -I benchmarks/trace_judge/baselines.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
PREPARED = HERE / "data" / "prepared"  # git-ignored
SCORES = HERE / "data" / "scores"
SETS = ("malt_test", "atbench500", "atbench", "atbench_no_tools", "controlarena")
QUESTIONS = ("unsafe", "hijacked", "hidden_goal", "sandbagged")


def read(name: str) -> list[dict]:
    csv.field_size_limit(2**31 - 1)
    with (PREPARED / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    train = read("train")
    models = {}
    for question in QUESTIONS:
        rows = [row for row in train if row[question] in ("true", "false")]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True
        )
        features = vectorizer.fit_transform(row["trace"] for row in rows)
        model = LogisticRegression(class_weight="balanced", max_iter=2000)
        model.fit(features, [row[question] == "true" for row in rows])
        models[question] = (vectorizer, model)
        print(f"{question}: fitted on {len(rows):,} training traces", flush=True)
    del train
    SCORES.mkdir(parents=True, exist_ok=True)
    for name in SETS:
        rows = read(name)
        scores: list[dict] = [{} for _ in rows]
        for question, (vectorizer, model) in models.items():
            if not any(row[question] for row in rows):
                continue
            margins = model.decision_function(vectorizer.transform(row["trace"] for row in rows))
            for item, margin in zip(scores, margins, strict=True):
                item[question] = round(float(margin), 5)
        with (SCORES / f"bow-{name}.jsonl").open("w", encoding="utf-8", newline="\n") as out:
            for row, item in zip(rows, scores, strict=True):
                out.write(json.dumps({"id": row["id"], "scores": item}) + "\n")
        print(f"{name}: scored {len(rows):,} traces", flush=True)


if __name__ == "__main__":
    main()

# Bank-complaint benchmark: fine-tuned Clef vs a frontier LLM — design

**Date:** 2026-11-03
**Status:** Draft for review
**Branch:** `benchmark-cfpb` (worktree `.claude/worktrees/cfpb-benchmark`), from `main` af1a9a4.
**Mockups:** `.superpowers/brainstorm/1025-1793756045/content/` in the main checkout (git-ignored):
`chart-options-v2.html` (the user chose **B**) and `video-storyboard-v2.html` (the user's
captions). This document wins where wording differs.

Choices the user did not make are marked **(my call)**. The user can overturn any of them.

---

## 1. Intent

clef-tune went public on LinkedIn on 2026-10-31. Its only real result so far is 14 → 18 of 31
correct on a 169-row sample, which is too small to claim anything. The user wants an honest,
shareable benchmark for a follow-up post: does a Clef model fine-tuned with clef-tune beat a
frontier LLM that is simply prompted, on a real business decision, and at what cost?

The audience is business users and practitioners on LinkedIn, mostly **on phones**. So the
wording stays plain, and every visual is designed for a phone screen first.

The test has to survive scrutiny. That means real data with real outcomes, a test set no model has
seen (including the frontier LLM in its training), the same instructions for every contender,
and reporting whatever comes out.

## 2. Goals, non-goals, anti-goals

**Goals**
- A reproducible pipeline under `benchmarks/` that goes from public data to the results table,
  chart, video and GIF, with each step re-runnable on its own.
- Six contenders scored on the same held-out test complaints, with accuracy, macro-F1, calibration,
  cost per 1,000 complaints and speed.
- A results table, a chart PNG for LinkedIn, `benchmarks/README.md` (method and limits), two
  tuned models and one dataset on the Hub, a 4:5 MP4 and a README GIF.

**Non-goals**
- Prompt-engineering Claude to its maximum, or hyperparameter-searching Clef. Each contender gets
  one sensible, fixed setup.
- Several runs per contender or several banks. One run each, stated as a limit.
- Changing clef-tune's product code. If the pipeline needs something the CLI can't do, it lives in
  `benchmarks/` and is listed as a possible follow-up.

**Anti-goals** (would make this a failure even if it runs)
- Any test row used for training, few-shot examples, prompt fixes or model selection.
- Cherry-picking: re-running a contender because its number disappointed, dropping a metric that
  looks bad, or a chart whose scale exaggerates a gap.
- Wording that oversells: "beats GPT-class models", "state of the art", or numbers without their
  95% ranges in the write-up.
- An API key, token or private path written into any committed file.
- Wording that reads like AI-generated copy (the user flagged double-barrelled short sentences).

## 3. Data: CFPB bank complaints

**Source** (user's choice, from four researched candidates).
- **Text:** the CFPB's narrative archive, monthly ZIPs in its FOIA reading room:
  `https://files.consumerfinance.gov/f/documents/CCDB_Export_<n>_<period>.zip`, 10–64 MB each,
  covering complaints from December 2011 to 14 August 2026. The CFPB stopped publishing narratives
  on 14 August 2026.
- **Labels:** the CFPB search API,
  `https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/`, joined on
  `complaint_id`. The API's `_meta.license` is `CC0`. The archive's own fields are not yet known;
  if it already carries the labels, the API join is only a check.
- **Licence:** CC0 (US government work). The data and anything trained on it can be published.
- **Downloads:** each archive file is a download, so the pipeline lists files and sizes and asks
  before fetching (§7). Files go to a git-ignored `benchmarks/cfpb/data/raw/`, one folder per
  download, and are read with `python -I`.

**Scope** (user agreed).
- One bank, **Bank of America** ("BANK OF AMERICA, NATIONAL ASSOCIATION") **(my call)**. About 25%
  of its complaints end with money back, compared with about 8% at Chase, so the outcome classes
  are less lopsided.
- Products: checking or savings accounts, credit cards, and money transfers. Credit reporting, debt
  collection, loans, mortgages and prepaid cards are excluded, so no row is a credit decision.
- Only complaints with a narrative, and only those whose company response is final. "In progress"
  and "Untimely response" are dropped.

**Naming the bank** **(my call, extending what the user agreed)**: the chart, video and LinkedIn
materials say "a large US bank". The write-up and the dataset card name Bank of America, because
the data is public and anyone rerunning the benchmark needs to know.

**The row** (the "state" Clef reads): the complaint narrative only **(my call)**. Product, issue,
dates, state and ZIP are not given to any contender. Masking such as `XXXX` stays as published.

**Questions** (all type `choice`; the wording is plain and the same for every contender):
1. `issue` — "What is this complaint mainly about?" The 7 most frequent CFPB issues for this bank
   and these products in the training months, plus "Other" **(my call on 7)**. Each choice gets a
   one-line description written from the CFPB's issue name.
2. `product` — "Which product is the complaint about?" Checking or savings account / Credit card /
   Money transfer.
3. `outcome` — "How did the bank close this complaint?" With an explanation only / With a fix but
   no money (non-monetary relief) / With money back (monetary relief).

`product` and `issue` are picked by the consumer on the CFPB form, so they are noisy. `outcome` is
the bank's recorded decision. The write-up says both plainly.

**Split by date** (user agreed).
| Split | Received | Rows | Used for |
|---|---|---|---|
| train | 2025-01-01 to 2026-05-31 | ~3,000, a seeded random sample | Clef training; the few-shot pool |
| validation | same months, disjoint from train | ~300 | Clef's model selection and calibration; checking Claude's output format |
| test | 2026-07-01 to 2026-08-14 | up to ~600, all that qualify, or a seeded sample if more | scoring only |

June 2026 is left out as a buffer around current Claude models' **June 2026 training-data cutoff**
(Anthropic's models overview, checked 2026-11-03). Test complaints were published after that
cutoff, so no contender can have trained on them. If fewer than about 400 test rows qualify, stop
and ask the user before widening anything. Real counts are reported, not the targets above.

**Length:** every contender sees the same text. Complaints longer than Clef's Balanced
`max_length` (4,096 tokens) allows are dropped from every split, and the count is reported
**(my call)**.

**Shape:** the files follow `examples/`: `train.csv`, `validation.csv` and `test.csv`, with
columns `complaint_id, complaint, issue, product, outcome`, plus one `cfpb_complaints.yaml` per model
(`id_column: complaint_id`, `state_columns: [complaint]`, `row_noun: complaint`,
`data.eval: validation.csv`). `clef-tune validate` must pass on all three files.

## 4. Contenders and fairness

| # | Contender | Setup |
|---|---|---|
| 1 | Clef-Flash, out of the box | The released model with the schema prompt, untrained |
| 2 | Clef-Flash, tuned | clef-tune **Balanced** preset (LoRA r16, 3 passes) via `clef-tune jobs launch` |
| 3 | Clef 27B, out of the box | as 1 |
| 4 | Clef 27B, tuned | as 2 |
| 5 | Claude Opus 5.5, instructions only | `claude-opus-5-5`, zero-shot |
| 6 | Claude Opus 5.5, with examples | the same, plus a fixed set of ~30 training complaints with their answers |

**"Out of the box"** means exactly what training starts from: the released backbone and head with
untouched LoRA weights. That is clef-tune's own "baseline" (`train.py`), so contenders 1 and 3 are
honest "before" points.

**Update count:** about 3,000 rows × ~550 tokens ÷ (16,384 × 4) ≈ 25 updates per pass, or about 75
on Balanced. The proposed small-sheet fix ("Give small spreadsheets enough training updates") has
not landed and isn't needed at this size. The builder confirms the real count with
`clef-tune estimate` before launching.

**Claude setup, the same for both variants:**
- The prompt holds the same question wording and choice descriptions as the YAML, rendered as
  plain text, and the complaint text. Claude is not told the bank's name; neither is Clef.
- Effort is set explicitly to the model's default, `medium` **(my call)**. Thinking can't be
  turned off on Opus 5.5.
- Output uses structured outputs (`output_config.format`): each question's answer is an enum of its
  choices, plus a 0–100 "confidence" per question.
- **No refusal fallback** **(my call)**, so every answer is Opus 5.5's own. A refusal or unparseable
  reply counts as wrong for every question and is reported by count.
- Few-shot examples are drawn by seeded random sampling from train, re-drawn only until every
  choice of every question appears at least once. They are never hand-picked. One fixed set serves
  every test row and is cached with prompt caching.
- The prompt is written once. It may get one round of fixes for format problems, checked on about
  50 validation rows, never on test rows. Any change is recorded in the write-up.
- The API key is read at run time from the Machine-scope `ANTHROPIC_API_KEY` environment variable
  (the user set it) and is never written to disk.

**Where things run:** the laptop's 12 GB GPU can't hold Clef-Flash in bf16, so training and test
scoring run on HF Jobs, using clef-tune's own image and install. Scoring goes through
`clef-tune eval`: the tuned adapters from their Hub repos, and the out-of-the-box models from an
adapter saved from the untrained baseline. Each scoring job writes `clef-tune eval --output` JSON
and per-row probabilities.

## 5. Metrics and reporting

All contenders are scored by **one shared scorer** in `benchmarks/` from per-row answers. Clef's
answers are the arg-max of calibrated probabilities; Claude's are its structured answers.

- **Headline:** "all three answers right" (exact match), the chart's measure.
- Per question: accuracy and macro-F1 (`outcome` and `issue` are imbalanced). Clef's numbers must
  match `clef-tune eval`'s own report, as a cross-check.
- **95% ranges** from 2,000 bootstrap resamples of the test rows **(my call)**. A paired bootstrap
  gives the difference between tuned Clef and Claude-with-examples.
- **Calibration:** expected calibration error. For Clef it comes from probabilities, before and
  after temperature scaling. For Claude it comes from its stated confidence, labelled "stated
  confidence, not a probability".
- **Cost per 1,000 complaints**, with prices and the date:
  - Clef: measured scoring time on the GPU × HF's hourly price for that flavor. Job start-up is shown
    separately, and training as a one-off cost.
  - Claude: billed tokens from `usage` × the list price on the day ($4 input, $20 output per million
    tokens for Opus 5.5 as of 2026-11-03, with cache reads at $0.20). The 50% batch price is shown
    as a second line, computed rather than run.
- **Speed:** complaints per minute. For Clef, measured while scoring. For Claude, measured at a fixed
  concurrency (8 parallel requests **(my call)**), with the median seconds per complaint.
- **Reported as is:** every number in the results table, including any where Claude wins. A
  contender is never re-run, except for an infrastructure failure (a job that dies, or an API
  outage), and any such re-run is noted.
- **If Clef 27B fails to train** (out of memory, a timeout): retry once on a larger flavor with the
  same setup. Any change of setup, such as 4-bit loading or a shorter `max_length`, is the user's
  call and is reported.

**Outputs** in `benchmarks/cfpb/results/`, committed: `results.md` and `results.csv` (the table),
`answers/<contender>.csv` (per-row answers, confidences and correct flags), `costs.json` (prices,
dates and measured times), and `chart.png`.

## 6. Pipeline under `benchmarks/`

Layout **(my call, the names may change)**:

```
benchmarks/
  README.md                 # the write-up: method, results, limits, how to rerun
  cfpb/
    fetch.py                # list archive files and sizes; download after a yes; API labels
    prepare.py              # filter, split by date, write CSV + YAML, a data card
    claude_arm.py           # zero-shot and few-shot runs; resumable; writes answers + usage
    clef_jobs.py            # validate, estimate, jobs launch; scoring jobs; download results
    score.py                # shared scorer: metrics, bootstrap ranges, costs, results.md/csv
    chart.py                # chart.png (chart B, 1080×1350)
    video/                  # stage page, recording script, ffmpeg encode (MP4 + GIF)
    data/                   # git-ignored: raw downloads, prepared CSVs
    results/                # committed outputs (see §5)
```

- Each step reads the previous step's files and writes its own, so a failed job or API outage
  never means redoing finished steps. The Claude runner skips rows it already has.
- Run with `uv run` and all extras (`uv sync --locked --extra dev --extra train --extra ui`). Any
  command given to the user is in cmd syntax (the user runs cmder).
- New dependencies are avoided where possible: `anthropic` is the one likely addition, as an
  optional `benchmark` extra **(my call)**.
- Tests: light unit tests for the scorer (metrics, bootstrap, cost arithmetic) and the split
  (no overlap, date bounds, no test row in the few-shot set). No tests for the one-off fetch or
  video scripts.

## 7. Spending and approvals

Nothing is spent without the user's yes on the exact estimate.

| Gate | What the user sees first |
|---|---|
| Download archives | file names, source, sizes (~150–250 MB for the needed months) |
| Clef-Flash training | `clef-tune estimate` output: flavor, time, $ range |
| Clef 27B training | the same (H200 or A100-80GB class; it hasn't been trained on HF Jobs before) |
| Scoring jobs (4) | flavor and $ estimate per job |
| Claude runs | token count from `count_tokens` on real prompts × price, for both variants |
| Making any repo public | an explicit yes per repo |
| Push to GitHub, Space upload, posting | not part of this work; ask first |

Rough totals from the brainstorm: Hugging Face about $9–12, Anthropic about $10–15. The user has
topped up both accounts.

## 8. Publishing on the Hub (all private until the user says otherwise)

- **Dataset** `hvgupta1/cfpb-bank-complaints-benchmark` **(my call on names)**: the three CSVs,
  the YAML and a data card. The card covers source and licence (CC0), filters, date windows,
  counts, label caveats and how to rebuild it with `benchmarks/cfpb/`.
- **Models** `hvgupta1/clef-flash-cfpb-complaints` and `hvgupta1/clef-cfpb-complaints`, set via
  `output.push_to_hub`. Each keeps clef-tune's generated card, with a section added that links the
  dataset, gives the test results with their ranges, and points to `benchmarks/README.md`.

## 9. Chart (LinkedIn PNG)

- **Form: B, before → after dumbbells** (the user's choice), echoing the Space's results page.
- **Rows:** Clef-Flash, Clef 27B and Claude Opus 5.5. Each row runs from a hollow oak dot ("out of
  the box", or "instructions only" for Claude) to a filled sage dot ("tuned", or "with examples"),
  with the cost per 1,000 complaints under the name.
- **Measure:** "all three answers right" on a 0–100% axis. The axis is never truncated.
- **Ranges:** a faint 95% range under each dot, if it stays legible at phone size; otherwise in the
  footnote **(my call)**.
- **Canvas:** 1080×1350, light theme: cream page, Noto Sans JP, hairlines, a thin-weight title with
  a short oak rule, and the kicker with the diamond mark.
- **Colour and shape:** oak and sage carry before/after, and hollow vs filled carries it a second
  way. Under colour-blind simulation oak and sage separate at ΔE 8.2 (dataviz validator).
- **Footnote:** the test window, the row count, "never used in training", training as a one-off
  cost, and the price date.
- **Title:** plain, and written after the results are known, so it says what they show.
- The same drawing code renders the chart's final frame in the video, so they match.

## 10. Video (4:5 MP4) and README GIF

**Format:** MP4, 1080×1350 (4:5) only (user's choice), H.264, 30 fps, about 65 s, light theme. It
must stay crisp on a phone. A large caption sits in the top third of every scene, since LinkedIn
autoplays muted.

**Scenes and captions** (the user's wording, lightly edited; final wording can still change):

| # | s | Caption | Picture |
|---|---|---|---|
| 1 | 0–5 | "Can a small decision model, tuned on real data, beat a frontier LLM?" / "Tested on 600 bank complaints" | Title card; the accent rule draws in |
| 2 | 5–13 | "Load 3,000 bank complaints from the CFPB's public records" | The real UI's data step with the complaints CSV; rows rise in a stagger; oak/sage dot tags |
| 3 | 13–21 | "Lay out the decisions the model should make, in plain words" | Questions step; "How did the bank close this complaint?" typed |
| 4 | 21–27 | "Configure training, and see the cost and time before you start" | Settings → train-ready, with the real run's estimate |
| 5 | 27–35 | "It trains in your own private account, so your data stays yours" | Training page, dot breathing, sped up and labelled "sped up" |
| 6 | 35–43 | "Measured on complaints it never saw" | Results page, before/after dots fill in |
| 7 | 43–60 | "Then measured against Claude Opus 5.5 on 600 newer complaints" | Chart B builds up: rows rise, hollow dots slide to filled, costs fade in; holds ~4 s |
| 8 | 60–65 | "Try it on your own data" | Space and GitHub links; one line on limits |

Captions don't name Hugging Face (the user finds it too technical). "Your data stays yours" is
accurate: the data goes only to the person's own private storage and jobs, and to no LLM
provider. Numbers in captions (3,000, 600) are replaced by the real counts.

**How it's made** **(my call)**:
- The real UI runs with fake services (as in `scripts/screenshots.py`), replaying a **recorded run
  made from the real Clef-Flash training job**: its metrics log, estimate and before/after on the
  validation rows. It never replays the support-ticket fixture. Scene 6 shows validation numbers
  and scene 7 shows test numbers, and the captions keep the two apart.
- A stage page at 1080×1350 holds the caption band and the UI at phone width (about 540 CSS px at
  2× scale), so the Space's phone layout is what's shown.
- Playwright drives the clicks and typing. Frames are captured on a fixed clock (deterministic
  animation time, not a live screen recording) and encoded with ffmpeg 9.0, which is already
  installed.
- Motion uses the Space's own keyframes (rise, breathe, fillin, grow) and its easing.
- One script regenerates everything: `uv run python benchmarks/cfpb/video/make_video.py`.

**About the UI scenes:** the real runs are launched with the CLI (`clef-tune jobs launch`, with a
separate validation file). Scenes 2–6 show the same workflow in the Space with the same data and
the real run's numbers. They illustrate how it works and are not a screen recording of the run
**(my call)**.

**GIF:** scene 7 alone, about 8 s, looping, 720 px wide at 4:5, under about 5 MB. It goes in a
short "Benchmark" section of the top-level `README.md` that links to `benchmarks/README.md`
**(my call)**.

**Review loop:** drafts go to the user with SendUserFile, frame stills first and then the MP4,
until the user approves.

## 11. The write-up (`benchmarks/README.md`)

The write-up is in plain words. It covers the question asked, the data and why it was chosen, the
contenders and how each was set up, the results table with 95% ranges, cost and speed with dates,
and how to rerun it.

**Limits, stated up front:**
- one bank and one kind of decision;
- consumer-chosen labels for issue and product;
- one run per contender;
- about 600 test rows;
- Claude's confidence is self-reported;
- costs are list prices on one date;
- the few-shot set is one random draw.

## 12. Constraints

- Windows; git works in Git Bash, not PowerShell. Use `uv run hf` from PowerShell.
- `HF_TOKEN` and `ANTHROPIC_API_KEY` are Machine-scope environment variables. They are read from
  there at run time and never printed or written.
- HF Jobs need prepaid credits; a 402 means top up.
- Commit only on `benchmark-cfpb`. No pushes, Space uploads or posts without asking. No
  Co-Authored-By or "Generated with Claude" lines.
- Downloaded files are untrusted data: each goes in its own new folder, and Python reading them
  runs with `-I`.

## 13. Left to the builder

- Exact script names and internals, API paging, and how the archive and API are joined.
- The exact issue choices and their one-line descriptions, taken from the training data.
- The Claude prompt's exact text, within §4's rules.
- How the scoring job is submitted: reusing `clef_tune.launch` helpers, or a plain `hf jobs run`
  with clef-tune's image.
- The video's frame-capture technique and the timing within each scene.

Test set: 378 complaints received 2026-07-01 to 2026-07-29. Ranges are 95% bootstrap ranges (2000 resamples).

| Contender | All three right | Issue | Product | Outcome | Cost per 1,000 | Complaints per minute |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5, instructions only | **33.3%** (28.6–37.8) | 56.1% | 91.8% | 64.0% | $8.04 ($4.02 batch) | 166 |
| Claude Opus 5.5, with examples | **41.3%** (36.2–46.3) | 60.8% | 91.3% | 68.0% | $7.54 ($3.77 batch) | 230 |
| Clef-Flash, out of the box | **19.6%** (15.6–23.8) | 43.7% | 86.5% | 51.1% | $0.046 | 993 |
| Clef-Flash, tuned | **44.4%** (39.2–49.7) | 69.8% | 91.0% | 65.3% | $0.062 | 738 |
| Clef 27B, out of the box | **17.7%** (14.0–21.4) | 38.1% | 87.3% | 52.4% | $0.13 | 359 |
| Clef 27B, tuned | **47.9%** (42.9–52.9) | 70.1% | 92.6% | 68.3% | $0.17 | 266 |

Macro-F1 (the average over each question's answer choices, so rare choices count as much as common ones) and calibration error:

| Contender | Issue F1 | Product F1 | Outcome F1 | Calibration error (issue / product / outcome) |
| --- | --- | --- | --- | --- |
| Claude Opus 5.5, instructions only | 0.492 | 0.891 | 0.496 | 0.135 / 0.075 / 0.055 |
| Claude Opus 5.5, with examples | 0.561 | 0.883 | 0.562 | 0.089 / 0.074 / 0.086 |
| Clef-Flash, out of the box | 0.437 | 0.838 | 0.275 | 0.191 / 0.068 / 0.230 |
| Clef-Flash, tuned | 0.660 | 0.882 | 0.427 | 0.072 / 0.031 / 0.067 |
| Clef 27B, out of the box | 0.364 | 0.848 | 0.311 | 0.335 / 0.063 / 0.162 |
| Clef 27B, tuned | 0.643 | 0.905 | 0.448 | 0.060 / 0.031 / 0.034 |

Paired differences in "all three right" (same resamples for both):

| Comparison | Difference | 95% range |
| --- | --- | --- |
| Clef-Flash, tuned vs Claude Opus 5.5, with examples | +3.2 points | -2.1 to +9.0 |
| Clef 27B, tuned vs Claude Opus 5.5, with examples | +6.6 points | +1.6 to +11.9 |
| Clef-Flash, tuned vs Claude Opus 5.5, instructions only | +11.1 points | +5.3 to +16.9 |
| Clef 27B, tuned vs Claude Opus 5.5, instructions only | +14.6 points | +9.3 to +19.6 |
| Clef-Flash, tuned vs Clef-Flash, out of the box | +24.9 points | +19.0 to +30.7 |
| Clef 27B, tuned vs Clef 27B, out of the box | +30.2 points | +24.6 to +35.7 |
| Claude Opus 5.5, with examples vs Claude Opus 5.5, instructions only | +7.9 points | +4.0 to +11.9 |

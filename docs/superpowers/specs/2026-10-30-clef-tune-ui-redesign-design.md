# clef-tune web UI redesign — design

**Date:** 2026-10-30
**Status:** Draft for review
**Replaces:** the layout and visual parts of the main spec's §9 (Web UI) in
`2026-10-26-clef-finetuning-harness-design.md`. Plan 4's Task 8 (Gradio layout and theme), the
Gradio side of Task 9 (Space deployment) and Task 10 (screenshot rounds) are superseded. §9's
rules about public Space mode still hold except where this document says otherwise.
**Mockups:** `2026-10-30-clef-tune-ui-redesign-mockups/` beside this file. They are the visual
source of truth for look, spacing and motion. Where the mockups' wording differs from this
document, this document wins.

Choices the user did not make are marked **(my call)**. The user can overturn any of them.

---

## 1. Intent

The Gradio UI on branch `plan-4-ui` works, but the user found it "very clunky", especially the
Configure step and the question pills. Each widget is its own boxed block, and custom styling
fights Gradio's at every turn. The user chose to start the UI over.

The new UI is for **business users who have never fine-tuned a model**. They have a spreadsheet
of examples: tickets with the team that handled them, claims with a review outcome, log lines
with a severity. They want a model that makes the same calls on new cases. They should get from
"here's my spreadsheet" to "here's a model that does this, and here's how much better it got" in
one sitting, without reading docs, and possibly from a phone.

The user's framing: **bespoke intelligence**. The model is refined to make the kinds of
judgements it will meet in real use, and the person owns the result.

The feel is **calm and Japandi**: tan and green, thin lines instead of boxes, gentle motion.

## 2. Goals, non-goals, anti-goals

**Goals**
- **Plain words on the main path.** No LoRA, epochs, tokens, prompt templates or GPU names unless
  the person opens Advanced or "For developers".
- **The app makes the technical choices.** It picks the hardware and shows a time and a price
  rather than a table of machines.
- **Every screen works on a phone** (360 px wide) as well as on a desktop.
- **Leave and come back.** Training and prediction run on Hugging Face. Closing the tab is safe,
  and "Your models" shows where everything stands.
- **Results in business terms first**, with the technical metrics one click away.
- **Predict a new spreadsheet** from a trained model: upload new rows, get them back with answers
  filled in.
- **Accessible:** WCAG AA contrast in light and dark, fully usable by keyboard and screen reader,
  and all motion off under `prefers-reduced-motion`.

**Non-goals (this round)**
- Models other than Clef and Clef-Flash. A model picker for other Hugging Face decision models was
  discussed and shelved; see §12.
- Email notifications.
- Renaming the project; see §12.

**Anti-goals: failure even if everything technically works**
- It looks like a stock form builder: boxed widgets, gray panels, default controls.
- A business user meets jargon before they reach Advanced.
- Money is spent without the price being shown first, including a hard upper limit.
- A whole page re-animates when only one part changed. The user caught this on the phone's
  row-card pager, and it applies everywhere: **only the part that changed moves.**
- Generated copy that is wrong for someone's data, such as calling claims "tickets" or making up
  domain words.

## 3. Constraints

- **Gradio is removed** from the UI. The screens are hand-built HTML, CSS and a little
  JavaScript, served by a small Python web server (§9).
- **The Gradio-free modules carry over** with only the changes this document calls for:
  `clef_tune/ui/` `handlers.py`, `session.py`, `runs.py`, `local.py`, `services.py`,
  `settings.py`, `fragments.py` and `charts.py`, plus everything outside `ui/`.
- **The UI imports only no-torch modules,** as before.
- **Hosting stays a public, multi-user Hugging Face Space,** with HF sign-in for anything that
  spends money or writes to the person's account.
- **The public Space rules from §9.3 still hold:**
  - visitors never act as the Space owner;
  - there is no local training on a Space;
  - each session gets its own temporary folder;
  - uploads are capped (`CLEF_TUNE_MAX_UPLOAD_MB`, default 200).
- **Hugging Face Jobs needs prepaid credits.** Without them, launches fail with HTTP 402.
- **A Space app runs inside an iframe on huggingface.co.**
  - Cookies (the session and the OAuth state) must work there: `SameSite=None; Secure`.
  - When the app is framed, sign-in must open at the top level, in a new tab, as Gradio's login
    button does.
  - Check both on the live Space before calling sign-in done.

## 4. Visual language

### 4.1 Colour tokens

The app **always opens in light**. Dark is only a choice from the header toggle. It is remembered
in `localStorage` and does not follow the OS setting. (The user asked for light as the default.)

| Token | Light | Dark | Use |
|---|---|---|---|
| `--page` | `#F7F2EA` cream | `#23211D` warm charcoal | page background |
| `--field` | `#FBF8F3` | `#2A2723` | text boxes, sheets, popovers |
| `--ink` | `#33302A` | `#ECE5D8` | body text |
| `--muted` | `#665E52` | `#A79E8F` | secondary text (≥4.5:1 on page) |
| `--line` / `--line-soft` | `#DDD0BC` / `#E8DFD1` | `#3A362F` / `#312E28` | hairlines |
| `--sage` | `#5C7A64` | `#9DB89F` | primary actions, "learns / predicts", progress |
| `--sage-ink` | `#48644F` | `#A9C4AB` | sage text (darker than `--sage` for contrast) |
| `--sage-wash` | `#E9EDE3` | `#2B302A` | hover and focus fills, info notes |
| `--oak` / `--oak-ink` | `#A88B66` / `#5E4E38` | `#C9A982` / `#D7BD99` | "reads it", warnings, accent rules |
| `--oak-wash` | `#F0E6D6` | `#332D24` | mistake cells, drop-zone hover |
| `--rust` | `#A4573A` | `#D9896B` | blocking errors only |

### 4.2 Elements

These come from the chosen style, "C".
- **No boxes.** Sections are separated by 1 px hairlines. Rows sit on the page.
- **Step bar.**
  - *Desktop:* four numbered tabs (`01 Your data`, `02 What to predict`, `03 Settings`,
    `04 Train`) over a hairline. The current tab has a 2 px sage underline and a breathing sage
    dot, and finished tabs show ✓. Reached steps can be revisited freely.
  - *Phone:* four thin segments plus "Step N of 4 · Name".
- **Titles:** light weight (300), about 23 px, with a 48 px oak accent rule under them.
- **Small labels:** 10 px, uppercase, 0.12 em tracking, muted.
- **Tags:** a small dot or ring before the text, never filled pills.
  - *Model reads it:* oak ring.
  - *Expected output:* filled sage dot, sage text.
  - *Image:* oak square.
  - *Row ID:* small muted ring.
  - *Ignore:* dashed ring.
- **Editable words** (the row noun, the review threshold): an underlined word in sage, with a
  dashed underline that turns solid on hover. A ✎ appears on hover and is always visible on touch
  screens. No box until focused.
- **Buttons.** One filled sage button per screen, for that screen's main action. Everything else
  is an outline button or an underlined text link.
- **Font:** Noto Sans JP (300, 400, 500) from Google Fonts. Code uses `ui-monospace` with
  ligatures off.

### 4.3 Motion

The motion comes from style "A".
- **Entrance:** content rises 8 px and fades in over 0.7 s, `cubic-bezier(.2,.7,.2,1)`, staggered
  80–90 ms.
- **Breathing:** the current step's dot and the progress line's head pulse slowly (3.2 s).
- **State changes:**
  - colours cross-fade over 0.3–0.45 s;
  - a tag gives a small "pop" when it changes;
  - changed numbers rise into place (0.55 s).
- **Containers:** sheets slide up (0.5 s), and folds open by height (0.55 s).
- **Success:** a check-mark circle draws itself.
- **Scope:** only the changed part animates; nothing re-plays the page entrance.
- **Reduced motion:** `prefers-reduced-motion: reduce` turns off all animation and transitions.

## 5. Vocabulary

- **"Examples"**, not "decisions", for what a person uploads. Each one can come from past work or
  be labeled just for this.
- **"Expected output"** is what is filled in the spreadsheet. **"Answer"** is what the model says
  to a question. The question step lists the possible **answers**. Results say how often the
  model **matched the expected output**.
- **Row noun.** One word for what a row is, such as ticket, claim or log line, used in sentences
  throughout.
  - **Guess:** taken from the file name. Drop the extension, split on non-letters, drop filler
    words (`data`, `dataset`, `export`, `final`, `train`, `training`, `labeled`, `labelled`,
    `sample`, `copy`, `new`, `all`), take the last word left, and make it singular.
  - **Fallback:** "example" when nothing is left.
  - **Plural:** formed automatically (`y`→`ies` after a consonant, `+es` after s/x/z/ch/sh,
    otherwise `+s`).
  - **Edit:** the person can change the word.
  - **Persistence (my call):** stored with the run, as a new optional config field
    `data.row_noun`. That way results opened later from "Your models" still use it.
- **Question wording is always the person's own:** the question sentence, answer names and
  answer descriptions.
- **No AI-written copy.** Everything generated comes from templates (§7.6).

## 6. Flow

```
Welcome ──► 01 Your data ──► 02 What to predict ──► 03 Settings ──► 04 Train (ready → running)
                                                                         │
"Your models" (header link on every screen) ◄────────────────────────────┤
     │                                                                   ▼
     └── a ready model ──► Results ──► Predict a new spreadsheet (choose → running → ready)
```

- **Steps 1–3 need no sign-in** on the Space. Sign-in is asked for at Train, where money is spent.
- **The old Validate step is gone.** Its checks appear in the step where you would fix them (§7.2).
- **Leaving is always safe once something is running.** "Your models" is where people land when
  they come back.

## 7. Screens

Each screen below is described for desktop. The phone version uses the same content, with the
differences noted.

### 7.1 Welcome

**First visit:**
- **Header:**
  - the diamond mark and `CLEF-TUNE` wordmark;
  - on the right: "Sign in with Hugging Face" (Space only, signed out), the theme toggle (a small
    half-filled circle).
- **Kicker:** "Fine-tuned decisions for everyone".
- **Headline:** "Bespoke intelligence, *owned by you*." The emphasised part is in sage.
- **Lede:** "Upload a spreadsheet of examples with the expected outputs filled in. clef-tune
  fine-tunes a decision model to predict them for new cases, and shows you how much better it got
  before you use it."
- **Buttons and links:**
  - one filled button, "Start with your data";
  - below it, "Or try a sample: support tickets · claim triage · server logs" as one line of
    links. Hovering a link (its title) says what that sample predicts.
- **Sign-in note:** "No sign-in needed to look around. You'll sign in with Hugging Face when
  you're ready to train."
- **Line drawing:**
  - *your data:* a sheet with check dots;
  - *your decision model:* the diamond, with a breathing halo;
  - *answers:* three bars that fill in.

  The labels read "YOUR DATA / YOUR DECISION MODEL / ANSWERS". Clef is not named here.
- **"How it works":** the four steps, one plain line each.
- **Three reassurances:** the user liked them, so keep the wording.
  - "Pay only for what runs. Often under a dollar for a few hundred examples, billed by Hugging
    Face. You see the price before anything starts."
  - "Your data stays yours. Training data and the finished model are stored privately in your own
    Hugging Face account."
  - "See before you trust. Every model is tested on examples it never saw, before and after
    training."
- **Footer:** "Fine-tunes Cloudflare's open Clef decision models · Open source on GitHub · MIT
  licence".

**Signed in, coming back:**
- **Header:** "Your models" and the username.
- **Body:** a greeting, the two newest models (rows as in §7.7), "See all N models", and a filled
  "Train a new model".

**Local mode (not a Space):** there is no sign-in link, and the machine's Hugging Face login is
used. The returning view shows when that login exists.

### 7.2 Step 1 — Your data

**Lede:** "Upload a spreadsheet of examples with the expected outputs filled in, from past work or
labeled just for this. The closer they are to what the model will see in real use, the better it
learns."

**Before a file is chosen:**
- **Desktop:** a dashed oak drop zone reading "Drop a spreadsheet here, or *choose a file*". The
  sheet icon floats gently.
- **Phone:** a filled "Choose a file" button, with the line "From your phone, iCloud, Google Drive
  or OneDrive". These come from the OS file picker.
- **Formats line:** "CSV, Excel, JSONL, or a zip with images · up to {cap} MB". The cap comes from
  settings. Excel is new (§10).
- **Samples:** "No data yet? Try a sample: support tickets · claim triage · server logs".

**After upload, top to bottom:**
1. **File row:** the file name, "200 rows · 7 columns", and "Replace".
2. **"Each row is one *ticket*"**, an editable word (§4.2). It shows a small "Guessed from the file
   name" marker while unchanged. The marker disappears once edited.
3. **"A first look."**
   - *Desktop:* the first 4 rows as a table. Each header's underline matches the column's role:
     oak for reads, sage for expected output, muted for ID. The table scrolls sideways inside its
     own box and never makes the page scroll sideways. "See all rows" sits under it.
   - *Phone:* one row as a card of `column: value` lines, with ‹ › and "Row N of M". Only the card
     changes, sliding in from the side you moved toward.
4. **"What's in each column?"** with the subtitle "We've made a first guess. Click a tag to change
   it." ("Tap" on a phone).
   - **Rows:** one per column: the name, sample values, and the role tag.
   - **Changing a role:** a popover on desktop, or a bottom sheet on a phone. It lists the five
     roles with one-line descriptions:
     - *Model reads it:* text or numbers the model reads for each row.
     - *Expected output:* what the model learns to predict; each one becomes a question.
     - *Image:* a picture the model looks at, given as a file name or link.
     - *Row ID:* names each row, so predictions can be matched back to it.
     - *Ignore:* left out completely.
   - **First guess:** from the existing auto-mapping.
5. **Checks** run automatically and appear in two places:
   - **Under the column they concern,** as a small oak "!" line. Example: "Empty in 3 rows. Those
     rows still teach the other questions."
   - **In a summary at the end:**
     - ✓ for passed checks, such as "All 200 rows read, with 7 columns", "Every row has its own
       ID", and "Every row is short enough for the model to read in full";
     - **!** "N things to look at", which still allows Continue;
     - a rust **×** for problems that block Continue. Example: "Choose at least one expected
       output. Mark the column that holds what the model should predict."
   - **When the tokenizer can't be reached,** the token checks say so and why. This was a
     deferred Plan 3 minor.
   - **Technical detail,** such as the token-length histogram, folds under the summary for those
     who want it **(my call)**.
6. **Footer:** "3 expected outputs · 3 columns read" and Continue. On a phone the Continue button
   is sticky and full-width.

### 7.3 Step 2 — What to predict

- **Title:** "What should the model learn to predict?"
- **Lede:** "Each question below is something the model learns to predict from your examples."
- **Column summary line:** "Reads **subject**, **body** · Ignores **ticket_id** · Predicts **3
  questions** · *Change columns*". The link goes back to step 1's column list.
- **Layout (the user picked layout 2):**
  - *Desktop:* a numbered question list on the left (about 160 px). Each entry shows `01`, the
    column name, and the question sentence, which updates as it is typed. The selected entry has
    a sage left border. The editor fills the right side.
  - *Phone:* the list becomes a "Question 1 of 3 ‹ ›" switcher above the editor **(my call; not
    mocked)**.
- **The editor:**
  - **"Asked as" label with a ⓘ button.** Its tooltip shows on hover and on keyboard focus:
    "This sentence becomes part of the **prompt**, the text the model reads. It reads it with
    every example while it's being tuned, and again each time it makes a prediction later. Ask one
    clear question about a single row of your data." (The user rejected "Say it the way you'd ask
    a colleague".)
  - **Question box:** a large text box (about 19 px, light weight, at least 3 lines, resizable).
    Under it: "From the column **department** · 3 answers in your data".
  - **"Type of answer":** three text tabs with a sage underline on the chosen one:
    - *Pick one*, "of named answers" (`choice`);
    - *Scale*, "answers in order" (`score`);
    - *Yes or no*, "true or false" (`noul`).
  - **Answers grid:** Answer | What it means | In your data.
    - *What it means* is edited in place. It is optional, with the placeholder "Optional: what
      this answer means".
    - *In your data* shows "78 tickets" with a thin sage bar relative to the most common answer
      **(header wording my call)**.
    - *Scale:* answers are numbered, with the hint "Lowest first. Drag an answer to reorder the
      scale."
    - *Yes or no:* the hint "Descriptions are optional. The question alone is often enough for yes
      or no."
  - **Checks** for a question appear under its editor, in the same flag style. Examples: rare
    answers, an answer in the data that isn't in the list, and the existing schema validation.
- **"See what the model reads":** a quiet link that opens the exact prompt preview for a sample
  row, with the state, question and answers highlighted. It uses the existing `prompt_preview`.
  It opens in a panel on desktop and a sheet on a phone.
- **Footer:** "3 questions · 200 tickets" and Continue.

### 7.4 Step 3 — Settings

The user chose layout "C": one summary row per choice. Layouts A (three side-by-side columns) and
B (a sentence with inline choices) are kept in the mockups, because the user wants to revisit them
later.

- **Title:** "How should it be trained?"
- **Lede:** "We've picked settings that suit most data. Change anything you like, and the time and
  cost update as you go."
- **Rows:** label | current value with a one-line description | ›
  - **Name:** an inline text field, not a sheet. Under the rows: "Saved to your Hugging Face
    account as **{namespace}/{name}**".
  - **Model:** Clef-Flash ("Fast and affordable · 9B") or Clef ("Most capable · 27B · about 3× the
    cost").
  - **Training:**
    - *Quick test:* "A fast, light run to check everything works".
    - *Balanced:* "A good first real model. Three passes over your examples", marked
      ● Recommended.
    - *Thorough:* "Trains longer and more deeply. Most useful with larger datasets".
- **Changing a row:** a panel just under the row on desktop, or a bottom sheet on a phone. Options
  are listed with descriptions. The training options also show each one's time and cost.
- **Estimate:**
  - large "Time" and "Cost" ranges for the recommended hardware, from `estimate.py`;
  - fine print: "Runs on Hugging Face's cloud on one L40S GPU, billed to your Hugging Face
    account. The run stops itself after **30 minutes**, so it can never cost more than **$0.90**."
    The cap is the suggested job timeout times the hourly price;
  - a note: "We'll set aside 31 of your 200 tickets and test the model on them before and after
    training, so you can see how much it improved."
- **Advanced settings:** folded, labelled "For people who know fine-tuning".
  - **Every config field** gets a row: a plain name, then the technical name in small mono, a
    one-line description, and the value.
  - **Hardware** comes first: "Picked for the lowest cost that fits", with "Change". That opens
    the full machine list with each machine's time, cost and memory fit (the old estimate table).
  - **The mockups show the main ones:** passes over your data (`epochs`), what gets trained
    (`mode`), adapter size (`lora.r`), longest example (`max_length`), memory saving
    (`quantization`), calibrate confidence, and stop early.
  - **Every other field still in `configure_view`** goes here too, such as learning rates, batch
    tokens, accumulation, eval split, class weighting and question weights.
- **Phone:** sticky full-width Continue, with Back under it.

### 7.5 Step 4 — Train

**Ready to train**
- **Title and lede:** "Ready to train", then "Check the details. Training starts on Hugging Face's
  cloud when you press Start, and you can leave while it runs."
- **Summary rows,** each with "Change", which goes back to the step:
  - Model: "Clef-Flash · Balanced";
  - Data: "200 tickets · 3 questions · 169 to learn from, 31 set aside to test it";
  - Saved as: "{ns}/{name} · Private to your Hugging Face account".
- **Time and cost** as in §7.4, then "Billed to your Hugging Face account's prepaid credits, and
  it never costs more than **$X**. *Add credits*". The link goes to the Hugging Face billing page.
- **Account row:** "Signed in as **hvgupta1** · Training runs and is billed under this account ·
  Switch".
- **Not signed in on the Space:** this row becomes a "Sign in with Hugging Face" button, and Start
  stays disabled with the reason shown.
- **"Delete my training data from Hugging Face when it finishes"** is a small checkbox, on by
  default. It keeps the old launch option **(placement my call)**.
- **Local mode with a CUDA GPU:** a "Where should it run? Hugging Face cloud / This computer" row
  appears. "This computer" uses the existing local runner **(my call; not mocked)**.
- **Start:** one filled "Start training" button, with "You pay only for the time it runs, at most
  $X" next to it.
- **Developer link:** "Download the training bundle", for running `clef-tune train` elsewhere
  **(my call)**.
- **Launch errors in plain words.** The key one is no credits (HTTP 402): "Your Hugging Face
  account has no prepaid credits for Jobs yet. *Add credits*, then start again."

**Training (running)**
- **Title:** "Training {name}".
- **Large "About N minutes left",** then "Should be ready around 3:42 PM" in the browser's local
  time.
- **Progress line** with a breathing head, plus "$0.19 so far · at most $0.90".
- **Stages,** with a breathing ring on the current one. They come from the job's stage plus
  `metrics.jsonl` events (`baseline`, `train` with `step`/`total_steps`/`eta_s`/
  `projected_cost`, `eval`, `done`):
  1. Started a machine ("One L40S GPU on Hugging Face"): the job is scheduled or running before
     any event.
  2. Loaded Clef-Flash: install and load are done. This may need a log marker or be merged into
     stage 1 (§13).
  3. Measured the starting point ("Before training it matched every expected output on 2 of 30
     tickets"): from the `baseline` event.
  4. Learning from your examples ("Step 64 of 120"): from `train` events.
  5. Testing the result: from the final `eval`.
  6. Saving to your account: after `done`, until the job completes.
- **Safe-to-leave note** in a sage-wash band: "You can close this page. Training keeps going on
  Hugging Face, and you'll find it under **Your models** when you come back."
- **"Notify me when it's done":** a switch. It asks for browser notification permission and
  notifies when polling sees the end. It only works while some tab is open.
- **"Stop training":** a quiet link. It confirms inline: "Stop now? You'll pay for the time so far
  ($0.19) and won't get a model." The choices are "Stop training" (rust outline) and "Keep going".
- **"Live training curve":** folded, showing the loss over steps for technical users. Desktop only
  **(my call)**.
- **Updates:** the page polls every few seconds.

### 7.6 Results

- **Header:** a check-mark circle that draws itself, then "{name} is ready".
- **Lede:** "We tested it on {N} of your {plural} that it never saw during training, before and
  after."
- **Headline:** "It now matches every expected output on **24 of 30** tickets. Before training, it
  managed 2." With one question, it reads "matches the expected output".
- **With 4 or more questions,** "every expected output" is too harsh a bar, so the headline
  switches to the average across questions: "Across its 8 questions, it's now right 89% of the
  time, up from 41%." The dot grid is dropped in that case **(my call)**.
- **Dot grid:** two rows, Before (oak) and After (sage), one dot per test row, filling in one by
  one. The caption: "Each dot is one test ticket. A filled dot means all three answers matched."
  The wording changes for one or two questions. With more than 60 test rows, each dot stands for
  several rows and the caption says how many **(my call)**.
- **"Question by question":** for each question, its sentence (and id), then:
  - **A before → after line,** from 37% to 93%. The oak dot sits at before and the sage dot slides
    to after.
  - **"Before: right 11 of 30 · After: right 28 of 30".**
  - **One summary line about its mistakes,** from the templates below.
  - **"See the details":** opens the confusion grid. Rows are the expected output and columns the
    model's answer. Matches are sage-wash, mistakes oak-wash, and zeros faint. The caption
    explains the grid. Under it go the technical numbers: macro-F1, log-loss, average error for
    scales, and calibration error before → after. When calibration was skipped (fewer than 20
    test rows), it says so.
- **Run line:** "Training took **25 minutes** and cost **$0.34**, against an estimate of $0.40."
- **"What's next":**
  - one filled "Open on Hugging Face ↗", with "Saved privately to **{repo}**";
  - Predict a new spreadsheet (§7.8);
  - Train again with changes ("Go back to any step. Your data and questions are kept.");
  - For developers: a fold with the `Predictor` snippet and the CLI command.

**Mistake-summary templates.** Answer names are the person's own. For a scale, use each answer's
description when its id is a number, otherwise its id **(my call)**. Ellipses mark variable parts.

| Type | Template |
|---|---|
| any, no mistakes | "No mistakes on this question." |
| Pick one | "Its {n} mistakes: one **{expected}** {noun} marked **{answer}** and {k} **{expected}** {nouns} marked **{answer}**[, and {rest} more]." It lists the two biggest mix-ups, with commas and "and" placed correctly for 1, 2 or 3 parts. |
| Scale, all too high | "When it's wrong, it leans toward the top of the scale (**{top answer}**): all {n} mistakes picked a higher answer." |
| Scale, all too low | the same, with "bottom", "{bottom answer}" and "lower". |
| Scale, mixed | "Its mistakes go both ways: {up} too high and {down} too low." |
| Scale, add-on | " Every mistake was one step off." when true. |
| Yes or no | "It caught all {Y} {nouns} where the answer was **{yes}**." or "It caught {TP} of the {Y} …". Then " {FP} mistakes said yes when the answer was no." or " It never said yes when the answer was no." |

### 7.7 Your models

- **Where:** reached from the header link on every screen. It is also the landing page for a
  signed-in person who comes back.
- **The list:** "Everything you've trained, newest first", one row per run.
  - *Training:* a breathing ring, then "About 12 minutes left · ready around 3:56 PM", with a
    thin progress bar.
  - *Ready:* the headline result, for example "Matched every expected output on 23 of 30 claims,
    up from 3 · Oct 29".
  - *Stopped:* "You stopped it after 4 minutes · $0.12 · Oct 28".
  - *Didn't finish:* a rust ring and the reason in plain words, with what to do. Example: "Hugging
    Face credits ran out. Add credits, then try again."
- **Opening a row:** goes to its running view, Results, or the failure detail.
- **Resuming:** stopped or failed runs that have a checkpoint offer "Try again from where it
  stopped". It shows the resume cost first, using the existing `review_resume`. The CLI's resume
  doesn't show a price, which is a deferred Plan 3 minor that the UI covers.
- **Prediction runs** also appear here, as "Predicting…" and "Predictions ready".
  - **A ready prediction run keeps its spreadsheet,** so it can be downloaded again at any time
    (CSV or Excel). The user decided this.
  - **"Delete files"** on a run removes everything that run stored in the person's bucket, after
    an inline confirmation. Training runs offer it too, for any data that was kept.
- **"Train a new model":** an outline button at the end.

### 7.8 Predict a new spreadsheet

You reach it from a ready model, through Results or Your models. It has three states.

1. **Choose a file**
   - **Header:** the crumb "{model} · Predict", the title "Predict with {model}", and the lede
     "Upload new {nouns}. The model reads the same columns it learned from and fills in its
     answers. Expected outputs aren't needed."
   - **File row:** as in step 1.
   - **Checks:**
     - it found the columns the model reads. When one is missing or renamed, the person picks the
       matching column from a list. If there's no match, the check blocks;
     - every row fits;
     - expected outputs present or not. "That's fine" if they're absent. If they're present,
       Ready also reports how often the model matched them.
   - **Review threshold,** written as a sentence: "Mark a row for review when the model is less
     than *70*% sure of an answer." The number is an editable word (§4.2). Under it: "Its
     confidence was calibrated on your test {nouns}, so 70% sure means right about 70% of the
     time."
   - **Time, cost and cap** from a new prediction estimate (§10), then a filled "Start
     predicting" and "You can leave while it runs."
2. **Running:** the same pattern as training. Its stages are Started a machine → Loaded {model} →
   Reading your {nouns} (640 of 1,240) → Preparing your spreadsheet. It has the safe-to-leave
   note.
3. **Ready**
   - **Summary:** a check, "{n} {nouns} predicted", "Took 11 minutes and cost $0.33."
   - **Downloads:** a filled "Download spreadsheet ↓" (CSV) and an outline "Download as Excel".
   - **Review note:** "**86 {nouns} are marked for review.** The model was less than 70% sure of
     at least one answer. Filter on the 'review' column to find them."
   - **"A first look":**
     - *Desktop:* a table where the person's columns are kept and the added columns are sage.
       Each answer shows its confidence %, in oak when under the threshold, plus a "review"
       column.
     - *Phone:* a card per row.
   - **"What it answered":** per question, a stacked bar of how the answers split, with a legend
     (for example technical 47%, billing 38%, account 15%).

## 8. Phone (360 px)

Every screen is designed at 360 px:
- the compact step bar;
- sticky full-width primary buttons;
- bottom sheets for all choices, which trap focus and close on Escape or a tap on the scrim;
- cards instead of wide tables;
- "Question N of M" on step 2.

Tables never make the page scroll sideways. Desktop targets are 1440 and 1024 px.

## 9. How it is built (my call, not objected to)

- **Server:** a FastAPI app replaces `app.py`. Pages are Jinja2 templates. The app keeps calling
  the existing handlers, and the handlers keep returning plain data. Plain-data fragments become
  template partials.
- **Partial updates:** a small hand-written script (`static/clef-tune.js`) instead of htmx, so no
  third-party file is vendored. Forms marked `data-post` send in the background, and the answer's
  `<template data-swap="#id">` blocks replace only those regions. That is what makes "only the
  part that changed moves" possible. (Built this way during implementation; the design first
  named htmx.)
- **Hand-written JavaScript** in one small module covers the theme toggle, sheets and popovers,
  tooltips, the editable words, notifications, the row-card pager and local-time formatting.
- **Sessions:** a signed cookie holds a session id that maps to the existing in-memory `Session`
  objects. Each session keeps its own temporary folder, and stale sessions are swept with the
  existing `sweep_stale`.
  - A Space restart loses unfinished drafts. That is acceptable: anything already running lives on
    Hugging Face and comes back through "Your models" **(my call)**.
- **Branch:** the work continues on `plan-4-ui`. Nothing from Plan 4 is merged yet, and the
  Gradio-free modules it built live there. A new plan replaces Plan 4's Tasks 8–10, and Plan 4's
  final whole-branch review then covers everything **(my call)**.
- **Sign-in:**
  - *On a Space:* `huggingface_hub.attach_huggingface_oauth(app)` (present in huggingface_hub
    1.33), and `parse_huggingface_oauth(request)` for the token. The Space README keeps
    `hf_oauth: true` and the scopes `read-repos`, `write-repos`, `manage-repos` and `jobs`.
  - *Locally:* the machine's login.
  - The existing `user_token` rule stays: the server's own token is never used for a visitor's
    action.
- **Live views** poll every few seconds (`data-poll`).
- **Hosting:** the Space switches from `sdk: gradio` to `sdk: docker`, with its own small
  Dockerfile serving on port 7860. The training `Dockerfile` is unrelated and unchanged.
- **Dependencies:** the `ui` extra drops Gradio and adds FastAPI, Uvicorn, Jinja2,
  python-multipart and a cookie-signing library. Excel support adds openpyxl; where it lives is
  left to the builder.
- **The `clef-tune ui` command** stays and serves the new app. `--share`, which made a gradio.live
  link, goes.
- **Removed:** `ui/app.py`, `ui/theme.py`, `ui/style.css`, Gradio, and the Gradio-only tests.
- **Tests:**
  - handler and session tests stay;
  - new HTTP-level tests use FastAPI's test client;
  - the opt-in Playwright browser test and `scripts/screenshots.py` are rewritten for the new
    pages, and keep the fake services and the recorded run in `tests/ui/fixtures/`.

## 10. New backend work behind the screens

- **Excel (.xlsx) uploads,** read into the same table as a CSV. Only the first sheet is read
  **(my call)**.
- **Row noun:** the guess, the plurals and the `data.row_noun` config field (§5).
- **Mistake-summary templates** (§7.6), as plain functions with tests.
- **Training stage detection** from job stage plus `metrics.jsonl` (§7.5).
- **Prediction jobs on Hugging Face,** plus local prediction with CUDA outside a Space, mirroring
  local training. Prediction:
  - loads the adapter;
  - reads the uploaded file from the run's private bucket folder;
  - writes the output file there;
  - the app downloads it for the person.

  **The output file stays in the person's private bucket** until they delete it from "Your
  models", so it can be downloaded again later (user's decision). When the delete-data option is
  on, only the uploaded input file is removed when the job finishes. The output keeps the
  person's columns, so it is their data, stored privately in their own account, which matches
  the welcome page's promise.
- **Prediction output format:** the person's columns are kept in order. For each question, add
  `{question}` (the answer) and `{question}_confidence` (0–1, shown as a %), then a final
  `review` column ("yes" or empty). Per-option probabilities are optional extra columns. Excel
  output is written alongside CSV. **(my call)** This one format replaces today's
  `clef-tune predict` CSV (`id`, answer, `q.p.option` columns) for the CLI too, with a flag to add
  the per-option columns, so there is one format to maintain. The project has no outside users
  yet.
- **Prediction estimate:** a time/cost/cap estimate for prediction in `estimate.py`. It is the
  model's fixed overhead plus a forward pass (2 × params × tokens), with the cap from the
  suggested timeout.
- **Two sample datasets:** `claims_triage.csv` and `server_logs.csv`, synthetic and seeded,
  200 rows each, three questions each (one pick-one, one scale, one yes/no), with a config like
  `support_tickets.yaml`. They must be learnable, with real signal in the text.
  - *Claims:* reviewer (standard / fraud review / medical review), risk (Low / Medium / High risk),
    pay in full (yes / no).
  - *Logs:* component (pick one), severity (scale), act now (yes / no).
  - Medical triage was avoided for a public demo **(my call)**.
- **The browser notification** (§7.5).

## 11. Accessibility

- **Contrast:** the tokens in §4.1 meet 4.5:1 for text on `--page` in both themes. Sage text uses
  `--sage-ink`, not `--sage`.
- **Keyboard and screen readers:**
  - every control works by keyboard, with a visible sage focus ring;
  - tooltips open on focus as well as hover and are linked with `aria-describedby`;
  - sheets and popovers move focus in and back out;
  - live progress announces stage changes politely (`aria-live="polite"`), not every step;
  - the line drawing has a text label.
- **Motion:** `prefers-reduced-motion` disables all motion.

## 12. Decided against, deferred, or open

- **Settings layouts A and B:** the user wants to revisit them. Both are in the mockups
  (`05b-settings-a-b.html`).
- **Email when a run finishes:** skipped for now. It would need an email service and the
  person's address.
- **Other Hugging Face decision models:** shelved. The user said "keep it to clef for now".
  - *Cheapest next step:* accept any Hub repo in Clef's release layout by replacing the
    hard-coded model lists with a Hub lookup.
  - *A different family,* such as h2oai/h2o-lightning-31b (Gemma 4 with a one-token
    option-letter readout), would need a second model backend. That is not a rewrite.
- **Rename:** keep `clef-tune` for now. Revisit when a second model family arrives.
  - *"decision-tune" is taken* by a neighbouring product: the PyPI package, the
    `decision-tune` Hugging Face org with DecisionTune 1.0 (a 395M decision model), GitHub, and
    decisiontune.com.
  - *Free as of 2026-10-30* on PyPI and as Hugging Face names: "decisionwright" and "verdictly".
- **"Frontier" in the headline:** not used. Clef-Flash is 9B, and the claim would come before the
  results back it.
- **The live Space deploy** still needs the user's explicit go-ahead, as Plan 4 required. So does
  any push to GitHub.

## 13. Left to the builder

- Module and route layout, template structure, and how handlers' data maps to partials.
- Exact polling intervals, debounce timings, and session storage details within §9's choices.
- How "Loaded the model" is detected, or whether stages 1 and 2 merge. Showing fewer stages is
  acceptable; showing a wrong one is not.
- "See all rows" (a paged view) and the full Advanced settings rows, following the patterns above.
- Handling of very wide files (50+ columns), many questions (12+), long answer names, image
  columns and very long cell text, using the same patterns (scroll inside the table, truncate
  with the full value on focus or hover, and so on).
- Empty, loading and error states for every screen in the same visual language. Every network
  wait gets a calm loading state, not a frozen page.
- Where openpyxl lives in the dependencies.
- Whether the existing SVG charts in `charts.py` are restyled or replaced by the new visuals (dot
  grid, before-after line, stacked bars, confusion grid).

## 14. My calls, for review

1. The FastAPI + Jinja2 + htmx stack, with Gradio removed and a Docker Space (§9).
2. The `data.row_noun` config field (§5).
3. The token histogram folded under the step 1 check summary (§7.2).
4. The phone's "Question N of M" switcher on step 2 (§7.3).
5. "In your data" as the answer-count header (§7.3).
6. The delete-data checkbox on the Ready screen; the "Where should it run?" row in local mode;
   the "Download the training bundle" link (§7.5).
7. The live training curve on desktop only (§7.5).
8. Dot grid scaling above 60 test rows (§7.6).
9. Scale answers named by description when their id is a number (§7.6).
10. Excel: first sheet only (§10).
11. One prediction output format for the UI and the CLI (§10).
12. The claim-triage and server-log samples, and no medical data (§10).
13. Copying the mockups into `docs/` beside this file.
14. The averaged Results headline, without the dot grid, for 4 or more questions (§7.6).
15. Building on `plan-4-ui` with a new plan replacing Plan 4's Tasks 8–10; in-memory sessions
    (§9).

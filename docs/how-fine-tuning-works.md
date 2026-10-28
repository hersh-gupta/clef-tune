# How clef-tune fine-tunes Clef

Plain-language notes on what training changes and what an adapter contains, using the
2026-10-28 GPU smoke tests as a worked example.

## Clef's two parts

```
ticket + questions ──► BACKBONE (Qwen3.5, 32 layers, ~9B weights) ──► JOINT HEAD (~122M weights) ──► a score per answer
                       turns the text into numbers that capture         turns those numbers into
                       its meaning ("hidden states")                    a probability for each option
```

Clef is not a chatbot. It reads a record (for example, a support ticket) and answers a fixed set
of typed questions with probabilities:

| Type | Meaning | Example |
|---|---|---|
| `choice` | one of several named options | department: billing / technical / account |
| `score` | ordered options, indexed from 0 | urgency: 0 "Can wait", 1 "This week", 2 "Today" |
| `noul` | true / false | outage: yes / no |

Clef writes the questions and their options into the prompt next to the record, so the untuned
model can already attempt questions it has never seen. That untuned attempt is the **baseline**
every run reports.

Fine-tuning changes both parts, but in very different ways.

## The joint head

The head is Clef's answering module, and it is small: about 122M weights for Clef-Flash, around
1% of the model. After the backbone has read the prompt, the head:

1. finds where each question and each option ("billing", "Today", …) sits in the text;
2. builds a summary for each option from the backbone's reading of that text, plus the option
   word's own meaning (the backbone's output embedding, which is why `lm_head` is never
   quantized);
3. lets those summaries look back over the record for evidence;
4. scores every option for every question.

A softmax then turns each question's scores into probabilities that add up to 1.

It is **"joint"** because all the questions pass through its layers together, so they can inform
each other. For example, "outage = yes" makes "department = technical" more likely. One forward
pass answers all of them.

**In training,** every weight in the head is updated. It is kept in full fp32 precision with its
own learning rate (`head_lr`, default 5e-5). That is why `joint_head.safetensors` is about
487 MB: 122M numbers × 4 bytes.

## The LoRA layers

**The problem:** the backbone has about 9 billion weights. Updating them directly needs memory for
each weight, its gradient and two optimizer statistics, far more than a 24 GB card has. With
`quantization: 4bit` the base weights are even stored compressed, and 4-bit weights cannot be
trained directly.

**The trick:** freeze every original weight matrix `W` and add a small trainable "detour" next to
it:

```
output = W·x  +  B·(A·x)
         ↑ frozen (4-bit with QLoRA)   ↑ trainable, rank r (default 16)
```

`A` squeezes the input down to `r` numbers, and `B` expands them back out. One of Clef-Flash's
4096 × 12288 matrices holds about 50M weights; its rank-16 detour holds only
16 × (4096 + 12288) ≈ 262K, about 0.5%.

- **Where they go:** clef-tune adds detours to the attention projections, the linear-attention
  projections and the feed-forward block of every language-model layer (`LORA_TARGET_MODULES`
  in `clef_tune/model.py`). For Clef-Flash at rank 16 that is **about 42M trainable weights,
  about 0.6% of the backbone**. The vision tower and the output embedding are left alone.
- **Why the baseline is the untouched model:** `B` starts at zero, so the detour adds nothing at
  first. Training starts from exactly the released model.
- **QLoRA** is LoRA applied on top of 4-bit frozen weights. It is what fits Clef-Flash training
  on a 24 GB GPU.

**Why train both:** the LoRA layers change how the backbone reads records, for example noticing
that "HTTP 500" signals an outage. The head changes how that reading becomes answers.
`mode: head_only` freezes the backbone completely. It is cheaper and faster, but it can only reuse
what the backbone already notices.

## The adapter

The adapter is **everything that differs from the base model, plus what is needed to use it**.
It never contains the base model (about 19 GB for Clef-Flash), only its name and pinned revision.

| File | What it is |
|---|---|
| `adapter/` | the LoRA detours (`A` and `B` for each layer), in standard PEFT format |
| `joint_head.safetensors`, `joint_head_config.json` | the fully retrained head (most of the size) |
| `calibration.json` | one temperature per question that adjusts how confident the probabilities are; 1.0 when a question has fewer than 20 labeled eval rows |
| `schema.yaml` | the questions and options the head was trained on |
| `train_config.yaml`, `metrics.json` | how it was trained; baseline vs. final results |
| `README.md`, plus Clef's code and tokenizer files | the model card, and what is needed to run it |

**Using it:** `Predictor.load(adapter)` downloads the base model, attaches the LoRA detours, swaps
in the new head and applies the calibration.

**For serving:** `clef-tune export --merge` folds each detour into its weight matrix (`W + B·A`)
and writes a standalone, full-size checkpoint in Cloudflare's release layout. Cloudflare's
`load_release_model()` and SGLang can serve it with no LoRA support needed. The trade-off is size:
about 19 GB instead of about 0.5 GB.

## Worked example: the GPU smoke tests (2026-10-28)

`tests/torch/test_gpu.py` is a **smoke test**: it checks that the pipeline runs end to end on the
real model and real hardware, not how good the result is. It passes if training writes an
adapter, every final metric is a real number, and the saved adapter loads back and answers all
three questions for 3 tickets. The tiny-model tests already check the logic on CPU.

### The data

The first 50 rows of `examples/support_tickets.csv`: 40 for training, 10 held out for evaluation.
A typical row:

> **T0003** · "API returning 500 errors" · "Since an hour ago our calls to the API return HTTP
> 500." · enterprise → *technical, urgency 2, outage yes*

Each ticket plus the question prompt comes to about 360 tokens, about 14,400 training tokens in
all.

### What the run did (one L4 GPU, 24 GB)

1. **Loaded Clef-Flash in 4-bit** and froze the base weights.
2. **Added the trainable parts:** rank-16 LoRA detours and the joint head.
3. **Measured the baseline** on the 10 held-out tickets.
4. **Trained one pass over the 40 tickets:**
   - **Batches:** 2 tickets per batch (`max_tokens_per_batch: 1024` fits two), so 20 batches.
   - **Steps:** gradients from 4 batches (`grad_accum_steps`) are added up before each weight
     update, giving 5 updates.
   - **Learning rate:** the cosine schedule decays it to zero by the last update, which is why
     the log shows `lr 0.00e+00` at step 5.
5. **Evaluated again** and kept the best result.
6. **Skipped calibration,** because no question had 20 labeled eval rows.
7. **Saved the adapter,** reloaded it with `Predictor.load` and predicted three tickets.

| On 10 held-out tickets | Before (baseline) | After 5 updates |
|---|---|---|
| Exact match (all 3 answers right) | 1 / 10 | 4 / 10 |
| Mean macro-F1 (0–1) | 0.385 | 0.756 |
| Mean log-loss (lower is better) | 0.855 | 0.470 |

Don't read much into these numbers. With 10 test tickets, one ticket moves exact match by
10 points, and 5 updates is far less training than a real run gets. They only show training moved
the model in the right direction.

### The 2-GPU run (DDP, 2× A10G)

The same job, split across two GPUs:

- **Copies:** each GPU holds a full copy of the model.
- **Work split:** each GPU takes 10 of the 20 batches.
- **Updates:** after every update the GPUs average their gradients over NCCL, giving 3 updates
  of 4, 4 and 2 batches.
- **Push:** only GPU 0 saves and pushes the adapter, while GPU 1 waits for the result.

It pushed its adapter to the private Hub repo `hvgupta1/clef-tune-ddp-smoke`, which is a good
place to see the adapter layout above. Predictions from that adapter:

| Ticket | Label | Prediction |
|---|---|---|
| T0003 "API returning 500 errors" | technical, 2, outage yes | technical (97%), urgency "Today" (55%), outage yes (83%) |
| T0002 "Invoice 1410 looks wrong" | billing, 1, outage no | billing (96%), urgency "Today" (56%) instead of "This week", outage no (99%) |

A `score` answer's `score` field is the expected value. For T0001,
0 × 0.16 + 1 × 0.22 + 2 × 0.62 = 1.46.

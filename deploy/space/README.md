---
title: clef-tune
emoji: 🎼
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_scopes:
  - read-repos
  - write-repos
  - manage-repos
  - jobs
pinned: false
short_description: Bespoke intelligence, owned by you. Tune decision models.
---

# clef-tune

Upload a spreadsheet of examples with the expected outputs filled in. clef-tune fine-tunes
Cloudflare's open [Clef](https://huggingface.co/Cloudflare/clef) decision models to predict
them for new cases, and shows how much better the model got before you use it.

1. **Your data:** upload a CSV, Excel or JSONL file (or try a sample), and say what each column
   is for.
2. **What to predict:** each expected output becomes a plain question.
3. **Settings:** pick how much training; the time and the most it can cost are shown first.
4. **Train:** sign in with Hugging Face and start. It runs on
   [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) under your
   account and your prepaid credits, and saves the model to a private repo you own. You can
   leave and come back; "Your models" shows where everything stands.

Then predict a new spreadsheet with the model, and download it with the answers filled in.

This Space never uses its own token for your actions: everything runs as the person signed in.

Source: <https://github.com/hersh-gupta/clef-tune>

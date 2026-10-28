---
title: clef-tune
emoji: 🎼
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
hf_oauth: true
hf_oauth_expiration_minutes: 480
hf_oauth_scopes:
  - read-repos
  - write-repos
  - manage-repos
  - jobs
pinned: false
short_description: Fine-tune Cloudflare's Clef decision models on your own data
---

# clef-tune

Fine-tune Cloudflare's [Clef](https://huggingface.co/Cloudflare/clef) decision models on your
own labeled data, from a CSV to a trained adapter, without writing YAML.

1. Upload a CSV, TSV or JSONL file, or load the example.
2. Check the proposed questions and the prompt Clef will see.
3. Review the data checks and the time and cost estimate.
4. Sign in with Hugging Face and launch training on
   [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). It runs on
   your account and your pre-paid credits, and pushes the adapter to a private repo you own.

This Space never uses its own token for your actions. If a launch fails with a permission
error, paste a [token](https://huggingface.co/settings/tokens) with `job.write` and
`repo.write` in the Hugging Face Jobs tab; it is kept for your browser session only.

Source: <https://github.com/hersh-gupta/clef-tune>

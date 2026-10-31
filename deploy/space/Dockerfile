# The clef-tune web UI as a Hugging Face Docker Space. Training runs on Hugging Face Jobs, so
# this container needs no GPU and no torch. scripts/prepare_space.py writes requirements.txt,
# pinned to an exact clef-tune commit on GitHub.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user app.py .

EXPOSE 7860
# Spaces terminate TLS in front of the container; trust its forwarded headers so the OAuth
# redirect goes back to https.
CMD uvicorn app:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips "*"

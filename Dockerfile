FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
# The base image's torch lives in Ubuntu's system Python, which PEP 668 marks externally managed.
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1
WORKDIR /opt/clef-tune
COPY . .
RUN pip install ".[train]"
# Kernels speed up Clef's linear-attention layers; without them training still works, slower.
# causal-conv1d must compile against the installed torch, so build it without isolation.
# transformers imports any installed copy, so a build that cannot import is removed.
RUN (pip install --no-build-isolation causal-conv1d==1.7.0 flash-linear-attention==0.6.0 \
        && python -c "import causal_conv1d") \
    || (pip uninstall -y causal-conv1d flash-linear-attention fla-core \
        && echo "WARNING: kernel packages failed to install; linear attention will use the slow PyTorch path")
ENTRYPOINT ["clef-tune"]

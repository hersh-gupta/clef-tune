FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /opt/clef-tune
COPY . .
RUN pip install ".[train]"
# Kernels speed up Clef's linear-attention layers; without them training still works, slower.
RUN pip install flash-linear-attention causal-conv1d \
    || echo "WARNING: kernel packages failed to install; linear attention will use the slow PyTorch path"
ENTRYPOINT ["clef-tune"]

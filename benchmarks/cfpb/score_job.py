"""Runs inside a scoring job on Hugging Face Jobs, next to an installed clef-tune.

  base CONFIG OUT          save the released model, untrained, as an adapter (the head as
                           released, no LoRA): "out of the box", exactly what training starts from
  fetch REPO OUT           download a trained adapter and point it at the mounted base model
  predict ADAPTER CSV OUT TIMING
                           predict every row with probabilities, timing only the prediction

Uses clef-tune's own code paths, so scores match `clef-tune eval` and `clef-tune predict`.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

MODEL_MOUNT = "/model"


def base(config_path: str, out: str) -> None:
    import torch

    import clef_tune.adapter as adapter
    from clef_tune.clef_code import load_clef_code, model_dir
    from clef_tune.config import load_config
    from clef_tune.model import load_head

    config = load_config(config_path)
    path = model_dir(config)
    head, head_config = load_head(
        path, load_clef_code(path), device=torch.device("cpu"), dtype=torch.float32
    )
    model = SimpleNamespace(language_model=None, head=head)  # no LoRA: the backbone as released
    card = adapter.render_card  # it needs training metrics, which an untrained model has none of
    adapter.render_card = lambda *args, **kwargs: f"# {config.run_name}, untrained\n"
    try:
        adapter.save_adapter(out, model, head_config, config, calibration={}, metrics={})
    finally:
        adapter.render_card = card
    print(f"saved the untrained model as an adapter in {out}")


def fetch(repo: str, out: str) -> None:
    from huggingface_hub import snapshot_download

    from clef_tune.adapter import CONFIG_FILE
    from clef_tune.config import load_config
    from clef_tune.yamlio import dump_yaml

    path = Path(snapshot_download(repo, local_dir=out))
    config_file = path / CONFIG_FILE
    data = load_config(config_file).to_portable_dict()
    data["model"]["local_path"] = MODEL_MOUNT
    config_file.write_text(dump_yaml(data), encoding="utf-8")
    print(f"downloaded {repo} to {path}; base model at {MODEL_MOUNT}")


def predict(adapter_dir: str, csv_path: str, out_csv: str, timing_path: str) -> None:
    import torch

    from clef_tune.predictor import Predictor, predict_file

    predictor = Predictor.load(adapter_dir, base_model_path=MODEL_MOUNT)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    rows = predict_file(predictor, csv_path, out_csv, calibrated=True, probabilities=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    Path(timing_path).write_text(
        json.dumps({"rows": rows, "seconds": seconds, "gpu": gpu}), encoding="utf-8"
    )
    print(f"predicted {rows} rows in {seconds:.1f} s on {gpu}")


def main() -> None:
    command, *args = sys.argv[1:]
    {"base": base, "fetch": fetch, "predict": predict}[command](*args)


if __name__ == "__main__":
    main()

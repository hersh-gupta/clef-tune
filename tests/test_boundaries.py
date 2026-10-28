import subprocess
import sys

NO_TORCH_MODULES = [
    "clef_tune",
    "clef_tune.cli",
    "clef_tune.bundle",
    "clef_tune.validation",
    "clef_tune.estimate",
    "clef_tune.calibration",
    "clef_tune.metrics",
    "clef_tune.runtime",
    "clef_tune.clef_code",
    "clef_tune.weights",
    "clef_tune.batching",
    "clef_tune.card",
    "clef_tune.hub",
    "clef_tune.adapter",
    "clef_tune.launch",
    "clef_tune.launch.source",
]


def test_core_modules_never_import_torch():
    imports = "; ".join(f"import {module}" for module in NO_TORCH_MODULES)
    check = "bad = [m for m in ('torch', 'transformers', 'peft') if m in sys.modules]"
    code = f"import sys; {imports}; {check}; assert not bad, bad"
    subprocess.run([sys.executable, "-c", code], check=True)

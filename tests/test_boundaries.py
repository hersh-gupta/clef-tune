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
    "clef_tune.launch.job_script",
    "clef_tune.launch.hf_jobs",
    "clef_tune.automap",
    "clef_tune.ui",
    "clef_tune.ui.charts",
    "clef_tune.ui.fragments",
    "clef_tune.ui.settings",
    "clef_tune.ui.session",
    "clef_tune.ui.services",
    "clef_tune.ui.handlers",
    "clef_tune.ui.runs",
]

# The UI's logic stays importable and testable without gradio; only the layout imports it.
NO_GRADIO_MODULES = [
    "clef_tune.ui.charts",
    "clef_tune.ui.fragments",
    "clef_tune.ui.settings",
    "clef_tune.ui.session",
    "clef_tune.ui.services",
    "clef_tune.ui.handlers",
    "clef_tune.ui.runs",
]


def test_core_modules_never_import_torch():
    imports = "; ".join(f"import {module}" for module in NO_TORCH_MODULES)
    check = "bad = [m for m in ('torch', 'transformers', 'peft') if m in sys.modules]"
    code = f"import sys; {imports}; {check}; assert not bad, bad"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ui_logic_never_imports_gradio():
    imports = "; ".join(f"import {module}" for module in NO_GRADIO_MODULES)
    code = f"import sys; {imports}; assert 'gradio' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)

"""Paths and parameters. Folders can be overridden with env vars (used by the tests)."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).parent


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def data_dir() -> Path:
    return Path(os.environ.get("PDM_DATA_DIR", ROOT / load_params()["data_dir"]))


def model_dir() -> Path:
    return Path(os.environ.get("PDM_MODEL_DIR", ROOT / "models"))


def report_dir() -> Path:
    return Path(os.environ.get("PDM_REPORT_DIR", ROOT / "reports"))

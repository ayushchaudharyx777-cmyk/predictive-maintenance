import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def dirs(tmp_path_factory):
    """Synthetic data + a trained model in temp folders (the real data/ and models/ are never touched)."""
    base = tmp_path_factory.mktemp("pdm")
    mp = pytest.MonkeyPatch()
    mp.setenv("PDM_DATA_DIR", str(base / "data"))
    mp.setenv("PDM_MODEL_DIR", str(base / "models"))
    mp.setenv("PDM_REPORT_DIR", str(base / "reports"))
    from make_synthetic_data import make

    make(base / "data")
    import train

    train.main()
    yield base
    mp.undo()


@pytest.fixture(scope="session")
def raw(dirs):
    from features import load_raw

    return load_raw(dirs / "data")

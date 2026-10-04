import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import load_config  # noqa: E402

@pytest.fixture
def config(tmp_path):
    """config/dataset.yaml with every output folder moved into a temporary folder"""
    config = load_config()
    config['paths'] = {key: tmp_path / path.name for key, path in config['paths'].items()}
    return config

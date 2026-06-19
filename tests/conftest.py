import os
import sys
from pathlib import Path

# MLflow local file backend is required; opt out of the v3 gate for tests too.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Register structured configs so compose() sees the schema.
import research_pipeline.config_schema  # noqa: E402,F401

CONFIG_DIR = str(ROOT / "configs")
PROJECT_ROOT = ROOT

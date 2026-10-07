from __future__ import annotations

import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("IRODORI_PROJECT_DIR", APP.parent)).expanduser().resolve()
DATA = Path(os.environ.get("IRODORI_TTS_DATA", ROOT / "tts_workspace")).expanduser().resolve()
RUNTIME = Path(os.environ.get("IRODORI_RUNTIME_DIR", ROOT / ".runtime" / "Irodori-TTS")).expanduser().resolve()
CACHE = Path(os.environ.get("IRODORI_MODEL_DIR", ROOT / "models")).expanduser().resolve()

sys.path.insert(0, str(RUNTIME))
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ.setdefault("GRADIO_TEMP_DIR", str(DATA / "temp"))
os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY"), "127.0.0.1", "localhost", "::1"]))


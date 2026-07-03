from pathlib import Path
import sys

_TEACHING_ROOT = Path(__file__).resolve().parents[2]
if str(_TEACHING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEACHING_ROOT))

from common.hf_loader import build_engine_from_pretrained, load_model_config_from_pretrained
from common.tokenizer import HFTokenizer, SimpleTokenizer

from .config import EngineConfig, ModelConfig
from .engine import SerialEngine, SimpleVLLMEngine

__all__ = [
    "EngineConfig",
    "HFTokenizer",
    "ModelConfig",
    "SerialEngine",
    "SimpleTokenizer",
    "SimpleVLLMEngine",
    "build_engine_from_pretrained",
    "load_model_config_from_pretrained",
]

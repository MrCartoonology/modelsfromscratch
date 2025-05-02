import sys
import ipdb
import traceback
import torch
import yaml
from pathlib import Path
from typing import List, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

from peft import get_peft_model, LoraConfig, TaskType


def debug_hook(type_, value, tb):
    traceback.print_exception(type_, value, tb)
    print("\n--- entering post-mortem debugger ---")
    ipdb.post_mortem(tb)


def load_config(path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    root_path = find_root()
    return replace_root_in_dict(d=cfg, root_path=root_path)


def find_root() -> str:
    # Get absolute path to current file
    current_file = Path(__file__).resolve()

    # Make sure we are where we think we are
    assert current_file.name == "setup.py", f"Unexpected file: {current_file.name}"
    assert (
        current_file.parent.name == "modelsfromscratch"
    ), f"Expected 'modelsfromscratch', got {current_file.parent.name}"
    assert (
        current_file.parent.parent.name == "src"
    ), f"Expected 'src', got {current_file.parent.parent.name}"

    return current_file.parent.parent.parent


def replace_root_in_dict(d: dict, root_path: str) -> dict:
    """Recursively replace '{{root}}' with root_path in all string values."""
    if isinstance(d, dict):
        return {k: replace_root_in_dict(v, root_path) for k, v in d.items()}
    elif isinstance(d, list):
        return [replace_root_in_dict(item, root_path) for item in d]
    elif isinstance(d, str):
        return d.replace("{{root}}", str(root_path))
    else:
        return d


def safe_open(file_path, mode="w"):
    """Ensure parent directory exists, then open the file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode)


class RunTracker(object):
    def __init__(self, cfg: dict):
        if cfg["drop_into_debugger_on_error"]:
            print("Setting up post-mortem debugger")
            sys.excepthook = debug_hook

        self.cfg = cfg
        self.tokenizer = None
        self.dataset = None
        self.model = None
        self.trainer = None

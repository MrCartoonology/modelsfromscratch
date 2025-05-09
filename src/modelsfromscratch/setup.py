import os
import sys
import yaml
import ipdb
import traceback
import yaml
from pathlib import Path
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from modelsfromscratch.utils import get_timestamp


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
        self.optimizer = None
        self.trainer = None
        self.timestamp = None
        self.logdir = None
        self.savedir = None
        self.train_writer = None
        self.val_writer = None
        self.train_losses = []
        self.val_losses = []
        self.prompts = []


def save_model_and_meta(res: RunTracker, epoch: int = 0) -> str:
    cfg = res.cfg['train']
    if not cfg['save']:
        return ""

    savedir = res.savedir

    assert savedir, "run tracker save dir not set"

    cfg_fname = os.path.join(savedir, "config.yaml")
    with open(cfg_fname, "w") as f:
        yaml.dump(res.cfg, f, sort_keys=False, default_flow_style=False, indent=2)

    meta_dict = dict(
        logdir=res.logdir,
        savedir=res.savedir,
        train_losses=res.train_losses,
        val_losses=res.val_losses,
        prompts=res.prompts,
    )
    for split, dl in res.dataloaders.items():
        meta_dict[split] = dict(
            num_chunks=dl['num_chunks'],
            files=dl['files'],
            mb=dl['mb'],
            num_tokens=dl['num_tokens'],
            n_steps=len(dl['dataloader'])
            )
    meta_fname = os.path.join(savedir, "meta.yaml")
    with open(meta_fname, "w") as f:
        yaml.dump(meta_dict, f, sort_keys=False, default_flow_style=False, indent=2)
    print(f"Saved meta to {meta_fname}")

    checkpoint_pth = os.path.join(savedir, f'model_epoch{epoch:02d}.pt')
    torch.save({
            'epoch': epoch,
            'model_state_dict': res.model.state_dict(),
            'optimizer_state_dict': res.optimizer.state_dict(),
        }, checkpoint_pth)
    print(f"Saved model to {checkpoint_pth}")

    
def setup_training(res: RunTracker) -> RunTracker:
    # Get timestamp and create log directory
    res.timestamp = get_timestamp()
    res.logdir = res.cfg["train"]["logdir"].format(timestamp=res.timestamp)
    res.savedir = res.cfg["train"]["savedir"].format(timestamp=res.timestamp)
    os.makedirs(res.logdir, exist_ok=True)
    os.makedirs(res.savedir, exist_ok=True)
    res.train_writer = SummaryWriter(res.logdir + "/train")
    res.val_writer = SummaryWriter(res.logdir + "/val")
    return res


def setup_optim(model: nn.Module, cfg: dict) -> optim.Optimizer:
    name = cfg["name"].lower()
    optimizer = None
    args = dict(lr=cfg["learning_rate"])
    if name == "adam":
        optimizer = optim.Adam
        args.update(cfg["adam_args"])
    elif name == "adamw":
        optimizer = optim.AdamW
        args.update(cfg["adam_args"])

    if optimizer is None:
        raise ValueError(f"unknown optimizer name: {name}")

    return optimizer(model.parameters(), **args)



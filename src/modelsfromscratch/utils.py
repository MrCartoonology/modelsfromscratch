import datetime
import torch
from typing import Iterator, Tuple


def get_timestamp():
    """Get current timestamp in YYYYMMDD-HHMM format"""
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")


def get_device(device="auto"):
    """
    Get the best available device for PyTorch operations.

    Args:
        device (str): Desired device. If 'auto', will automatically select the best available device.
                     Options: 'auto', 'cpu', 'cuda', 'mps'

    Returns:
        str: The device to use ('cpu', 'cuda', or 'mps')
    """
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    return device

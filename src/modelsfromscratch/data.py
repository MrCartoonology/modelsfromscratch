import os
from typing import List, Tuple, Generator
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

from modelsfromscratch.setup import RunTracker
import random


def get_tokenizer(cfg: dict) -> AutoTokenizer:
    # Load tokenizer from Hugging Face Transformers
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    print(f"Tokenizer: {cfg['tokenizer']} num_token_ids: {len(tokenizer)}")
    return tokenizer


def iter_files(
    data_dir: str,
    include_hidden_dirs: bool,
    include_hidden_files: bool,
    only_include_specified_extensions: bool,
    specified_extensions: List[str],
) -> Generator[Tuple[str, int], None, None]:

    # Iterate over all files in the directory and subdirectories
    for root, dirs, files in os.walk(data_dir):
        if not include_hidden_dirs:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        if not include_hidden_files:
            files[:] = [f for f in files if not f.startswith(".")]
        if only_include_specified_extensions:
            files[:] = [f for f in files if f.endswith(tuple(specified_extensions))]
        for fname in files:
            pth = os.path.join(root, fname)
            sz_bytes = os.path.getsize(pth)
            yield pth, sz_bytes


def apply_max_mb_filter(
    files_and_sizes: List[Tuple[str, int]], max_mb: float
) -> List[Tuple[str, int]]:
    max_bytes = max_mb * 1024 * 1024
    selected_files_and_sizes = []
    total_size = 0

    random.shuffle(files_and_sizes)  # Shuffle the files randomly
    for file, size in files_and_sizes:
        if total_size + size < max_bytes:
            selected_files_and_sizes.append((file, size))
            total_size += size

    return selected_files_and_sizes


def split_files(
    files_and_sizes: List[Tuple[str, int]], train_ratio: float, verbose: bool = False
) -> Tuple[List[str], List[str], float, float]:
    total_mb = sum(size for _, size in files_and_sizes) / (1024 * 1024)
    train_mb = train_ratio * total_mb

    train_files_and_sizes = apply_max_mb_filter(
        files_and_sizes=files_and_sizes, max_mb=train_mb
    )
    train_files = [f for f, _ in train_files_and_sizes]
    train_mb = sum(size for _, size in train_files_and_sizes) / (1024 * 1024)
    val_files = [f for f, _ in files_and_sizes if f not in train_files]
    val_mb = total_mb - train_mb

    if verbose:
        print(f"Train size: {train_mb:.2f} MB")
        print(f"Validation size: {val_mb:.2f} MB")
        print(f"Number of training files: {len(train_files)}")
        print(f"Number of validation files: {len(val_files)}")
    return train_files, val_files, train_mb, val_mb


def report_files_and_sizes(files_and_sizes: List[Tuple[str, int]]) -> None:
    orig_total = sum(size for _, size in files_and_sizes) / (1024 * 1024)
    files_and_sizes = apply_max_mb_filter(
        files_and_sizes=files_and_sizes, max_mb=cfg["max_mb"]
    )
    new_total = sum(size for _, size in files_and_sizes) / (1024 * 1024)
    print(f"Original total size: {orig_total:.2f} MB")
    print(f"New total size: {new_total:.2f} MB")
    print(f"Number of files: {len(files_and_sizes)}")


def get_dataloaders(res: RunTracker) -> DataLoader:
    cfg = res.cfg["dataloader"]
    files_and_sizes = list(iter_files(**cfg["files"]))
    report_files_and_sizes(files_and_sizes)

    train_files, val_files, train_mb, val_mb = split_files(
        files_and_sizes, cfg["train_ratio"], verbose=True
    )

    dataloaders = dict()
    seq_len = cfg["seq_len"]
    batch_size = cfg["batch_size"]

    for split, files, mb in zip(["train", "val"], [train_files, val_files], [train_mb, val_mb]):
        token_ids = files_to_token_ids(files=files, tokenizer=res.tokenizer)
        # 3. Chunk into sequences
        num_chunks = len(token_ids) // seq_len
        inputs = token_ids[: num_chunks * seq_len].view(num_chunks, seq_len)
        targets = token_ids[1 : num_chunks * seq_len + 1].view(num_chunks, seq_len)

        dataset = TensorDataset(inputs, targets)
        dataloaders[split] = dict(
            dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True),
            num_chunks=num_chunks,
            files=files,
            mb=mb,
        )
    return dataloaders


def files_to_token_ids(files: List[str], tokenizer: AutoTokenizer) -> torch.Tensor:
    all_text = ""
    for fname in files:
        with open(fname, "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"
    token_ids = tokenizer.encode(all_text, add_special_tokens=False)
    token_ids = torch.tensor(token_ids)
    return token_ids
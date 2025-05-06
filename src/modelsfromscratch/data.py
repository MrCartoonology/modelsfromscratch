import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

from modelsfromscratch.setup import RunTracker


def get_tokenizer(cfg: dict) -> AutoTokenizer:
    # Load tokenizer from Hugging Face Transformers
    return AutoTokenizer.from_pretrained(cfg["tokenizer"])


def get_dataloader(res: RunTracker) -> DataLoader:
    cfg = res.cfg["dataloader"]
    seq_len = cfg["seq_len"]
    batch_size = cfg["batch_size"]
    data_dir = cfg["data_dir"]
    include_hidden_dirs = cfg["include_hidden_dirs"]

    # 1. Load and concatenate all text files
    all_text = ""
    for fname in os.listdir(data_dir):
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            all_text += f.read() + "\n"

    # 2. Tokenize entire corpus
    token_ids = res.tokenizer.encode(all_text, add_special_tokens=False)
    token_ids = torch.tensor(token_ids)

    # 3. Chunk into sequences
    num_chunks = len(token_ids) // seq_len
    inputs = token_ids[: num_chunks * seq_len].view(num_chunks, seq_len)
    targets = token_ids[1 : num_chunks * seq_len + 1].view(num_chunks, seq_len)

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

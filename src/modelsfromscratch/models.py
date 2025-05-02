import numpy as np
import torch
import torch.nn as nn
from modelsfromscratch.setup import RunTracker


class BasicRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super(BasicRNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        rnn_out, _ = self.rnn(x)  # [batch_size, seq_len, hidden_dim]
        logits = self.fc(rnn_out)  # [batch_size, seq_len, vocab_size]
        return logits


class TransformerModel(nn.Module):
    def __init__(
        self, vocab_size, seq_len, embedding_dim, wave_dim, petype="sin"
    ):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.wave_dim = wave_dim
        self.petype = petype
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if petype == "sin":
            self.register_buffer(
                "pe",
                torch.tensor(
                    get_positional_encoding(slen=seq_len, emb_dim=embedding_dim, wave_dim=wave_dim), dtype=torch.float32
                ),
            )
        else:
            raise ValueError(f"Positional encoding type {petype} not supported.")
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        x = x + self.pe[: x.size(1), :]  # Add positional encoding
        logits = self.fc(x)  # [batch_size, sbatch_size, vocab_size]
        return logits


def get_positional_encoding(slen, emb_dim, wave_dim=10000):
    pos = np.arange(slen)[:, np.newaxis]  # (slen, 1)
    i = np.arange(emb_dim)[np.newaxis, :]  # (1, emb_dim)
    pow = (2 * i) / emb_dim
    angle_rates = 1 / (wave_dim**pow)
    angle_radians = pos * angle_rates

    encoding = np.zeros((slen, emb_dim))
    encoding[:, 0::2] = np.sin(angle_radians)[:, 0::2]
    encoding[:, 1::2] = np.cos(angle_radians)[:, 1::2]
    return encoding


def load_model(res: RunTracker) -> nn.Module:
    cfg = res.cfg
    tokenizer = res.tokenizer

    model_name = cfg["model_name"]
    assert (
        model_name in cfg["models"]
    ), f"Model {model_name} not found in config. Available models: {cfg['models'].keys()}"
    seq_len = cfg["dataloader"]["seq_len"]
    mdl_cfg = cfg["models"][model_name]

    if model_name == "basic_rnn":
        return BasicRNNModel(vocab_size=tokenizer.vocab_size, **mdl_cfg)
    elif model_name == "transformer":
        return TransformerModel(vocab_size=tokenizer.vocab_size, seq_len=seq_len, **mdl_cfg)
    raise ValueError(f"Model {model_name} not implemented yet.")

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
    def __init__(self, vocab_size, seq_len, embedding_dim, wave_dim, petype="sin"):
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.wave_dim = wave_dim
        self.petype = petype
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        pe_np = get_positional_encoding(
            slen=seq_len, emb_dim=embedding_dim, wave_dim=wave_dim
        )
        self.register_buffer("pe", torch.tensor(pe_np, dtype=torch.float32))
        assert petype in [
            "sin",
            "rot",
        ], f"Positional encoding type {petype} not supported."
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        if self.petype == "sin":
            x = x + self.pe[: x.size(1), :]  # Add positional encoding
        elif self.petype == "rot":
            pass
        logits = self.fc(x)  # [batch_size, sbatch_size, vocab_size]
        return logits


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
        return TransformerModel(
            vocab_size=tokenizer.vocab_size, seq_len=seq_len, **mdl_cfg
        )
    raise ValueError(f"Model {model_name} not implemented yet.")

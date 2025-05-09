import torch
import torch.nn as nn
from modelsfromscratch.setup import RunTracker
from modelsfromscratch.transformer import TransformerLM
from transformers import AutoTokenizer


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


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_mb = total_bytes / (1024**2)
    return total_params, trainable_params, total_mb


def load_model(res: RunTracker) -> nn.Module:
    cfg = res.cfg
    tokenizer = res.tokenizer
    return load_model_from(cfg=cfg, tokenizer=tokenizer)


def load_model_from(cfg: dict, tokenizer: AutoTokenizer):
    num_token_ids = len(tokenizer)
    model_name = cfg["model_name"]
    assert (
        model_name in cfg["models"]
    ), f"Model {model_name} not found in config. Available models: {cfg['models'].keys()}"
    seq_len = cfg["dataloader"]["seq_len"]
    mdl_cfg = cfg["models"][model_name]
    device = cfg["device"]
    # Determine the actual device to use
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    if model_name == "basic_rnn":
        model = BasicRNNModel(vocab_size=num_token_ids, **mdl_cfg)
    elif model_name == "transformer":
        model = TransformerLM(
            num_token_ids=num_token_ids, seq_len=seq_len, device=device, **mdl_cfg
        ).to(device)
    else:
        raise ValueError(f"Model {model_name} not implemented yet.")

    print(model)
    total, trainable, size_mb = count_parameters(model)
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Model size:       {size_mb:.2f} MB")
    return model

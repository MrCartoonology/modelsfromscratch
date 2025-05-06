import torch.nn as nn
from modelsfromscratch.multiheadattn import MultiHeadAttn
from modelsfromscratch.positionalencoding import RotationalPositionalEncoding


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        depth: int = 6,
        num_heads: int = 4,
        ff_hidden_dim=2048,
        seq_len=512,
    ):
        super(TransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.depth = depth
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=model_dim
        )

        self.pos_embed = RotationalPositionalEncoding(
            wave_dim=10000, model_dim=model_dim
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim=model_dim, ff_dim=ff_hidden_dim, num_heads=num_heads
                )
                for _ in range(depth)
            ]
        )
        self.logits = nn.Linear(in_features=model_dim, out_features=vocab_size)

    def forward(self, inputs):
        X = self.token_embed(inputs)
        X = self.pos_embed(X)
        for i in range(self.depth):
            X = self.transformer_blocks[i](X)
        logits = self.logits(X)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.multi_headed_attn = MultiHeadAttn(model_dim=model_dim, n_heads=num_heads)
        self.ln1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=ff_dim),
            nn.GELU(),
            nn.Linear(in_features=ff_dim, out_features=model_dim),
        )
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, X):
        attn_out = self.multi_headed_attn(X)
        ln1_out = self.ln1(X + attn_out)
        ff_out = self.ff(ln1_out)
        ln2_out = self.ln2(ln1_out + ff_out)
        return ln2_out

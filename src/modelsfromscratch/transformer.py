import torch.nn as nn
from modelsfromscratch.multiheadattn import MultiHeadAttnWithRoPE
from modelsfromscratch.positionalencoding import RotationalPositionalEncoding


class TransformerLM(nn.Module):
    def __init__(
        self,
        num_token_ids: int,
        model_dim: int,
        depth: int = 6,
        num_heads: int = 4,
        ff_hidden_dim=2048,
        seq_len=512,
        freq_base=10000,
        device="cpu",
    ):
        super(TransformerLM, self).__init__()
        self.num_token_ids = num_token_ids
        self.model_dim = model_dim
        self.depth = depth
        self.num_heads = num_heads
        self.ff_hidden_dim = ff_hidden_dim
        self.seq_len = seq_len

        self.token_embed = nn.Embedding(
            num_embeddings=num_token_ids, embedding_dim=model_dim
        )

        self.rope_encoder = RotationalPositionalEncoding(
            freq_base=freq_base, model_dim=model_dim, device=device, seq_len=seq_len
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim=model_dim,
                    ff_dim=ff_hidden_dim,
                    num_heads=num_heads,
                    rope_encoder=self.rope_encoder,
                )
                for _ in range(depth)
            ]
        )
        self.logits = nn.Linear(in_features=model_dim, out_features=num_token_ids)

    def forward(self, inputs, return_attn=False):
        X = self.token_embed(inputs)
        transformer_block_attn_weights = []

        for i in range(self.depth):
            res = self.transformer_blocks[i](X, return_attn=return_attn)
            if return_attn:
                X, attn_weights = res
                transformer_block_attn_weights.append(attn_weights)
            else:
                X = res
        logits = self.logits(X)

        if return_attn:
            return logits, transformer_block_attn_weights
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, model_dim, ff_dim, num_heads, rope_encoder):
        super(TransformerBlock, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.rope_encoder = rope_encoder

        self.multi_headed_attn = MultiHeadAttnWithRoPE(
            model_dim=model_dim, num_heads=num_heads, rope_encoder=rope_encoder
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(in_features=model_dim, out_features=ff_dim),
            nn.GELU(),
            nn.Linear(in_features=ff_dim, out_features=model_dim),
        )
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, X, return_attn=False):
        res = self.multi_headed_attn(X, return_attn=return_attn)
        if return_attn:
            attn_out, attn_weights = res
        else:
            attn_out = res

        ln1_out = self.ln1(X + attn_out)
        ff_out = self.ff(ln1_out)
        ln2_out = self.ln2(ln1_out + ff_out)
        if return_attn:
            return ln2_out, attn_weights
        return ln2_out

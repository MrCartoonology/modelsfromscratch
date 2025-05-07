import torch
import torch.nn as nn
from modelsfromscratch.positionalencoding import RotationalPositionalEncoding


class MultiHeadAttnWithRoPE(nn.Module):
    def __init__(
        self, model_dim: int, num_heads: int, rope_encoder: RotationalPositionalEncoding
    ):
        super(MultiHeadAttnWithRoPE, self).__init__()
        assert model_dim % num_heads == 0, "model dim must be divisible by n_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.rope_encoder = rope_encoder

        self.head_dim = model_dim // num_heads

        self.Q = nn.Linear(model_dim, model_dim)
        self.K = nn.Linear(model_dim, model_dim)
        self.V = nn.Linear(model_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)

    def forward(self, X):
        _, S, _ = X.size()
        q = self.rope_encoder(self.Q(X))  # B x S x D
        k = self.rope_encoder(self.K(X))
        v = self.V(X)

        mask = torch.tril(torch.ones(S, S, device=X.device)).unsqueeze(0)  # [1, S, S]

        outputs = []
        for head in range(self.num_heads):
            d1 = head * self.head_dim
            d2 = d1 + self.head_dim
            qh = q[:, :, d1:d2]
            kh = k[:, :, d1:d2]
            vh = v[:, :, d1:d2]
            attn_vh = calc_attn(Q=qh, K=kh, V=vh, mask=mask)
            outputs.append(attn_vh)
        outputs = torch.cat(outputs, dim=-1)  # Concatenate on the last dimension
        return self.output_proj(outputs)  # Pass through the output linear layer


def calc_attn(Q, K, V, mask):
    _, _, model_dim = Q.size()
    #    QKT = torch.einsum('bsd,bqd -> bsq', Q, K)
    norm = model_dim**0.5
    scores = torch.matmul(Q, K.transpose(2, 1)) / norm
    scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)  # B x S x S
    return torch.matmul(weights, V)


if __name__ == "__main__":
    # Example usage
    model_dim = 64
    num_heads = 8
    seq_len = 10
    batch_size = 2

    rope_encoder = RotationalPositionalEncoding(
        freq_base=10000, seq_len=seq_len, model_dim=model_dim
    )
    attn_layer = MultiHeadAttnWithRoPE(
        model_dim=model_dim, num_heads=num_heads, rope_encoder=rope_encoder
    )

    X = torch.randn(batch_size, seq_len, model_dim)
    output = attn_layer(X)
    print(output.shape)  # Should be [batch_size, seq_len, model_dim]
    # Check if the output shape is correct

import torch
import torch.nn as nn


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "model dim must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, X):
        _, S, _ = X.size()
        q = self.Q(X)  # B x S x D
        k = self.K(X)
        v = self.V(X)

        mask = torch.tril(torch.ones(S, S, device=X.device)).unsqueeze(0)  # [1, S, S]

        outputs = []
        for head in range(self.n_heads):
            d1 = head * self.d_head
            d2 = d1 + self.d_head
            qh = q[:, :, d1:d2]
            kh = k[:, :, d1:d2]
            vh = v[:, :, d1:d2]
            attn_vh = calc_attn(Q=qh, K=kh, V=vh, mask=mask, dim=self.d_head)
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

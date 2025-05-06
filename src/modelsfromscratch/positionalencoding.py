import numpy as np
import torch
import torch.nn as nn


def get_pos_encoding_frequencies(wave_dim: int, model_dim: int) -> torch.Tensor:
    exponents = torch.arange(0, model_dim, 2) / model_dim
    return 1.0 / (wave_dim**exponents)


class RotationalPositionalEncoding(nn.Module):
    def __init__(self, wave_dim: int, seq_len: int, model_dim: int, device: str = "cpu", dbg=False):
        super(RotationalPositionalEncoding, self).__init__()
        self.wave_dim = wave_dim
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.dbg = dbg

        freqs = get_pos_encoding_frequencies(self.wave_dim, self.model_dim).to(device)  # [dim/2]
        pos = torch.arange(seq_len, device=device).float()  # [seq_len]

        angles = torch.einsum('i,j->ij', pos, freqs)  # [seq_len, dim/2]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)
        if self.dbg:
            print("RotationalPositionalEncoding initialized with:")
            print(f"wave_dim: {self.wave_dim}, seq_len: {self.seq_len}, model_dim: {self.model_dim}")
            print(f"sin: {self.sin}")
            print(f"cos: {self.cos}")
            print(f"freqs: {freqs}")
            print(f"angles: {angles}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = x.shape

        sin = self.sin[None, :, :].repeat(batch_size, 1, 1)  # [B, S, D/2]
        cos = self.cos[None, :, :].repeat(batch_size, 1, 1)  # [B, S, D/2]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        if self.dbg:
            print("RotationalPositionalEncoding forward pass:")
            print(f"x: {x}")
            print(f"x1: {x1}")
            print(f"x2: {x2}")
            print(f"rotated: {rotated}")
            print(f"sin: {sin}")
            print(f"cos: {cos}")
        
#        import IPython
#        IPython.embed()
#        1/0

        return rotated

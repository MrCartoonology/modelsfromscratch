import numpy as np
import torch
import torch.nn as nn


def get_positional_encoding(seq_len, emb_dim, wave_dim):
    pos = np.arange(seq_len)[:, np.newaxis]  # (slen, 1)
    i = np.arange(emb_dim)[np.newaxis, :]  # (1, emb_dim)
    pow = (2 * i) / emb_dim
    angle_rates = 1 / (wave_dim**pow)
    angle_radians = pos * angle_rates

    encoding = np.zeros((seq_len, emb_dim))
    encoding[:, 0::2] = np.sin(angle_radians)[:, 0::2]
    encoding[:, 1::2] = np.cos(angle_radians)[:, 1::2]
    return torch.tensor(encoding, dtype=torch.float32)


class RotationalPositionalEncoding(nn.Module):
    def __init__(self, seq_len, embedding_dim, wave_dim):
        super(RotationalPositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.wave_dim = wave_dim
        self.register_buffer("pe", get_positional_encoding(seq_len, embedding_dim, wave_dim))

    def forward(self, x):
        raise NotImplementedError("Rotational positional encoding not implemented yet.")


class AddRotationalPositionalEncoding(RotationalPositionalEncoding):
    def __init__(self, seq_len, embedding_dim, wave_dim):
        super(AddRotationalPositionalEncoding, self).__init__(seq_len, embedding_dim, wave_dim)

    def forward(self, x):
        return x + self.pe[: x.size(1), :]


class MulRotationalPositionalEncoding(RotationalPositionalEncoding):
    def __init__(self, seq_len, embedding_dim, wave_dim):
        super(MulRotationalPositionalEncoding, self).__init__(seq_len, embedding_dim, wave_dim)

    def forward(self, x):
        xr = x[:, 0::2, :]
        xi = x[:, 1::2, :]
        cos_th = self.pe[: x.size(1), 0::2]
        sin_th = self.pe[: x.size(1), 1::2]

        rot_r = xr * cos_th - xi * sin_th
        rot_i = xr * sin_th + xi * cos_th

        x_rot = torch.empty_like(x)
        
        x_rot[:, 0::2, :] = rot_r
        x_rot[:, 1::2, :] = rot_i

        return x_rot



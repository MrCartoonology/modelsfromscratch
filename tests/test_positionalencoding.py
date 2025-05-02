import torch
import numpy as np
import pytest

import modelsfromscratch.positionalencoding as pe


def test_mul_rotational_positional_encoding():
    seq_len = 4
    emb_dim = 4  # must be even for real/imag split
    wave_dim = 10.0

    # Create a small input tensor: batch_size=1, seq_len=4, emb_dim=4
    # We'll set real parts in 0::2 and imaginary parts in 1::2
    x = torch.zeros(1, seq_len, emb_dim)
    for i in range(seq_len):
        x[0, i, 0] = i     # real part for dim 0
        x[0, i, 1] = 0     # imag part for dim 0
        x[0, i, 2] = 0     # real part for dim 1
        x[0, i, 3] = i     # imag part for dim 1

    # Instantiate the encoding module
    mpe = pe.MulRotationalPositionalEncoding(seq_len, emb_dim, wave_dim)
    x_rot = mpe(x)

    # Compute expected rotated result manually
    pe = mpe.pe.detach().numpy()
    expected = torch.zeros_like(x)

    for i in range(seq_len):
        # Real part at 0::2, Imag part at 1::2
        for d in range(0, emb_dim, 2):
            xr = x[0, i, d].item()
            xi = x[0, i, d + 1].item()
            cos_th = pe[i, d]
            sin_th = pe[i, d + 1]

            rot_r = xr * cos_th - xi * sin_th
            rot_i = xr * sin_th + xi * cos_th

            expected[0, i, d] = rot_r
            expected[0, i, d + 1] = rot_i

    assert torch.allclose(x_rot, expected, atol=1e-5), "Rotation mismatch"
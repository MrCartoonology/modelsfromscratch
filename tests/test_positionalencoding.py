import math
import torch
import numpy as np


from modelsfromscratch.positionalencoding import (
    get_pos_encoding_frequencies,
    rotate_pairs,
    RotationalPositionalEncoding
)

 
def test_get_pos_encoding_frequencies():
    wave_dim = 10.0
    model_dim = 6

    twoi = np.array([0.0, 2.0, 4.0])
    twoi_ovr_d = twoi / model_dim
    rates = wave_dim ** twoi_ovr_d
    factors = 1.0 / rates
    expected = torch.tensor(factors, dtype=torch.float32)
    
    predicted = get_pos_encoding_frequencies(freq_base=wave_dim, model_dim=model_dim)
    assert torch.allclose(predicted, expected), f"Frequencies mismatch:\nExpected {expected}\nGot {predicted}"


def test_rotate_pairs():
    A = torch.tensor(
        [  # b=0
            [  # s=0
                [1.0, 0.0, 1.0, 1.0],   # b=0, s=0, d = 0..3
                [0.0, 1.0, -1.0, -1.0]  # b=0, s=1, d=0..3
            ],
            [  # b=1
                [-1.0, 0.0, -1.0, -1.0],  # b=1, s=0, d=0..3
                [0.0, -1.0, 1.0, 1.0]     # b=1, s=1, d=0..3
            ]
        ], dtype=torch.float32)
    theta = torch.tensor(
        [  # s = 0
            [torch.pi / 2, torch.pi / 4],  # s=0, d=0..1
            [torch.pi, torch.pi / 2]       # s=1, d=0..1
        ], dtype=torch.float32)

    expected = torch.tensor(
        [
            [
                [0.0, 1.0, 0.0, math.sqrt(2)],
                [0.0, -1.0, 1.0, -1.0]
            ],
            [
                [0.0, -1.0, 0.0, -math.sqrt(2)],
                [0.0, 1.0, -1.0, 1.0]
            ]
        ], dtype=torch.float32)
    predicted = rotate_pairs(A, theta)
    assert torch.allclose(predicted, expected, rtol=1.e-3, atol=1.e-5), f"Rotation mismatch:\nExpected {expected}\nGot {predicted}"

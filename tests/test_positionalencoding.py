import torch
import numpy as np


from modelsfromscratch.positionalencoding import (
    get_pos_encoding_frequencies,
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
    
    predicted = get_pos_encoding_frequencies(wave_dim=wave_dim, model_dim=model_dim)
    assert torch.allclose(predicted, expected), f"Frequencies mismatch:\nExpected {expected}\nGot {predicted}"


def test_rotational_pe():
    wave_dim = 10.0
    model_dim = 4
    seq_len = 4
    batch_size = 1

    X = torch.randn(batch_size, seq_len, model_dim)
    freq = get_pos_encoding_frequencies(wave_dim=wave_dim, model_dim=model_dim)
    print("test freq: ", freq)
    expected = torch.zeros_like(X)
    print("test X: ", X)

    for b in range(batch_size):
        for s in range(seq_len):
            for d in range(0, model_dim, 2):
                start = d
                end = d + 2
                v = X[b, s, start:end]
                theta = freq[d // 2] * s
                print("test b,s,d, v, theta: ", b, s, d, v, theta)
                rotation_matrix = torch.tensor([
                    [torch.cos(theta), -torch.sin(theta)],
                    [torch.sin(theta), torch.cos(theta)]
                ], dtype=torch.float32)
                rotated = torch.matmul(rotation_matrix, v)
                expected[b, s, start:end] = rotated[:]

    predicted = RotationalPositionalEncoding(wave_dim=wave_dim, seq_len=seq_len, model_dim=model_dim, dbg=True)(X)
    torch.testing.assert_close(predicted, expected, rtol=1e-5, atol=1e-8)

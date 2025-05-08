import torch
import torch.nn as nn


def get_pos_encoding_frequencies(freq_base: int, model_dim: int) -> torch.Tensor:
    """_summary_

    Args:
        freq_base (int): base for frequency scaling of angle
        model_dim (int): dimension of the model

    Returns:
        torch.Tensor: of length model_dim / 2 with frequency scales for positional encoding:
                    1.0 / (freq_base ** (2i / model_dim))
    """
    exponents = torch.arange(0, model_dim, 2) / model_dim
    return 1.0 / (freq_base**exponents)


def rotate_pairs(A: torch.tensor, theta: torch.tensor) -> torch.tensor:
    """
    Rotate pairs of elements in a 3D tensor A by angles specified in theta.
    The tensor A is expected to have a shape of (b, s, d), where:
    - b: batch size
    - s: sequence length
    - d: model dimension (must be even).

    The tensor theta is expected to have a shape of (b, s, d/2), where:
    - b: batch size
    - s: sequence length
    - d/2: half the model dimension (for pairs of elements).

    Each pair of elements in the last dimension (d) of A is rotated by the corresponding angle in theta.
    """
    A1 = A[..., 0::2]
    A2 = A[..., 1::2]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    A_rotated = torch.empty_like(A)
    A_rotated[..., 0::2] = A1 * cos_theta - A2 * sin_theta
    A_rotated[..., 1::2] = A1 * sin_theta + A2 * cos_theta
    return A_rotated


class RotationalPositionalEncoding(nn.Module):
    def __init__(
        self, freq_base: int, seq_len: int, model_dim: int, device: str = "cpu"
    ):
        super(RotationalPositionalEncoding, self).__init__()
        self.freq_base = freq_base
        self.seq_len = seq_len
        self.model_dim = model_dim

        freqs = get_pos_encoding_frequencies(
            freq_base=self.freq_base, model_dim=self.model_dim
        ).to(
            device
        )  # [dim/2]
        pos = torch.arange(seq_len, device=device).float()  # [seq_len]

        angles = torch.einsum("i,j->ij", pos, freqs)  # [seq_len, dim/2]
        self.register_buffer("angles", angles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = x.size()
        return rotate_pairs(x, self.angles[:seq_len, :])

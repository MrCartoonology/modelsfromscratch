import torch

from modelsfromscratch.multiheadattn import calc_attn


def test_attn():
    batch_size = 4
    seq_len = 6
    model_dim = 3

    Q = torch.randn(batch_size, seq_len, model_dim)
    K = torch.randn(batch_size, seq_len, model_dim)
    V = torch.randn(batch_size, seq_len, model_dim)

    expected_output = for_loop_attn(Q=Q, K=K, V=V)

    mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device)).unsqueeze(
        0
    )  # [1, S, S]
    predicted_output = calc_attn(Q=Q, K=K, V=V, mask=mask)

    torch.testing.assert_close(predicted_output, expected_output, rtol=1e-5, atol=1e-8)


def for_loop_attn(Q, K, V):
    batch_size, seq_len, model_dim = Q.size()
    scores = torch.zeros(batch_size, seq_len, seq_len)

    for bi in range(batch_size):
        for si in range(seq_len):
            for sj in range(seq_len):
                for dk in range(model_dim):
                    scores[bi, si, sj] += Q[bi, si, dk] * K[bi, sj, dk]
    scores /= torch.sqrt(torch.tensor(model_dim, dtype=torch.float32))

    # mask/softmax
    attn = torch.zeros(batch_size, seq_len, seq_len)
    for bi in range(batch_size):
        for si in range(seq_len):
            sip1 = si + 1
            attn[bi, si, 0:sip1] = torch.softmax(scores[bi, si, 0:sip1], dim=0)

    output = torch.zeros_like(V)
    for bi in range(batch_size):
        for si in range(seq_len):
            for dk in range(model_dim):
                for sj in range(seq_len):
                    output[bi, si, dk] += attn[bi, si, sj] * V[bi, sj, dk]

    return output

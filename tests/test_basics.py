import torch


def test_tensor_matrix_multiplication():
    # Create random tensors
    tensor_a = torch.rand(2, 3, 4)  # Shape: (2, 3, 4)
    tensor_b = torch.rand(4, 5)  # Shape: (4, 5)

    # Perform matrix multiplication using torch
    torch_result = torch.matmul(tensor_a, tensor_b)

    assert torch_result.shape == (2, 3, 5), "Result shape mismatch!"
    # Perform matrix multiplication using a for loop
    loop_result = torch.zeros_like(torch_result)
    for i in range(tensor_a.shape[0]):
        for j in range(tensor_a.shape[1]):
            for k in range(5):
                res = 0
                for z in range(4):
                    res += tensor_a[i, j, z] * tensor_b[z, k]
                loop_result[i, j, k] = res

    # Compare the results
    assert torch.allclose(torch_result, loop_result), "Results do not match!"


if __name__ == "__main__":
    test_tensor_matrix_multiplication()
    print("Test passed!")

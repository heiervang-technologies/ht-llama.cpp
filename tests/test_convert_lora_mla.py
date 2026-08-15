import pytest
import torch

from conversion import LazyTorchTensor
from convert_lora_to_gguf import split_mla_kv_b_lora


@pytest.mark.parametrize("lazy", [False, True])
def test_split_mla_kv_b_lora_reconstructs_transformed_delta(lazy: bool):
    n_head_kv, qk_dim, v_dim, rank, n_input = 2, 3, 2, 4, 5
    lora_a = torch.arange(rank * n_input, dtype=torch.float32).view(rank, n_input) / 7
    lora_b = torch.arange(
        n_head_kv * (qk_dim + v_dim) * rank,
        dtype=torch.float32,
    ).view(n_head_kv * (qk_dim + v_dim), rank) / 11

    input_a, input_b = lora_a, lora_b
    if lazy:
        input_a = LazyTorchTensor.from_eager(input_a)
        input_b = LazyTorchTensor.from_eager(input_b)

    k_a, k_b, v_a, v_b = split_mla_kv_b_lora(
        input_a, input_b, n_head_kv, qk_dim, v_dim
    )
    if lazy:
        k_a, k_b, v_a, v_b = LazyTorchTensor.to_eager((k_a, k_b, v_a, v_b))

    fused_delta = (lora_b @ lora_a).view(n_head_kv, qk_dim + v_dim, n_input)
    expected_k, expected_v = torch.split(fused_delta, [qk_dim, v_dim], dim=1)
    expected_k = expected_k.transpose(1, 2)

    assert k_a.shape == (n_head_kv, rank, qk_dim)
    assert k_b.shape == (1, n_input, rank)
    assert v_a.shape == (1, rank, n_input)
    assert v_b.shape == (n_head_kv, v_dim, rank)
    torch.testing.assert_close(k_b @ k_a, expected_k)
    torch.testing.assert_close(v_b @ v_a, expected_v)


def test_split_mla_kv_b_lora_rejects_bad_shape():
    with pytest.raises(ValueError, match="expected first dimension"):
        split_mla_kv_b_lora(
            torch.zeros(2, 4),
            torch.zeros(9, 2),
            n_head_kv=2,
            qk_nope_head_dim=3,
            v_head_dim=2,
        )

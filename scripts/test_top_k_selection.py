from __future__ import annotations

import torch

from top_k_selection import (
    get_cu_block_seqlens,
    top_k_middle_block_mask,
    topk_block_selection_from_scores,
    topkselectionvarlen_reference,
)


def make_cu_seqlens(seqlens: list[int]) -> torch.Tensor:
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    if seqlens:
        cu[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0)
    return cu


def assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(f"{name}: actual={actual.cpu()} expected={expected.cpu()}")


def test_topk_keeps_sink_tail_and_middle_budget() -> None:
    block_size = 64
    cu = make_cu_seqlens([8192])
    cu_b = get_cu_block_seqlens(cu, block_size)
    scores = torch.arange(int(cu_b[-1].item()), dtype=torch.float32)

    block_mask = top_k_middle_block_mask(
        scores,
        cu,
        cu_b,
        block_size=block_size,
        attention_sink=128,
        last_q=128,
        middle_keep_ratio=0.05,
        min_middle_keep_blocks=2,
    )

    # 8192 / 64 = 128 blocks. Force first 2 and last 2 blocks.
    # Middle region has 124 blocks, ceil(124 * 0.05) = 7.
    expected_kept = 2 + 7 + 2
    if int(block_mask.sum().item()) != expected_kept:
        raise AssertionError(f"kept blocks mismatch: {int(block_mask.sum().item())}")

    assert block_mask[0].item()
    assert block_mask[1].item()
    assert block_mask[-1].item()
    assert block_mask[-2].item()

    # Scores are ascending, so the top middle blocks are right before the tail.
    selected_middle = torch.where(block_mask[2:-2])[0] + 2
    assert_equal(
        "selected_middle",
        selected_middle,
        torch.tensor([119, 120, 121, 122, 123, 124, 125]),
    )


def test_block_score_helper_outputs_metadata() -> None:
    block_size = 64
    cu = make_cu_seqlens([8192, 4096])
    cu_b = get_cu_block_seqlens(cu, block_size)
    scores = torch.arange(int(cu_b[-1].item()), dtype=torch.float32)

    block_mask, token_mask, new_cu, new_max = topk_block_selection_from_scores(
        scores,
        cu,
        cu_b,
        block_size=block_size,
        attention_sink=128,
        last_q=128,
        middle_keep_ratio=0.05,
        min_middle_keep_blocks=2,
    )

    assert int(block_mask.sum().item()) == 18
    assert_equal("new_cu", new_cu, torch.tensor([0, 704, 1152], dtype=torch.int32))
    assert int(new_max.item()) == 704
    assert int(token_mask.sum().item()) == 1152


def test_short_prompt_kept_dense_in_qk_reference() -> None:
    seq_len = 256
    head_dim = 8
    num_q_heads = 4
    num_kv_heads = 2
    cu = make_cu_seqlens([seq_len])
    q = torch.randn(seq_len, num_q_heads, head_dim)
    k = torch.randn(seq_len, num_kv_heads, head_dim)

    token_mask, new_max, new_cu, block_mask = topkselectionvarlen_reference(
        q,
        k,
        head_dim=head_dim,
        cu_seqlens=cu,
        max_seq_len=seq_len,
        block_size=64,
        attention_sink=128,
        last_q=128,
        middle_keep_ratio=0.05,
        min_middle_keep_blocks=2,
        drop_threshold_extra_blocks=4,
    )

    assert token_mask.all().item()
    assert new_max == seq_len
    assert_equal("new_cu", new_cu, cu)
    assert block_mask.numel() == 0


def test_qk_reference_runs_for_long_prompt() -> None:
    torch.manual_seed(0)
    seq_len = 1024
    head_dim = 8
    num_q_heads = 4
    num_kv_heads = 2
    cu = make_cu_seqlens([seq_len])
    q = torch.randn(seq_len, num_q_heads, head_dim)
    k = torch.randn(seq_len, num_kv_heads, head_dim)

    token_mask, new_max, new_cu, block_mask = topkselectionvarlen_reference(
        q,
        k,
        head_dim=head_dim,
        cu_seqlens=cu,
        max_seq_len=seq_len,
        block_size=64,
        attention_sink=128,
        last_q=128,
        middle_keep_ratio=0.05,
        min_middle_keep_blocks=2,
        drop_threshold_extra_blocks=4,
    )

    # 16 blocks total, 2 sink + 2 tail + max(ceil(12*0.05), 2) middle = 6 blocks.
    assert int(block_mask.sum().item()) == 6
    assert int(token_mask.sum().item()) == 384
    assert_equal("new_cu", new_cu, torch.tensor([0, 384], dtype=torch.int32))
    assert new_max == 384


def main() -> None:
    tests = [
        test_topk_keeps_sink_tail_and_middle_budget,
        test_block_score_helper_outputs_metadata,
        test_short_prompt_kept_dense_in_qk_reference,
        test_qk_reference_runs_for_long_prompt,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASSED")
    print(f"Total: {len(tests)}, Passed: {len(tests)}, Failed: 0")


if __name__ == "__main__":
    main()

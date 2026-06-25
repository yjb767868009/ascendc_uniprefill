"""Python reference for fixed-budget block top-k compact.

This prototype demonstrates the no-host-sync idea for token dropping:
output shapes are determined by CPU-side budget metadata, not by
``token_mask.sum()`` produced on device.

The future AscendC op should implement the same data movement pattern:
  block_scores + fixed keep_blocks_per_req -> selected blocks -> compact tokens

Run:
  python3 /autodl-fs/data/yjb/ascendc_uniprefill/prototypes/topk_block_compact.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopKCompactBudget:
    """Host-known shape metadata for fixed-budget compact."""

    keep_blocks_per_req: torch.Tensor  # int32 CPU, [batch]
    budget_lens: torch.Tensor  # int32 CPU, [batch]
    budget_cu_seqlens: torch.Tensor  # int32 CPU, [batch + 1]
    max_budget_len: int


@dataclass(frozen=True)
class TopKCompactResult:
    hidden_out: torch.Tensor
    residual_out: torch.Tensor
    positions_out: torch.Tensor
    budget_cu_seqlens: torch.Tensor
    selected_block_mask: torch.Tensor
    valid_token_mask: torch.Tensor


def get_cu_block_seqlens(cu_seqlens: torch.Tensor, block_size: int) -> torch.Tensor:
    """Compute cumulative block lengths from cumulative token lengths."""
    cu_cpu = cu_seqlens.detach().cpu().to(torch.int32)
    seq_lens = cu_cpu[1:] - cu_cpu[:-1]
    block_lens = torch.div(seq_lens + block_size - 1, block_size, rounding_mode="floor")
    cu_blocks = torch.zeros_like(cu_cpu)
    cu_blocks[1:] = torch.cumsum(block_lens, dim=0)
    return cu_blocks


def forced_block_mask_for_request(
    seq_len: int,
    num_blocks: int,
    block_size: int,
    attention_sink: int,
    last_q: int,
) -> torch.Tensor:
    """Return bool mask over blocks that must be kept for sink/tail tokens."""
    forced = torch.zeros(num_blocks, dtype=torch.bool)
    if seq_len <= 0 or num_blocks <= 0:
        return forced

    sink_tokens = min(max(attention_sink, 0), seq_len)
    sink_blocks = (sink_tokens + block_size - 1) // block_size
    if sink_blocks > 0:
        forced[:sink_blocks] = True

    tail_tokens = min(max(last_q, 0), seq_len)
    if tail_tokens > 0:
        tail_start = max(seq_len - tail_tokens, 0)
        tail_block_start = tail_start // block_size
        forced[tail_block_start:] = True

    return forced


def compute_fixed_budget(
    cu_seqlens: torch.Tensor,
    block_size: int,
    keep_ratio: float,
    attention_sink: int,
    last_q: int,
) -> TopKCompactBudget:
    """Compute all output shapes on CPU before any device selection runs.

    This is the no-sync contract: future kernels must write into exactly this
    preallocated capacity. The budget is based only on host-known metadata.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1]")

    cu_cpu = cu_seqlens.detach().cpu().to(torch.int32)
    seq_lens = cu_cpu[1:] - cu_cpu[:-1]
    batch = int(seq_lens.numel())

    keep_blocks = torch.zeros(batch, dtype=torch.int32)
    budget_lens = torch.zeros(batch, dtype=torch.int32)

    for req, seq_len_t in enumerate(seq_lens.tolist()):
        seq_len = int(seq_len_t)
        num_blocks = (seq_len + block_size - 1) // block_size
        ratio_blocks = int(torch.ceil(torch.tensor(num_blocks * keep_ratio)).item())
        forced = forced_block_mask_for_request(
            seq_len, num_blocks, block_size, attention_sink, last_q
        )
        forced_blocks = int(forced.sum().item())
        k = max(ratio_blocks, forced_blocks)
        k = min(k, num_blocks)
        keep_blocks[req] = k
        # Fixed budget uses full block capacity. Tail/padding tokens may exist.
        budget_lens[req] = k * block_size

    budget_cu = torch.zeros(batch + 1, dtype=torch.int32)
    if batch:
        budget_cu[1:] = torch.cumsum(budget_lens, dim=0)
    max_budget_len = int(budget_lens.max().item()) if batch else 0
    return TopKCompactBudget(keep_blocks, budget_lens, budget_cu, max_budget_len)


def select_topk_blocks_for_request(
    scores: torch.Tensor,
    seq_len: int,
    block_size: int,
    keep_blocks: int,
    attention_sink: int,
    last_q: int,
) -> torch.Tensor:
    """Select exactly keep_blocks blocks, with forced sink/tail blocks included."""
    num_blocks = int(scores.numel())
    selected = forced_block_mask_for_request(
        seq_len, num_blocks, block_size, attention_sink, last_q
    ).to(scores.device)

    keep_blocks = min(max(int(keep_blocks), int(selected.sum().item())), num_blocks)
    remaining_k = keep_blocks - int(selected.sum().item())
    if remaining_k <= 0:
        return selected

    candidate_scores = scores.float().clone()
    candidate_scores[selected] = -torch.inf
    topk = torch.topk(candidate_scores, k=remaining_k, largest=True, sorted=False).indices
    selected[topk] = True
    return selected


def topk_block_compact_reference(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    block_scores: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_seqlens: torch.Tensor,
    budget: TopKCompactBudget,
    block_size: int,
    attention_sink: int,
    last_q: int,
) -> TopKCompactResult:
    """Compact selected top-k blocks into fixed-shape outputs.

    Shapes:
      hidden_states: [T, H]
      residual:      [T, H]
      positions:     [T]
      block_scores:  [total_blocks]

    Outputs:
      hidden_out:    [budget_cu_seqlens[-1], H]
      residual_out:  [budget_cu_seqlens[-1], H]
      positions_out: [budget_cu_seqlens[-1]]

    The output first dimension is host-known from ``budget``. It never depends on
    device-computed ``token_mask.sum()``.
    """
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [T, H]")
    if residual.shape != hidden_states.shape:
        raise ValueError("residual must have the same shape as hidden_states")
    if positions.ndim != 1 or positions.shape[0] != hidden_states.shape[0]:
        raise ValueError("positions must be [T]")

    device = hidden_states.device
    cu_seq = cu_seqlens.detach().cpu().to(torch.int32)
    cu_blk = cu_block_seqlens.detach().cpu().to(torch.int32)
    batch = int(cu_seq.numel() - 1)
    total_budget = int(budget.budget_cu_seqlens[-1].item())
    hidden_size = hidden_states.shape[1]

    hidden_out = torch.empty((total_budget, hidden_size), dtype=hidden_states.dtype, device=device)
    residual_out = torch.empty((total_budget, hidden_size), dtype=residual.dtype, device=device)
    positions_out = torch.empty((total_budget,), dtype=positions.dtype, device=device)
    valid_token_mask = torch.zeros((total_budget,), dtype=torch.bool, device=device)
    selected_block_mask = torch.zeros_like(block_scores, dtype=torch.bool, device=device)

    for req in range(batch):
        src_token_start = int(cu_seq[req].item())
        src_token_end = int(cu_seq[req + 1].item())
        seq_len = src_token_end - src_token_start
        src_block_start = int(cu_blk[req].item())
        src_block_end = int(cu_blk[req + 1].item())
        dst_start = int(budget.budget_cu_seqlens[req].item())
        keep_blocks = int(budget.keep_blocks_per_req[req].item())

        scores = block_scores[src_block_start:src_block_end]
        selected = select_topk_blocks_for_request(
            scores, seq_len, block_size, keep_blocks, attention_sink, last_q
        )
        selected_block_mask[src_block_start:src_block_end] = selected

        write = dst_start
        for block_offset in range(int(selected.numel())):
            if not bool(selected[block_offset].item()):
                continue
            src_begin = src_token_start + block_offset * block_size
            src_end = min(src_begin + block_size, src_token_end)
            real_len = src_end - src_begin
            if real_len <= 0:
                continue

            # Copy real tokens.
            hidden_out[write : write + real_len] = hidden_states[src_begin:src_end]
            residual_out[write : write + real_len] = residual[src_begin:src_end]
            positions_out[write : write + real_len] = positions[src_begin:src_end]
            valid_token_mask[write : write + real_len] = True

            # Pad tail block to full block_size so the request output length is fixed.
            pad_len = block_size - real_len
            if pad_len > 0:
                pad_start = write + real_len
                pad_end = write + block_size
                hidden_out[pad_start:pad_end] = 0
                residual_out[pad_start:pad_end] = 0
                positions_out[pad_start:pad_end] = 0

            write += block_size

        expected_end = dst_start + int(budget.budget_lens[req].item())
        if write != expected_end:
            raise RuntimeError(
                f"request {req} wrote {write - dst_start} tokens, "
                f"expected budget {expected_end - dst_start}"
            )

    return TopKCompactResult(
        hidden_out=hidden_out,
        residual_out=residual_out,
        positions_out=positions_out,
        budget_cu_seqlens=budget.budget_cu_seqlens.to(device),
        selected_block_mask=selected_block_mask,
        valid_token_mask=valid_token_mask,
    )


def demo() -> None:
    torch.manual_seed(0)

    # Example batch:
    #   req0: 10 tokens -> 3 blocks when block_size=4
    #   req1: 14 tokens -> 4 blocks when block_size=4
    seq_lens = torch.tensor([10, 14], dtype=torch.int32)
    cu_seqlens = torch.zeros(3, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
    block_size = 4
    keep_ratio = 0.5
    attention_sink = 2
    last_q = 3
    hidden_size = 3

    cu_block_seqlens = get_cu_block_seqlens(cu_seqlens, block_size)
    total_tokens = int(cu_seqlens[-1].item())
    total_blocks = int(cu_block_seqlens[-1].item())

    # Make contents easy to inspect: each row stores token id repeated.
    token_ids = torch.arange(total_tokens, dtype=torch.float32)
    hidden_states = token_ids[:, None].repeat(1, hidden_size)
    residual = hidden_states + 1000
    positions = torch.arange(total_tokens, dtype=torch.int64)

    # Higher score means more important block.
    block_scores = torch.tensor([0.1, 0.9, 0.2, 0.5, 0.8, 0.3, 0.7], dtype=torch.float32)
    assert block_scores.numel() == total_blocks

    budget = compute_fixed_budget(
        cu_seqlens, block_size, keep_ratio, attention_sink, last_q
    )
    result = topk_block_compact_reference(
        hidden_states,
        residual,
        positions,
        block_scores,
        cu_seqlens,
        cu_block_seqlens,
        budget,
        block_size,
        attention_sink,
        last_q,
    )

    print("Input shapes:")
    print(f"  hidden_states: {tuple(hidden_states.shape)}")
    print(f"  residual:      {tuple(residual.shape)}")
    print(f"  positions:     {tuple(positions.shape)}")
    print(f"  block_scores:  {tuple(block_scores.shape)}")
    print("Host-known budget:")
    print(f"  cu_seqlens:           {cu_seqlens.tolist()}")
    print(f"  cu_block_seqlens:     {cu_block_seqlens.tolist()}")
    print(f"  keep_blocks_per_req:  {budget.keep_blocks_per_req.tolist()}")
    print(f"  budget_lens:          {budget.budget_lens.tolist()}")
    print(f"  budget_cu_seqlens:    {budget.budget_cu_seqlens.tolist()}")
    print("Output shapes:")
    print(f"  hidden_out:     {tuple(result.hidden_out.shape)}")
    print(f"  residual_out:   {tuple(result.residual_out.shape)}")
    print(f"  positions_out:  {tuple(result.positions_out.shape)}")
    print("Selection:")
    print(f"  selected_blocks: {result.selected_block_mask.tolist()}")
    print(f"  valid_token_mask:{result.valid_token_mask.tolist()}")
    print("Compacted token ids, padding shows as 0:")
    print(result.hidden_out[:, 0].to(torch.int64).tolist())


if __name__ == "__main__":
    demo()

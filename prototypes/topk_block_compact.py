"""Python reference for fixed 5% bottom-k middle-block drop with real compact.

Answer to the earlier question first:
this version does not use "top-p + padding budget" any more. Instead, for each
request we:

1. force-keep sink blocks,
2. force-keep tail blocks,
3. drop a fixed ratio of the remaining middle blocks,
4. compact the surviving tokens into a real dense output with no padding.

The key point is that the output shape is still host-known before launch:
- sink/tail token counts are known from seq_len/block_size/attention_sink/last_q
- dropped middle block count is a fixed host policy
- every middle block has length exactly block_size

So we avoid dynamic ``token_mask.sum()`` while also avoiding fake padding tokens
entering FA / proposer / slot_mapping.

Important reference-code boundary:
Python control flow in this file is CPU-only. If a tensor is produced on an
accelerator, the production AscendC kernel must branch on it inside the kernel;
Python must not use ``.item()`` / ``.cpu()`` to inspect it.

Answer to the latest question:
the right policy is fixed 5% drop on middle blocks only. Sink/tail blocks are
always kept, and the host precomputes the exact real output length from the
number of kept complete blocks plus the exact sink/tail token lengths.

Run:
  python3 /autodl-fs/data/yjb/ascendc_uniprefill/prototypes/topk_block_compact.py
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RealCompactPlan:
    """Host-known compact plan.

    All fields are computable on CPU from sequence metadata and the fixed drop
    policy. Future AscendC kernels should consume the same metadata.
    """

    drop_middle_blocks_per_req: torch.Tensor  # int32 CPU, [batch]
    keep_middle_blocks_per_req: torch.Tensor  # int32 CPU, [batch]
    sink_token_lens: torch.Tensor  # int32 CPU, [batch]
    middle_token_lens: torch.Tensor  # int32 CPU, [batch]
    tail_token_lens: torch.Tensor  # int32 CPU, [batch]
    real_lens: torch.Tensor  # int32 CPU, [batch]
    real_cu_seqlens: torch.Tensor  # int32 CPU, [batch + 1]
    max_real_len: int


@dataclass(frozen=True)
class RealCompactResult:
    hidden_out: torch.Tensor
    residual_out: torch.Tensor
    positions_out: torch.Tensor
    slot_mapping_out: torch.Tensor
    real_cu_seqlens: torch.Tensor
    kept_block_mask: torch.Tensor


def _as_cpu_int32_1d(values: torch.Tensor | list[int] | tuple[int, ...], name: str) -> torch.Tensor:
    """Normalize host metadata without ever copying from an accelerator."""
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(
                f"{name} must be CPU/host metadata. Do not pass an accelerator "
                "tensor here, because .cpu() would synchronize the stream."
            )
        out = values.to(dtype=torch.int32)
    else:
        out = torch.tensor(values, dtype=torch.int32)

    if out.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor/list")
    return out.contiguous()


def get_cu_seqlens_from_seq_lens(seq_lens: torch.Tensor | list[int] | tuple[int, ...]) -> torch.Tensor:
    """Build CPU cu_seqlens from CPU request lengths."""
    seq_lens_cpu = _as_cpu_int32_1d(seq_lens, "seq_lens")
    cu = torch.zeros(seq_lens_cpu.numel() + 1, dtype=torch.int32)
    if seq_lens_cpu.numel():
        cu[1:] = torch.cumsum(seq_lens_cpu, dim=0)
    return cu


def get_cu_block_seqlens_from_seq_lens(
    seq_lens: torch.Tensor | list[int] | tuple[int, ...],
    block_size: int,
) -> torch.Tensor:
    """Build CPU cumulative block lengths from CPU request lengths."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    seq_lens_cpu = _as_cpu_int32_1d(seq_lens, "seq_lens")
    block_lens = torch.div(seq_lens_cpu + block_size - 1, block_size, rounding_mode="floor")
    cu_blocks = torch.zeros(seq_lens_cpu.numel() + 1, dtype=torch.int32)
    if seq_lens_cpu.numel():
        cu_blocks[1:] = torch.cumsum(block_lens, dim=0)
    return cu_blocks


def get_cu_block_seqlens(cu_seqlens: torch.Tensor | list[int] | tuple[int, ...], block_size: int) -> torch.Tensor:
    """CPU-only compatibility wrapper. Prefer get_cu_block_seqlens_from_seq_lens."""
    cu_cpu = _as_cpu_int32_1d(cu_seqlens, "cu_seqlens")
    if cu_cpu.numel() == 0:
        raise ValueError("cu_seqlens must contain at least one element")
    seq_lens = cu_cpu[1:] - cu_cpu[:-1]
    return get_cu_block_seqlens_from_seq_lens(seq_lens, block_size)


def classify_blocks(
    seq_len: int,
    block_size: int,
    attention_sink: int,
    last_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sink, middle, tail block masks for one request.

    The three masks are disjoint and cover all blocks.
    """
    num_blocks = (seq_len + block_size - 1) // block_size
    sink = torch.zeros(num_blocks, dtype=torch.bool)
    tail = torch.zeros(num_blocks, dtype=torch.bool)
    middle = torch.zeros(num_blocks, dtype=torch.bool)

    if num_blocks == 0:
        return sink, middle, tail

    sink_tokens = min(max(attention_sink, 0), seq_len)
    if sink_tokens > 0:
        sink_blocks = (sink_tokens + block_size - 1) // block_size
        sink[:sink_blocks] = True

    tail_tokens = min(max(last_q, 0), seq_len)
    if tail_tokens > 0:
        tail_start = max(seq_len - tail_tokens, 0)
        tail_block_start = tail_start // block_size
        tail[tail_block_start:] = True

    middle = ~(sink | tail)
    return sink, middle, tail


def block_real_token_length(
    block_idx: int,
    seq_len: int,
    block_size: int,
) -> int:
    start = block_idx * block_size
    end = min(start + block_size, seq_len)
    return max(end - start, 0)


def count_mask_real_tokens(mask: torch.Tensor, seq_len: int, block_size: int) -> int:
    total = 0
    for block_idx, keep in enumerate(mask.tolist()):
        if keep:
            total += block_real_token_length(block_idx, seq_len, block_size)
    return total


def compute_real_compact_plan_from_seq_lens(
    seq_lens: torch.Tensor | list[int] | tuple[int, ...],
    block_size: int,
    drop_ratio: float,
    attention_sink: int,
    last_q: int,
) -> RealCompactPlan:
    """Compute exact real output lengths from CPU request lengths.

    This is the no-sync production contract: the planner consumes host-known
    sequence metadata, not device cu_seqlens. Passing an accelerator tensor is
    rejected instead of silently doing ``.cpu()`` and draining the queue.

    Policy:
    - sink blocks are always kept
    - tail blocks are always kept
    - among middle blocks, drop floor(num_middle_blocks * drop_ratio)
    - surviving middle blocks are complete blocks, so each contributes block_size
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not (0.0 <= drop_ratio < 1.0):
        raise ValueError("drop_ratio must be in [0, 1)")

    seq_lens = _as_cpu_int32_1d(seq_lens, "seq_lens")
    batch = int(seq_lens.numel())

    drop_middle = torch.zeros(batch, dtype=torch.int32)
    keep_middle = torch.zeros(batch, dtype=torch.int32)
    sink_lens = torch.zeros(batch, dtype=torch.int32)
    middle_lens = torch.zeros(batch, dtype=torch.int32)
    tail_lens = torch.zeros(batch, dtype=torch.int32)
    real_lens = torch.zeros(batch, dtype=torch.int32)

    for req, seq_len_t in enumerate(seq_lens.tolist()):
        seq_len = int(seq_len_t)
        sink, middle, tail = classify_blocks(seq_len, block_size, attention_sink, last_q)
        num_middle = middle.tolist().count(True)
        drop_k = int(num_middle * drop_ratio)  # floor by Python int conversion
        keep_k = num_middle - drop_k

        sink_len = count_mask_real_tokens(sink, seq_len, block_size)
        tail_len = count_mask_real_tokens(tail, seq_len, block_size)
        middle_len = keep_k * block_size

        drop_middle[req] = drop_k
        keep_middle[req] = keep_k
        sink_lens[req] = sink_len
        middle_lens[req] = middle_len
        tail_lens[req] = tail_len
        real_lens[req] = sink_len + middle_len + tail_len

    real_cu = torch.zeros(batch + 1, dtype=torch.int32)
    if batch:
        real_cu[1:] = torch.cumsum(real_lens, dim=0)

    max_real_len = max(real_lens.tolist(), default=0)
    return RealCompactPlan(
        drop_middle_blocks_per_req=drop_middle,
        keep_middle_blocks_per_req=keep_middle,
        sink_token_lens=sink_lens,
        middle_token_lens=middle_lens,
        tail_token_lens=tail_lens,
        real_lens=real_lens,
        real_cu_seqlens=real_cu,
        max_real_len=max_real_len,
    )


def compute_real_compact_plan_from_cu_seqlens(
    cu_seqlens: torch.Tensor | list[int] | tuple[int, ...],
    block_size: int,
    drop_ratio: float,
    attention_sink: int,
    last_q: int,
) -> RealCompactPlan:
    """CPU-only compatibility wrapper around compute_real_compact_plan_from_seq_lens."""
    cu_cpu = _as_cpu_int32_1d(cu_seqlens, "cu_seqlens")
    if cu_cpu.numel() == 0:
        raise ValueError("cu_seqlens must contain at least one element")
    seq_lens = cu_cpu[1:] - cu_cpu[:-1]
    return compute_real_compact_plan_from_seq_lens(
        seq_lens, block_size, drop_ratio, attention_sink, last_q
    )


# Backward-compatible name, but still CPU-only.
compute_real_compact_plan = compute_real_compact_plan_from_cu_seqlens


def select_kept_blocks_for_request_cpu(
    scores: torch.Tensor,
    seq_len: int,
    block_size: int,
    keep_middle_blocks: int,
    attention_sink: int,
    last_q: int,
) -> torch.Tensor:
    """CPU reference: keep sink+tail blocks and top-k middle blocks by score.

    This helper intentionally rejects accelerator tensors. The real no-sync path
    must do this selection inside the AscendC kernel, not by reading device
    booleans in Python.
    """
    if scores.device.type != "cpu":
        raise ValueError(
            "scores must be CPU for the Python reference. Production AscendC "
            "must select kept blocks on device without Python .item()/.cpu()."
        )

    sink, middle, tail = classify_blocks(seq_len, block_size, attention_sink, last_q)
    kept = sink | tail

    middle_indices = torch.nonzero(middle, as_tuple=False).flatten()
    if middle_indices.numel() == 0 or keep_middle_blocks <= 0:
        return kept

    keep_middle_blocks = min(int(keep_middle_blocks), int(middle_indices.numel()))
    middle_scores = scores[middle_indices].float()
    topk_local = torch.topk(middle_scores, k=keep_middle_blocks, largest=True, sorted=False).indices
    kept[middle_indices[topk_local]] = True
    return kept


# Backward-compatible name for old prototype users. Still CPU-only by design.
select_kept_blocks_for_request = select_kept_blocks_for_request_cpu


def topk_block_compact_reference(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_scores: torch.Tensor,
    seq_lens: torch.Tensor | list[int] | tuple[int, ...],
    cu_block_seqlens: torch.Tensor | list[int] | tuple[int, ...],
    plan: RealCompactPlan,
    block_size: int,
    attention_sink: int,
    last_q: int,
) -> RealCompactResult:
    """Compact kept blocks into a real dense output with no padding tokens.

    Shapes:
      hidden_states: [T, H]
      residual:      [T, H]
      positions:     [T]
      slot_mapping:  [T]
      block_scores:  [total_blocks]
      seq_lens:      CPU [batch], host-known request lengths

    Outputs:
      hidden_out:       [real_cu_seqlens[-1], H]
      residual_out:     [real_cu_seqlens[-1], H]
      positions_out:    [real_cu_seqlens[-1]]
      slot_mapping_out: [real_cu_seqlens[-1]]

    The first dimension is host-known from ``plan``. No device-side token count
    is needed to allocate it. ``block_scores`` is CPU-only in this Python
    reference so the loop below never inspects an accelerator boolean.
    """
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [T, H]")
    if residual.shape != hidden_states.shape:
        raise ValueError("residual must have the same shape as hidden_states")
    if positions.ndim != 1 or positions.shape[0] != hidden_states.shape[0]:
        raise ValueError("positions must be [T]")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != hidden_states.shape[0]:
        raise ValueError("slot_mapping must be [T]")

    device = hidden_states.device
    seq_lens_cpu = _as_cpu_int32_1d(seq_lens, "seq_lens")
    batch = int(seq_lens_cpu.numel())
    cu_seq_list = get_cu_seqlens_from_seq_lens(seq_lens_cpu).tolist()
    cu_blk_list = _as_cpu_int32_1d(cu_block_seqlens, "cu_block_seqlens").tolist()
    real_cu_list = plan.real_cu_seqlens.tolist()
    keep_middle_list = plan.keep_middle_blocks_per_req.tolist()
    total_real = real_cu_list[-1]
    hidden_size = hidden_states.shape[1]

    hidden_out = torch.empty((total_real, hidden_size), dtype=hidden_states.dtype, device=device)
    residual_out = torch.empty((total_real, hidden_size), dtype=residual.dtype, device=device)
    positions_out = torch.empty((total_real,), dtype=positions.dtype, device=device)
    slot_mapping_out = torch.empty((total_real,), dtype=slot_mapping.dtype, device=device)
    kept_block_mask_cpu = torch.zeros(block_scores.numel(), dtype=torch.bool)

    for req in range(batch):
        src_token_start = cu_seq_list[req]
        src_token_end = cu_seq_list[req + 1]
        seq_len = src_token_end - src_token_start
        src_block_start = cu_blk_list[req]
        src_block_end = cu_blk_list[req + 1]
        dst_start = real_cu_list[req]
        expected_end = real_cu_list[req + 1]
        keep_middle = keep_middle_list[req]

        scores = block_scores[src_block_start:src_block_end]
        kept_cpu = select_kept_blocks_for_request_cpu(
            scores, seq_len, block_size, keep_middle, attention_sink, last_q
        )
        kept_block_mask_cpu[src_block_start:src_block_end] = kept_cpu

        write = dst_start
        for block_offset, keep in enumerate(kept_cpu.tolist()):
            if not keep:
                continue
            src_begin = src_token_start + block_offset * block_size
            src_end = min(src_begin + block_size, src_token_end)
            real_len = src_end - src_begin
            if real_len <= 0:
                continue

            hidden_out[write : write + real_len] = hidden_states[src_begin:src_end]
            residual_out[write : write + real_len] = residual[src_begin:src_end]
            positions_out[write : write + real_len] = positions[src_begin:src_end]
            slot_mapping_out[write : write + real_len] = slot_mapping[src_begin:src_end]
            write += real_len

        if write != expected_end:
            raise RuntimeError(
                f"request {req} wrote {write - dst_start} tokens, expected {expected_end - dst_start}"
            )

    return RealCompactResult(
        hidden_out=hidden_out,
        residual_out=residual_out,
        positions_out=positions_out,
        slot_mapping_out=slot_mapping_out,
        real_cu_seqlens=plan.real_cu_seqlens.to(device),
        kept_block_mask=kept_block_mask_cpu.to(device),
    )


def _indices_from_mask(mask: torch.Tensor, value: bool) -> list[int]:
    return torch.nonzero(mask.cpu() == value, as_tuple=False).flatten().tolist()


def _head_tail(values: torch.Tensor, n: int = 16) -> tuple[list[int], list[int]]:
    values_cpu = values.detach().cpu().to(torch.int64)
    if values_cpu.numel() <= 2 * n:
        vals = values_cpu.tolist()
        return vals, []
    return values_cpu[:n].tolist(), values_cpu[-n:].tolist()


def demo() -> None:
    torch.manual_seed(0)

    # Example batch:
    #   req0: 8192 tokens -> 128 blocks when block_size=64
    #   req1: 8192 tokens -> 128 blocks when block_size=64
    seq_lens = torch.tensor([8192, 8192], dtype=torch.int32)
    cu_seqlens = get_cu_seqlens_from_seq_lens(seq_lens)
    block_size = 64
    drop_ratio = 0.05
    attention_sink = 128
    last_q = 128
    hidden_size = 3

    cu_block_seqlens = get_cu_block_seqlens_from_seq_lens(seq_lens, block_size)
    total_tokens = int(cu_seqlens[-1].item())
    total_blocks = int(cu_block_seqlens[-1].item())

    token_ids = torch.arange(total_tokens, dtype=torch.float32)
    hidden_states = token_ids[:, None].repeat(1, hidden_size)
    residual = hidden_states + 1000
    positions = torch.arange(total_tokens, dtype=torch.int64)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int32) + 5000

    # Higher score means more important middle block.
    block_scores = torch.linspace(0.1, 1.0, steps=total_blocks, dtype=torch.float32)
    assert block_scores.numel() == total_blocks

    plan = compute_real_compact_plan_from_seq_lens(
        seq_lens, block_size, drop_ratio, attention_sink, last_q
    )
    result = topk_block_compact_reference(
        hidden_states,
        residual,
        positions,
        slot_mapping,
        block_scores,
        seq_lens,
        cu_block_seqlens,
        plan,
        block_size,
        attention_sink,
        last_q,
    )

    assert plan.drop_middle_blocks_per_req.tolist() == [6, 6]
    assert plan.real_lens.tolist() == [7808, 7808]
    assert tuple(result.hidden_out.shape) == (15616, hidden_size)
    assert int((~result.kept_block_mask.cpu()).sum()) == 12

    print("Input shapes:")
    print(f"  hidden_states: {tuple(hidden_states.shape)}")
    print(f"  residual:      {tuple(residual.shape)}")
    print(f"  positions:     {tuple(positions.shape)}")
    print(f"  slot_mapping:  {tuple(slot_mapping.shape)}")
    print(f"  block_scores:  {tuple(block_scores.shape)}")
    print("Host-known real compact plan:")
    print(f"  cu_seqlens:                 {cu_seqlens.tolist()}")
    print(f"  cu_block_seqlens:           {cu_block_seqlens.tolist()}")
    print(f"  drop_middle_blocks_per_req: {plan.drop_middle_blocks_per_req.tolist()}")
    print(f"  keep_middle_blocks_per_req: {plan.keep_middle_blocks_per_req.tolist()}")
    print(f"  sink_token_lens:            {plan.sink_token_lens.tolist()}")
    print(f"  middle_token_lens:          {plan.middle_token_lens.tolist()}")
    print(f"  tail_token_lens:            {plan.tail_token_lens.tolist()}")
    print(f"  real_lens:                  {plan.real_lens.tolist()}")
    print(f"  real_cu_seqlens:            {plan.real_cu_seqlens.tolist()}")
    print("Output shapes:")
    print(f"  hidden_out:       {tuple(result.hidden_out.shape)}")
    print(f"  residual_out:     {tuple(result.residual_out.shape)}")
    print(f"  positions_out:    {tuple(result.positions_out.shape)}")
    print(f"  slot_mapping_out: {tuple(result.slot_mapping_out.shape)}")
    dropped_blocks = _indices_from_mask(result.kept_block_mask, False)
    kept_blocks = _indices_from_mask(result.kept_block_mask, True)
    token_head, token_tail = _head_tail(result.hidden_out[:, 0])
    slot_head, slot_tail = _head_tail(result.slot_mapping_out)

    print("Selection:")
    print(f"  kept_block_count:    {len(kept_blocks)}")
    print(f"  dropped_block_count: {len(dropped_blocks)}")
    print(f"  dropped_blocks:      {dropped_blocks}")
    print("Compacted token ids:")
    print(f"  head: {token_head}")
    print(f"  tail: {token_tail}")
    print("Compacted slot ids:")
    print(f"  head: {slot_head}")
    print(f"  tail: {slot_tail}")


if __name__ == "__main__":
    demo()

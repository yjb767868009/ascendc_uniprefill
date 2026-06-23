from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TopKSelectionConfig:
    block_size: int = 64
    attention_sink: int = 128
    last_q: int = 128
    middle_keep_ratio: float = 0.05
    min_middle_keep_blocks: int = 2
    # Short prompts are cheap and less stable to prune. Keep them dense.
    drop_threshold_extra_blocks: int = 4


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def get_cu_block_seqlens(cu_seqlens: torch.Tensor, block_size: int) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    blocks = torch.div(seqlens + block_size - 1, block_size, rounding_mode="floor")
    out = torch.zeros_like(cu_seqlens)
    out[1:] = torch.cumsum(blocks, dim=0)
    return out


def _safe_last_q(seq_len: int, last_q: int) -> int:
    return max(1, min(seq_len, last_q))


def compute_block_scores_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    block_size: int,
    last_q: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for UniPrefill block importance.

    Args:
        q: [total_tokens, num_q_heads * head_dim] or
           [total_tokens, num_q_heads, head_dim].
        k: [total_tokens, num_kv_heads * head_dim] or
           [total_tokens, num_kv_heads, head_dim].
        cu_seqlens: int32 tensor, shape [batch + 1].

    Returns:
        block_scores: [total_blocks], same device as q.
        cu_block_seqlens: int32 tensor, shape [batch + 1].

    This intentionally mirrors the GPU top-p selection path up to block_scores.
    It is a correctness/reference implementation; the final NPU path should
    replace this with AscendC kernels to avoid Python-side dynamic control.
    """
    q = q.reshape(q.shape[0], -1, head_dim)
    k = k.reshape(k.shape[0], -1, head_dim)

    total_tokens, num_q_heads, d_head = q.shape
    _, num_kv_heads, _ = k.shape
    if d_head != head_dim:
        raise ValueError(f"head_dim mismatch: got {d_head}, expected {head_dim}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads must be divisible by num_kv_heads, got "
            f"{num_q_heads} and {num_kv_heads}"
        )

    cu_block_seqlens = get_cu_block_seqlens(cu_seqlens, block_size)
    total_blocks = int(cu_block_seqlens[-1].item())
    block_scores = torch.zeros(total_blocks, dtype=torch.float32, device=q.device)

    group_size = num_q_heads // num_kv_heads
    scale = head_dim ** -0.5
    batch = cu_seqlens.numel() - 1

    for req in range(batch):
        token_start = int(cu_seqlens[req].item())
        token_end = int(cu_seqlens[req + 1].item())
        seq_len = token_end - token_start
        if seq_len <= 0:
            continue

        effective_last_q = _safe_last_q(seq_len, last_q)
        q_req = q[token_end - effective_last_q:token_end]
        k_req = k[token_start:token_end]
        k_rep = k_req.repeat_interleave(group_size, dim=1)

        # [last_q, num_q_heads, head_dim] x [seq_len, num_q_heads, head_dim]
        # -> [num_q_heads, last_q, seq_len]
        scores = torch.einsum("qhd,khd->hqk", q_req, k_rep) * scale

        q_abs = torch.arange(
            seq_len - effective_last_q,
            seq_len,
            device=q.device,
            dtype=torch.long,
        )
        k_abs = torch.arange(seq_len, device=q.device, dtype=torch.long)
        causal_mask = k_abs[None, :] > q_abs[:, None]
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))

        probs = torch.softmax(scores, dim=-1)
        probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)

        # Match the Triton path more closely: sum over last_q rows, then sum heads
        # inside each block. Constant factors do not affect top-k ordering.
        token_scores = probs.sum(dim=1).sum(dim=0).float()  # [seq_len]

        block_start = int(cu_block_seqlens[req].item())
        num_blocks = int(cu_block_seqlens[req + 1].item()) - block_start
        for block in range(num_blocks):
            lo = block * block_size
            hi = min(lo + block_size, seq_len)
            block_scores[block_start + block] = token_scores[lo:hi].sum()

    return block_scores.to(q.dtype), cu_block_seqlens


def top_k_middle_block_mask(
    block_scores: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_seqlens: torch.Tensor,
    block_size: int,
    attention_sink: int,
    last_q: int,
    middle_keep_ratio: float = 0.05,
    min_middle_keep_blocks: int = 2,
) -> torch.Tensor:
    """Select fixed-budget middle blocks and always keep sink/tail blocks.

    The 5% ratio is applied to the prunable middle region, not to sink/tail.
    This keeps the output budget predictable while preserving safety tokens.
    """
    if not (0.0 < middle_keep_ratio <= 1.0):
        raise ValueError("middle_keep_ratio must be in (0, 1]")
    if min_middle_keep_blocks < 0:
        raise ValueError("min_middle_keep_blocks must be non-negative")

    mask = torch.zeros(block_scores.numel(), dtype=torch.bool, device=block_scores.device)
    batch = cu_block_seqlens.numel() - 1

    for req in range(batch):
        token_start = int(cu_seqlens[req].item())
        token_end = int(cu_seqlens[req + 1].item())
        seq_len = token_end - token_start
        block_start = int(cu_block_seqlens[req].item())
        block_end = int(cu_block_seqlens[req + 1].item())
        num_blocks = block_end - block_start
        if seq_len <= 0 or num_blocks <= 0:
            continue

        sink_blocks = min(num_blocks, ceil_div(min(attention_sink, seq_len), block_size))
        tail_tokens = min(last_q, max(0, seq_len - sink_blocks * block_size))
        tail_blocks = min(num_blocks - sink_blocks, ceil_div(tail_tokens, block_size))

        sink_end = block_start + sink_blocks
        tail_start = block_end - tail_blocks
        if sink_blocks > 0:
            mask[block_start:sink_end] = True
        if tail_blocks > 0:
            mask[tail_start:block_end] = True

        middle_start = sink_end
        middle_end = tail_start
        middle_blocks = max(0, middle_end - middle_start)
        if middle_blocks <= 0:
            continue

        keep_middle = ceil_div(int(middle_blocks * middle_keep_ratio * 10000), 10000)
        keep_middle = max(keep_middle, min_middle_keep_blocks)
        keep_middle = min(keep_middle, middle_blocks)
        if keep_middle <= 0:
            continue

        middle_scores = block_scores[middle_start:middle_end].float().clamp_min(0)
        # stable=True helps CPU reference determinism. NPU kernels can use any
        # deterministic tie-breaking rule as long as tests mirror it.
        order = torch.argsort(middle_scores, descending=True, stable=True)
        selected = order[:keep_middle] + middle_start
        mask[selected] = True

    return mask


def expand_block_mask(
    block_mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_seqlens: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand a block mask to token mask and compact sequence metadata."""
    total_tokens = int(cu_seqlens[-1].item())
    token_mask = torch.zeros(total_tokens, dtype=torch.bool, device=block_mask.device)
    batch = cu_seqlens.numel() - 1
    new_seq_lens = torch.zeros(batch, dtype=torch.int32, device=cu_seqlens.device)

    for req in range(batch):
        token_start = int(cu_seqlens[req].item())
        token_end = int(cu_seqlens[req + 1].item())
        seq_len = token_end - token_start
        block_start = int(cu_block_seqlens[req].item())
        block_end = int(cu_block_seqlens[req + 1].item())
        kept = 0

        for block in range(block_end - block_start):
            if not bool(block_mask[block_start + block].item()):
                continue
            lo = block * block_size
            hi = min(lo + block_size, seq_len)
            if hi <= lo:
                continue
            token_mask[token_start + lo:token_start + hi] = True
            kept += hi - lo

        new_seq_lens[req] = kept

    new_cu_seqlens = torch.zeros_like(cu_seqlens)
    if batch > 0:
        new_cu_seqlens[1:] = torch.cumsum(new_seq_lens, dim=0)
    new_max_seq_len = torch.max(new_seq_lens) if batch > 0 else torch.zeros((), dtype=torch.int32, device=cu_seqlens.device)
    return token_mask, new_cu_seqlens, new_max_seq_len.to(torch.int32)


def topkselectionvarlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    block_size: int = 64,
    attention_sink: int = 128,
    last_q: int = 128,
    middle_keep_ratio: float = 0.05,
    min_middle_keep_blocks: int = 2,
    drop_threshold_extra_blocks: int = 4,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """GPU top-p style reference with fixed-ratio block top-k selection.

    Returns:
        token_mask: [total_tokens] bool tensor.
        new_max_seq_len: Python int, kept for compatibility with current vLLM
            Python integration. AscendC/vLLM optimized paths should avoid reading
            this from device in the hot path.
        new_cu_seqlens: int32 tensor, shape [batch + 1].
        block_mask: bool tensor over prefill compact blocks. This is useful for
            validating the future AscendC top-k kernel and block compact kernel.
    """
    cfg = TopKSelectionConfig(
        block_size=block_size,
        attention_sink=attention_sink,
        last_q=last_q,
        middle_keep_ratio=middle_keep_ratio,
        min_middle_keep_blocks=min_middle_keep_blocks,
        drop_threshold_extra_blocks=drop_threshold_extra_blocks,
    )

    q = q.reshape(q.shape[0], -1, head_dim)
    k = k.reshape(k.shape[0], -1, head_dim)
    total_tokens = q.shape[0]
    batch = cu_seqlens.numel() - 1
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

    dense_threshold = attention_sink + last_q + drop_threshold_extra_blocks * block_size
    is_prefill_req = seq_lens > 1
    should_drop_req = is_prefill_req & (seq_lens > dense_threshold)
    prefill_indices = torch.nonzero(should_drop_req, as_tuple=True)[0]

    if prefill_indices.numel() == 0:
        token_mask = torch.ones(total_tokens, dtype=torch.bool, device=q.device)
        empty_block_mask = torch.empty(0, dtype=torch.bool, device=q.device)
        return token_mask, int(max_seq_len), cu_seqlens.clone(), empty_block_mask

    prefill_seq_lens = seq_lens[prefill_indices]
    prefill_cu_seqlens = torch.zeros(prefill_indices.numel() + 1, dtype=torch.int32, device=q.device)
    prefill_cu_seqlens[1:] = torch.cumsum(prefill_seq_lens.to(torch.int32), dim=0)
    prefill_total = int(prefill_cu_seqlens[-1].item())

    starts = cu_seqlens[prefill_indices]
    req_ids = torch.repeat_interleave(
        torch.arange(prefill_indices.numel(), device=q.device, dtype=torch.int64),
        prefill_seq_lens.long(),
    )
    local_offsets = torch.arange(prefill_total, device=q.device, dtype=torch.int64)
    intra_pos = local_offsets - prefill_cu_seqlens[req_ids].long()
    prefill_token_indices = starts[req_ids].long() + intra_pos

    prefill_q = q[prefill_token_indices]
    prefill_k = k[prefill_token_indices]

    block_scores, prefill_cu_block_seqlens = compute_block_scores_from_qk(
        prefill_q,
        prefill_k,
        head_dim,
        prefill_cu_seqlens,
        cfg.block_size,
        cfg.last_q,
    )
    block_mask = top_k_middle_block_mask(
        block_scores,
        prefill_cu_seqlens,
        prefill_cu_block_seqlens,
        cfg.block_size,
        cfg.attention_sink,
        cfg.last_q,
        cfg.middle_keep_ratio,
        cfg.min_middle_keep_blocks,
    )
    prefill_token_mask, prefill_new_cu, _ = expand_block_mask(
        block_mask,
        prefill_cu_seqlens,
        prefill_cu_block_seqlens,
        cfg.block_size,
    )

    token_mask = torch.ones(total_tokens, dtype=torch.bool, device=q.device)
    token_mask[prefill_token_indices] = prefill_token_mask

    new_seq_lens = seq_lens.to(torch.int32).clone()
    new_seq_lens[prefill_indices] = prefill_new_cu[1:] - prefill_new_cu[:-1]
    new_cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32, device=q.device)
    new_cu_seqlens[1:] = torch.cumsum(new_seq_lens, dim=0)
    new_max_seq_len = int(new_seq_lens.max().item())

    return token_mask, new_max_seq_len, new_cu_seqlens, block_mask


def topk_block_selection_from_scores(
    block_scores: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_seqlens: torch.Tensor,
    block_size: int = 64,
    attention_sink: int = 128,
    last_q: int = 128,
    middle_keep_ratio: float = 0.05,
    min_middle_keep_blocks: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Small helper matching the current AscendC project scope.

    This starts from block_scores and returns the same four tensors as the
    existing top-p golden: block_mask, token_mask, new_cu_seqlens, new_max_seq_len.
    """
    block_mask = top_k_middle_block_mask(
        block_scores,
        cu_seqlens,
        cu_block_seqlens,
        block_size,
        attention_sink,
        last_q,
        middle_keep_ratio,
        min_middle_keep_blocks,
    )
    token_mask, new_cu_seqlens, new_max_seq_len = expand_block_mask(
        block_mask,
        cu_seqlens,
        cu_block_seqlens,
        block_size,
    )
    return block_mask, token_mask, new_cu_seqlens, new_max_seq_len

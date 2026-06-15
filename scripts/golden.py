from __future__ import annotations

import torch


def get_cu_block_seqlens(cu_seqlens: torch.Tensor, block_size: int) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    blocks = torch.div(seqlens + block_size - 1, block_size, rounding_mode="floor")
    out = torch.zeros_like(cu_seqlens)
    out[1:] = torch.cumsum(blocks, dim=0)
    return out


def top_p_block_mask(
    block_scores: torch.Tensor,
    cu_block_seqlens: torch.Tensor,
    p: float,
) -> torch.Tensor:
    block_scores_cpu = block_scores.detach().cpu().float()
    cu_blocks_cpu = cu_block_seqlens.detach().cpu().int()
    mask = torch.zeros(block_scores_cpu.numel(), dtype=torch.bool)

    batch = cu_blocks_cpu.numel() - 1
    for req in range(batch):
        start = int(cu_blocks_cpu[req].item())
        end = int(cu_blocks_cpu[req + 1].item())
        scores = block_scores_cpu[start:end].clamp_min(0)
        if scores.numel() == 0:
            continue

        score_sum = scores.sum()
        if score_sum <= 0:
            mask[start:end] = True
            continue

        probs = scores / score_sum
        order = torch.argsort(probs, descending=True, stable=True)
        cumulative = 0.0
        kept = 0
        for idx in order.tolist():
            if cumulative <= p or kept == 0:
                mask[start + idx] = True
                cumulative += float(probs[idx].item())
                kept += 1
            else:
                break

    return mask.to(block_scores.device)


def expand_block_mask(
    block_mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_seqlens: torch.Tensor,
    block_size: int,
    attention_sink: int,
    last_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cu_seq_cpu = cu_seqlens.detach().cpu().int()
    cu_block_cpu = cu_block_seqlens.detach().cpu().int()
    block_mask_cpu = block_mask.detach().cpu().bool()

    total_tokens = int(cu_seq_cpu[-1].item())
    token_mask = torch.zeros(total_tokens, dtype=torch.bool)
    new_lens = []

    batch = cu_seq_cpu.numel() - 1
    for req in range(batch):
        token_start = int(cu_seq_cpu[req].item())
        token_end = int(cu_seq_cpu[req + 1].item())
        block_start = int(cu_block_cpu[req].item())
        seq_len = token_end - token_start

        kept = 0
        for offset in range(seq_len):
            block_offset = offset // block_size
            keep = bool(block_mask_cpu[block_start + block_offset].item())
            keep = keep or offset < attention_sink
            keep = keep or offset >= seq_len - last_q
            token_mask[token_start + offset] = keep
            kept += int(keep)
        new_lens.append(kept)

    new_lens_t = torch.tensor(new_lens, dtype=torch.int32)
    new_cu = torch.zeros(batch + 1, dtype=torch.int32)
    if batch:
        new_cu[1:] = torch.cumsum(new_lens_t, dim=0)
    new_max = torch.tensor(int(new_lens_t.max().item()) if batch else 0, dtype=torch.int32)
    return token_mask.to(block_mask.device), new_cu.to(cu_seqlens.device), new_max.to(cu_seqlens.device)


def compute_golden(
    block_scores: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_seqlens: torch.Tensor,
    block_size: int,
    attention_sink: int,
    last_q: int,
    p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_mask = top_p_block_mask(block_scores, cu_block_seqlens, p)
    token_mask, new_cu, new_max = expand_block_mask(
        block_mask,
        cu_seqlens,
        cu_block_seqlens,
        block_size,
        attention_sink,
        last_q,
    )
    return block_mask, token_mask, new_cu, new_max


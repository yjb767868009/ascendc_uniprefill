# AscendC TopSelection Implementation Plan

## Goal

Port UniPrefill's top-p block selection path to AscendC.

The complete original flow is:

```text
q, k
  -> QK score for last_q queries
  -> causal softmax
  -> token/head importance
  -> block_scores
  -> top-p block_mask
  -> token_mask and new sequence metadata
```

This standalone AscendC project starts from the last two stages:

```text
block_scores -> block_mask -> token_mask
```

## Milestones

| Milestone | Status | Deliverable |
| --- | --- | --- |
| M1 | Done | standalone project scaffold |
| M2 | Done | PyTorch golden reference |
| M3 | Done | AscendC top-p block selection kernel |
| M4 | Done | AscendC block-mask-to-token-mask kernel |
| M5 | Next | build and runtime debug on NPU |
| M6 | Next | replace scalar GM access with UB/SIMT optimized path |
| M7 | Later | block reduce kernel: token/head importance -> block_scores |
| M8 | Later | softmax importance kernel |
| M9 | Later | QK part GEMM kernel |
| M10 | Later | vLLM integration |

## Current MVP Kernels

### `topselection_top_p_kernel`

Input:

```text
block_scores[total_blocks]
cu_block_seqlens[batch + 1]
p
```

Output:

```text
block_mask[total_blocks]
```

Semantics:

1. For each request, read valid block scores.
2. Clamp negative scores to zero.
3. Normalize by score sum.
4. Repeatedly select the largest remaining normalized score.
5. Keep blocks while previous cumulative probability is `<= p`.
6. Keep at least one block.
7. If score sum is zero, keep all blocks for that request.

### `topselection_expand_mask_kernel`

Input:

```text
block_mask[total_blocks]
cu_seqlens[batch + 1]
cu_block_seqlens[batch + 1]
block_size
attention_sink
last_q
```

Output:

```text
token_mask[total_tokens]
new_seq_lens[batch]
```

Semantics:

```text
keep_token =
    block_mask[token_offset // block_size]
    OR token_offset < attention_sink
    OR token_offset >= seq_len - last_q
```

The host wrapper converts `new_seq_lens` to `new_cu_seqlens` and
`new_max_seq_len`.

## Build/Debug Plan

1. Build the project with `bash run.sh`.
2. Fix any AscendC compile issues from the first NPU build.
3. Run `scripts/test_torch.py`.
4. Compare `block_mask`, `token_mask`, `new_cu_seqlens`, and
   `new_max_seq_len` against `scripts/golden.py`.
5. Replace scalar GM access with UB staging or SIMT direct loops.
6. Add zero-score and boundary tests after first successful device run.

## Accuracy Tests

Required cases:

- single request, small sequence
- single request, tail block
- multiple varlen requests
- `p = 1.0`
- zero block scores fallback
- `attention_sink > seq_len`
- `last_q > seq_len`
- non-multiple `block_size`

## Future Full-Chain Work

### Block Reduce

Add:

```text
token_importance[prefill_total, num_q_heads] -> block_scores[total_blocks]
```

Use fp32 accumulation and handle tail blocks carefully.

### Softmax Importance

Add:

```text
scores[last_q, prefill_total, num_q_heads] -> token_importance[prefill_total, num_q_heads]
```

Use stable softmax and causal masking.

### QK Part GEMM

Add:

```text
q[prefill_total, num_q_heads, head_dim]
k[prefill_total, num_k_heads, head_dim]
-> scores[last_q, prefill_total, num_q_heads]
```

Start with a simple correctness implementation, then optimize with Cube or a
matmul-fusion design.

## Known Limitations

- MVP supports fp32 `block_scores` only.
- The kernel currently prioritizes correctness over performance.
- The full `q/k` to `block_scores` chain is not implemented yet.
- The project has not been fully compiled and executed after relocation until
  `bash run.sh` is run on the target NPU environment.

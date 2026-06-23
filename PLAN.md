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

## Sparse Hidden-State Update Designs

The current UniPrefill Python path uses dynamic indexing after top-p selection:

```python
kept_indices = token_mask.nonzero(as_tuple=True)[0]
n_kept = kept_indices.shape[0]
hidden_states[:n_kept] = hidden_states[kept_indices]
residual[:n_kept] = residual[kept_indices]
positions[:n_kept] = positions[kept_indices]
```

This is unfriendly to NPU execution because `nonzero`, dynamic index gather,
and host-visible dynamic lengths can introduce synchronization. The AscendC
port should avoid exposing variable-length indices to Python.

### Option A: Block-Level Compact Operator

Compact at block granularity and keep output buffers with fixed capacity.

```text
hidden_states[total_tokens, hidden_dim]
residual[total_tokens, hidden_dim]
positions[total_tokens]
block_mask[total_blocks]
cu_seqlens[batch + 1]
cu_block_seqlens[batch + 1]
  -> compact_hidden[total_tokens, hidden_dim]
  -> compact_residual[total_tokens, hidden_dim]
  -> compact_positions[total_tokens]
  -> new_cu_seqlens[batch + 1]
  -> new_max_seq_len
```

Semantics:

1. Treat `block_mask` as the primary selection result.
2. Force keep prefix and suffix by block, not by individual token:
   - `sink_blocks = ceil(attention_sink / block_size)`
   - `tail_blocks = ceil(last_q / block_size)`
3. For each request, copy selected blocks into a compact contiguous region.
4. Write `new_seq_lens[req]` as the kept token count after block expansion.
5. Keep tensor capacity static; downstream metadata determines valid tokens.

Recommended API:

```python
compact_hidden, compact_residual, compact_positions, new_cu_seqlens, new_max_seq_len = (
    torch.ops.npu.uniprefill_block_compact(
        hidden_states,
        residual,
        positions,
        block_mask,
        cu_seqlens,
        cu_block_seqlens,
        block_size,
        attention_sink,
        last_q,
    )
)
```

Pros:

- Avoids token-level `nonzero`.
- Produces regular block copies that are easier to implement and optimize in
  AscendC.
- Matches the current top-p selection granularity.
- More likely to avoid host-device synchronization in vLLM integration.

Cons:

- Keeps extra tokens when forced sink/tail boundaries are not block-aligned.
- May lose some sparsity compared with exact token-level compaction.
- Requires downstream code to consume fixed-capacity compact buffers using
  device-side sequence metadata.

Implementation plan:

1. Add `uniprefill_block_compact` to `op_extension/register.cpp`.
2. Add tiling data for `batch`, `maxSeqLen`, `hiddenDim`, `blockSize`, and
   maximum blocks per request.
3. Kernel 1 computes per-request kept block counts and `new_seq_lens`.
4. Host wrapper or a later scan kernel builds `new_cu_seqlens`.
5. Kernel 2 copies selected blocks of hidden/residual/positions into compact
   output.
6. Validate against a PyTorch block-compaction golden reference.

This should be the first implementation target because it removes the largest
NPU synchronization hazard with the least semantic complexity.

### Option B: Token-Level Prefix-Sum Compact Operator

Keep exact token-level semantics but move all dynamic indexing into AscendC.

```text
hidden_states[total_tokens, hidden_dim]
residual[total_tokens, hidden_dim]
positions[total_tokens]
token_mask[total_tokens]
cu_seqlens[batch + 1]
  -> compact_hidden[total_tokens, hidden_dim]
  -> compact_residual[total_tokens, hidden_dim]
  -> compact_positions[total_tokens]
  -> new_cu_seqlens[batch + 1]
  -> new_max_seq_len
```

Semantics:

1. Convert `token_mask` to per-token keep flags.
2. Compute a per-request prefix sum over keep flags.
3. For each kept token, scatter it to:

   ```text
   output_offset = new_cu_seqlens[req] + prefix_keep_count[token] - 1
   ```

4. Keep output buffers fixed-capacity and expose valid lengths through
   `new_cu_seqlens`.

Recommended API:

```python
compact_hidden, compact_residual, compact_positions, new_cu_seqlens, new_max_seq_len = (
    torch.ops.npu.uniprefill_token_compact(
        hidden_states,
        residual,
        positions,
        token_mask,
        cu_seqlens,
    )
)
```

Pros:

- Preserves exact current Python semantics.
- Maximizes sparsity and avoids block-boundary over-retention.
- Can be reused if future selection becomes token-level instead of block-level.

Cons:

- Requires a parallel prefix-sum implementation.
- More difficult to optimize for long variable-length requests.
- Needs careful handling of cross-request offsets and large hidden dimensions.
- More likely to need multiple kernels or a temporary prefix buffer.

Implementation plan:

1. Add a PyTorch golden reference for exact token compaction.
2. Implement per-request keep-count kernel.
3. Build `new_cu_seqlens` on host for MVP, then replace with device scan if
   host synchronization becomes measurable.
4. Implement token scatter-copy kernel using prefix offsets.
5. Extend the copy path to hidden/residual/positions together to avoid repeated
   memory passes.
6. Benchmark against Option A using the same `block_mask`/`token_mask` inputs.

This option should be kept as the accuracy-preserving fallback, but it is not
the best first target for NPU performance.

### Fixed-Ratio Top-K Selection Reference

For NPU serving, top-p can be replaced by fixed-ratio block top-k to make the
kept-token budget predictable. The current reference implementation is:

```text
scripts/top_k_selection.py
```

Default policy:

```text
keep all sink blocks
keep all tail blocks
keep top ceil(middle_blocks * 0.05) middle blocks
keep at least 2 middle blocks when a middle region exists
```

This ratio is applied only to the prunable middle region, not to forced
attention-sink or tail blocks. The script exposes both a block-score-only helper
for the current AscendC project scope and a q/k-to-mask reference mirroring the
GPU top-p path.

### Decision

Start with Option A. UniPrefill already selects blocks, so block-level compact
keeps the execution regular and avoids dynamic token indices. Implement Option B
only if block over-retention causes unacceptable quality or speed loss.

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

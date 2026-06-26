# No-Sync Fixed Top-K Compact Design

This document defines the production direction for UniPrefill sparse prefill on
Ascend NPU. The key requirement is: **no D2H readback is allowed on the runtime
critical path**.

## Problem

The original Python/Triton integration can introduce synchronization through
patterns like:

```python
hidden_states = hidden_states[token_mask]
int(cu_seqlens[-1].item())
new_seq_lens.cpu()
token_mask.sum()
```

Any of these can drain the NPU queue because host code needs a value produced by
device work. A single synchronization is already harmful; repeated synchroniza-
tions are worse, but the real target is zero synchronization in the selection and
compact path.

## Contract

The no-sync operator must be an `_out` style operator:

```python
torch.ops.npu.uniprefill_fixed_topk_compact_out(
    hidden_states,
    residual,
    positions,
    slot_mapping,
    block_scores,
    cu_seqlens,
    cu_block_seqlens,
    real_cu_seqlens,
    keep_middle_blocks_per_req,
    hidden_out,
    residual_out,
    positions_out,
    slot_mapping_out,
    kept_block_mask,
    block_size,
    attention_sink,
    last_q,
)
```

The caller allocates every output before launch. The operator only fills existing
NPU tensors and returns them. It does not allocate an output whose shape depends
on device-computed data.

## Host Responsibilities

The host may use CPU-side request metadata that already exists before launch:

- `seq_lens_cpu`: `[batch]`
- `block_size`
- `attention_sink`
- `last_q`
- `drop_ratio`, currently fixed to `0.05`

From those values, the host computes:

- `cu_seqlens_cpu`: `[batch + 1]`
- `cu_block_seqlens_cpu`: `[batch + 1]`
- `keep_middle_blocks_per_req_cpu`: `[batch]`
- `real_cu_seqlens_cpu`: `[batch + 1]`
- `total_real_tokens = real_cu_seqlens_cpu[-1]`
- `total_blocks = cu_block_seqlens_cpu[-1]`

The host then creates/copies the device metadata and allocates outputs with
fixed, host-known shapes:

```text
hidden_out       [total_real_tokens, hidden_size]
residual_out     [total_real_tokens, hidden_size]
positions_out    [total_real_tokens]
slot_mapping_out [total_real_tokens]
kept_block_mask  [total_blocks]
```

Copying host metadata to device is H2D and does not require reading the NPU
queue. The forbidden operation is reading a value produced on device back to host
before launching the next step.

## Device Responsibilities

The device performs all data-dependent selection and copy work:

1. Read `block_scores` and request metadata.
2. Force keep sink and tail blocks.
3. For middle blocks, keep the `keep_middle_blocks_per_req[req]` largest scores.
4. Write `kept_block_mask` for debug/inspection.
5. Compact kept real tokens into the fixed output ranges described by
   `real_cu_seqlens`.

The kernel may branch on device values internally. It must not expose a dynamic
count to Python/C++ for allocation.

## Current MVP Scope

The first no-sync MVP uses existing `block_scores` as input and implements:

```text
block_scores -> fixed 5% top-k middle block selection -> real compact copy
```

This removes the dynamic `hidden_states[token_mask]` path. The next kernel stage
will compute `q/k -> block_scores` on device and either feed this compact kernel
or fuse both stages.

## Shape Example: 8K x 2

Configuration:

```text
seq_lens = [8192, 8192]
block_size = 64
attention_sink = 128
last_q = 128
drop_ratio = 0.05
```

Per request:

```text
num_blocks = 128
sink_blocks = 2
tail_blocks = 2
middle_blocks = 124
drop_middle_blocks = floor(124 * 0.05) = 6
keep_middle_blocks = 118
real_len = 128 + 118 * 64 + 128 = 7808
```

Batch outputs:

```text
real_cu_seqlens = [0, 7808, 15616]
hidden_out.shape = [15616, hidden_size]
kept_block_mask.shape = [256]
```

## Explicit Non-Goals

- No `token_mask.sum()`.
- No `new_seq_lens.cpu()`.
- No `cu_seqlens[-1].item()` from an NPU tensor.
- No output allocation based on a device-computed scalar.
- No Python boolean indexing for compacting hidden states.

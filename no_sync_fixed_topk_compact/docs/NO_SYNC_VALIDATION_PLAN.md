# No-Sync Fixed Top-K Compact Validation Plan

This document is for the engineer who has access to an Ascend NPU machine. The
agent that prepared this code cannot validate NPU runtime behavior locally, so
this plan defines the exact experiments, scripts, baselines, and acceptance
criteria.

## Goal

Validate the no-sync `_out` operator:

```python
torch.ops.npu.uniprefill_fixed_topk_compact_out(...)
```

The operator must compact real tokens according to a host-known fixed 5% middle
block drop policy without reading any device-produced value back to host in the
hot path.

Current MVP scope:

```text
block_scores -> fixed top-k middle block selection -> real compact copy
```

Not yet included in this MVP:

```text
q/k -> block_scores
```

That stage will be validated separately once the AscendC block-score kernel is
added or fused.

## Repository And Build

```bash
cd /autodl-fs/data/yjb/ascendc_uniprefill/no_sync_fixed_topk_compact
source "$ASCEND_HOME_PATH/set_env.sh"
cmake -S . -B build
cmake --build build -j4
```

Expected build artifact:

```text
build/libuniprefill_no_sync_ops.so
```

If build fails, save:

```bash
cmake -S . -B build 2>&1 | tee build_config.log
cmake --build build -j4 2>&1 | tee build_compile.log
```

## Test Script

Correctness:

```bash
python3 scripts/validate_fixed_topk_compact_out.py --mode correctness
```

Benchmark:

```bash
python3 scripts/validate_fixed_topk_compact_out.py --mode benchmark --iters 100 --warmup 20
```

8K target case:

```bash
python3 scripts/validate_fixed_topk_compact_out.py \
  --mode benchmark \
  --seq-lens 8192,8192 \
  --hidden-size 4096 \
  --block-size 64 \
  --attention-sink 128 \
  --last-q 128 \
  --drop-ratio 0.05 \
  --iters 100 \
  --warmup 20
```

Profiler trace:

```bash
python3 scripts/validate_fixed_topk_compact_out.py \
  --mode benchmark \
  --profile \
  --profile-dir ./prof_fixed_topk_compact \
  --seq-lens 8192,8192 \
  --hidden-size 4096
```

## Baselines

The script reports three paths:

- `cpu_golden`: pure CPU correctness reference.
- `npu_python_mask_baseline`: deliberately bad baseline using Python boolean indexing.
- `npu_fixed_topk_compact_out`: new `_out` operator with preallocated outputs.

The `_out` measured loop must not call `.cpu()`, `.item()`, `nonzero`, or `sum`
on device values.

## Correctness Acceptance

For every case, the script must print `PASSED` for:

```text
hidden_out
residual_out
positions_out
slot_mapping_out
kept_block_mask
```

Required cases:

```text
short_dense:      seq_lens=[256]
single_8k:        seq_lens=[8192]
double_8k:        seq_lens=[8192,8192]
varlen_batch:     seq_lens=[1024,4096,8192]
non_multiple:     seq_lens=[1000,8193]
```

For `double_8k` with `block_size=64`, `attention_sink=128`, `last_q=128`,
`drop_ratio=0.05`, expected metadata is:

```text
cu_seqlens          = [0, 8192, 16384]
cu_block_seqlens    = [0, 128, 256]
keep_middle_blocks  = [118, 118]
real_cu_seqlens     = [0, 7808, 15616]
hidden_out.shape    = [15616, hidden_size]
kept_block_mask.sum = 244
```

## Performance Acceptance

Minimum acceptance for the 8K x 2 case:

```text
npu_fixed_topk_compact_out correctness: PASS
npu_fixed_topk_compact_out median latency < npu_python_mask_baseline median latency
```

Preferred target:

```text
npu_fixed_topk_compact_out median latency <= 70% of npu_python_mask_baseline
```

The first MVP kernel uses scalar GlobalTensor access and is correctness-first,
not final-performance.

## No-Sync Acceptance

The measured `_out` region must not contain D2H readback caused by this op.
Check profiler trace for absence of:

```text
Tensor.cpu()
Tensor.item()
torch.nonzero(device_tensor)
torch.sum(device_mask) used for output shape
Python boolean indexing used to allocate compact output
aclrtMemcpy DEVICE_TO_HOST
aclrtSynchronizeStream caused by dynamic output shape
```

Allowed before the measured region:

```text
H2D copy of host metadata:
  cu_seqlens
  cu_block_seqlens
  real_cu_seqlens
  keep_middle_blocks_per_req

Output allocation using host-known shapes.
```

## Results Template

| Case | Correctness | Python Mask Median us | Out Op Median us | Speedup | D2H In Out Trace |
| --- | --- | ---: | ---: | ---: | --- |
| short_dense |  |  |  |  |  |
| single_8k |  |  |  |  |  |
| double_8k |  |  |  |  |  |
| varlen_batch |  |  |  |  |  |
| non_multiple |  |  |  |  |  |

Attach script stdout and profiler trace directory if `--profile` was used.

## Implementation Note: GM Visibility

Do not use GlobalTensor `SetValue`/`GetValue` as same-kernel scratch state.
A value written to GM inside a kernel is not guaranteed to be immediately visible
to another `GetValue` in the same launch. The fixed top-k compact kernel must
compute keep/drop decisions from source tensors and host metadata directly, then
write `kept_block_mask` as an output.

## Implementation Note: Bisheng Build And Optimization

The AscendC kernel declaration and the C++ launcher declaration intentionally do
not use `extern "C"`. In remote validation, bisheng still emitted a C++-mangled
kernel symbol, so the C++ side must use the same C++ name mangling for symbol
matching.

The kernel also writes `kept_block_mask` after the compact copy loop. If mask
stores are interleaved before data copy and the mask is not read again, bisheng
can treat them as dead stores and remove them. The final mask loop keeps the
stores as observable kernel output side effects.

## Tiled Performance Variant

The scalar `_out` kernel is a correctness MVP and may be much slower than the
Python mask baseline because it copies `[tokens, hidden]` with one core per
request. The tiled variant should be used for performance validation:

```bash
python3 scripts/validate_fixed_topk_compact_out.py \
  --mode correctness \
  --variant tiled \
  --hidden-tile 256

python3 scripts/validate_fixed_topk_compact_out.py \
  --mode benchmark \
  --variant tiled \
  --hidden-tile 256 \
  --seq-lens 8192,8192 \
  --hidden-size 4096
```

The tiled operator adds host-known `kept_block_cu_seqlens` and preallocated
`kept_block_indices`. These do not require D2H because their shapes and values
come from CPU request metadata and the fixed 5% policy.

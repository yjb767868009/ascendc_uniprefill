# Tiled Fixed-TopK Fix Test Report

## Test Metadata

- Tester:
- Date:
- Machine:
- NPU model:
- CANN version:
- torch version:
- torch_npu version:
- Code commit:
- Branch:

## Build Result

Command:

```bash
cd /home/yjb/ascendc_uniprefill/no_sync_fixed_topk_compact
source "$ASCEND_HOME_PATH/set_env.sh"
rm -rf build
cmake -S . -B build
cmake --build build -j4
```

Result:

- [ ] Passed
- [ ] Failed

Failure log summary:

```text

```

## Correctness Tests

### Tiled Default Correctness

Command:

```bash
python3 scripts/validate_fixed_topk_compact_out.py \
  --mode correctness \
  --variant tiled \
  --hidden-tile 256
```

Expected result:

```text
Total: 5, Passed: 5, Failed: 0
```

Result:

- [ ] Passed
- [ ] Failed

Output summary:

```text
Total:
Passed:
Failed:
```

Failed case:

Failed field:

Log:

```text

```

### Non-Multiple Sequence Correctness

Command:

```bash
python3 scripts/validate_fixed_topk_compact_out.py \
  --mode correctness \
  --variant tiled \
  --hidden-tile 128 \
  --seq-lens 1000,8193 \
  --hidden-size 513
```

Result:

- [ ] Passed
- [ ] Failed

Checklist:

- [ ] `kept_block_indices` passed
- [ ] `kept_block_indices` contains no `-1` sentinel values
- [ ] `kept_block_mask` passed
- [ ] `hidden_out` passed
- [ ] `residual_out` passed
- [ ] `positions_out` passed
- [ ] `slot_mapping_out` passed
- [ ] No vector core exception
- [ ] No out-of-bounds or illegal-address exception

Log:

```text

```

### Scalar Reference Correctness

Command:

```bash
python3 scripts/validate_fixed_topk_compact_out.py \
  --mode correctness \
  --variant scalar
```

Result:

- [ ] Passed
- [ ] Failed

Log:

```text

```

## Stability Test

Repeat tiled default correctness 10 times.

Result:

- [ ] 10/10 passed
- [ ] Failed

Failure count:

Failure log summary:

```text

```

## Performance Tests

### 8K x 2 Benchmark

Command:

```bash
python3 scripts/validate_fixed_topk_compact_out.py \
  --mode benchmark \
  --variant tiled \
  --hidden-tile 256 \
  --seq-lens 8192,8192 \
  --hidden-size 4096 \
  --iters 100 \
  --warmup 20
```

Result:

```text
python_mask_baseline_median_us:
fixed_topk_compact_out_median_us:
speedup_vs_python_mask:
```

Performance regression:

- [ ] No
- [ ] Yes

Notes:

```text

```

### Non-Multiple Benchmark

Command:

```bash
python3 scripts/validate_fixed_topk_compact_out.py \
  --mode benchmark \
  --variant tiled \
  --hidden-tile 128 \
  --seq-lens 1000,8193 \
  --hidden-size 4096 \
  --iters 100 \
  --warmup 20
```

Result:

```text
python_mask_baseline_median_us:
fixed_topk_compact_out_median_us:
speedup_vs_python_mask:
```

Crash observed:

- [ ] No
- [ ] Yes

Notes:

```text

```

## Bug Verification

### Bug 1: `keptBlockIndices` DSE / Uninitialized Slots

Conclusion:

- [ ] Fixed
- [ ] Not fixed

Evidence:

```text
`kept_block_indices` validation result:
Any `-1` sentinel values after tiled run:
RebuildIndices kernel evidence/log summary:
Vector core exception observed:
```

### Bug 2: Incorrect `dstToken` for Partial Final Blocks

Conclusion:

- [ ] Fixed
- [ ] Not fixed

Evidence:

```text
`non_multiple` validation result:
Output mismatch observed:
Out-of-bounds or illegal-address exception observed:
```

## Final Decision

- [ ] Ready to merge
- [ ] Needs more fixes

Remarks:

```text

```

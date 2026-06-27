# No-Sync Fixed Top-K Compact

This isolated subproject contains the no-sync fixed 5% middle-block compact MVP
for UniPrefill. It is intentionally separated from the older `topselection`
MVP because the old path returns dynamic metadata and currently performs D2H
readback in its wrapper.

Current MVP scope:

```text
block_scores -> fixed top-k middle block selection -> real compact copy
```

Not included yet:

```text
q/k -> block_scores
```

The public op is `_out` style:

```python
torch.ops.npu.uniprefill_fixed_topk_compact_out(...)
```

All output tensors are allocated by the caller using host-known metadata. The
wrapper does not read device-produced values via `.cpu()` or `.item()`.

## Layout

```text
no_sync_fixed_topk_compact/
├── CMakeLists.txt
├── README.md
├── run_validation.sh
├── docs/
│   ├── NO_SYNC_FIXED_TOPK_COMPACT.md
│   ├── NO_SYNC_VALIDATION_PLAN.md
│   └── TILED_FIX_TEST_REPORT.md
├── op_kernel/
│   ├── no_sync_tiling.h
│   └── uniprefill_fixed_topk_compact_kernel.asc
├── op_extension/
│   ├── ops.h
│   ├── register.cpp
│   └── uniprefill_fixed_topk_compact_torch.cpp
└── scripts/
    └── validate_fixed_topk_compact_out.py
```

## Build

Run on an Ascend NPU machine:

```bash
cd /autodl-fs/data/yjb/ascendc_uniprefill/no_sync_fixed_topk_compact
source "$ASCEND_HOME_PATH/set_env.sh"
cmake -S . -B build
cmake --build build -j4
```

Expected artifact:

```text
build/libuniprefill_no_sync_ops.so
```

## Validate

Correctness:

```bash
python3 scripts/validate_fixed_topk_compact_out.py --mode correctness
```

8K x 2 benchmark:

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

Or:

```bash
bash run_validation.sh
```

See `docs/NO_SYNC_VALIDATION_PLAN.md` for the full acceptance criteria and
results table. For the tiled indices and non-multiple-block regression fix,
fill out `docs/TILED_FIX_TEST_REPORT.md` on an Ascend NPU machine.

## Tiled Compact Variant

`uniprefill_fixed_topk_compact_tiled_out` is the performance-oriented variant.
It launches separate select kernels for `kept_block_indices` and
`kept_block_mask`, then launches a copy kernel over `kept_block x hidden_tile`.
The select split is intentional: remote validation showed bisheng DSE can remove
non-final GM stores, so each metadata output is written as the final side effect
of its own kernel. The tiled copy gives the 8K hidden-state copy much more
parallelism than the scalar correctness MVP, which used one core per request.

Use `--variant tiled --hidden-tile 256` in the validation script to test it.

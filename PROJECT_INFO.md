# Project Information

## Name

`ascendc_uniprefill`

## Purpose

Standalone AscendC operator project for UniPrefill block top-p token selection.

## Canonical Location

```text
/autodl-fs/data/yjb/ascendc_uniprefill
```

This directory is intentionally placed next to:

```text
/autodl-fs/data/yjb/UniPrefill
```

## Source Context

The operator is derived from the UniPrefill top-p selection flow originally
implemented in:

```text
/autodl-fs/data/yjb/UniPrefill/vllm-releases-v0.16.0/vllm/model_executor/layers/fused_top_p_selection_tp_pd.py
```

This project is the AscendC migration target and does not depend on Triton.

## Implemented Operator

Registered PyTorch op:

```text
torch.ops.npu.topselection_top_p_mask
```

Compiled library:

```text
build/libtopselection_ops.so
```

## Main Files

```text
op_kernel/topselection_top_p_kernel.asc
op_kernel/topselection_expand_mask_kernel.asc
op_kernel/topselection_tiling.h
op_extension/topselection_torch.cpp
op_extension/register.cpp
op_extension/ops.h
scripts/golden.py
scripts/test_torch.py
CMakeLists.txt
run.sh
```

## Dependencies

Required:

- CANN/AscendC toolchain
- `ASCEND_HOME_PATH` set
- Python 3
- PyTorch
- torch_npu
- CMake

Expected NPU architecture in `CMakeLists.txt`:

```text
dav-2201
```

If running on a different Ascend generation, update:

```cmake
$<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
```

## Build Command

```bash
cd /autodl-fs/data/yjb/ascendc_uniprefill
bash run.sh
```

## Current Status

The project contains the first AscendC MVP implementation and test harness.
The next required step is to compile and run it on the target NPU environment,
then fix any compiler/runtime issues.

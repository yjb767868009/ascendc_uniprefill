# AscendC TopSelection Operator

This is an independent AscendC prototype project for UniPrefill token selection.
It lives next to the `UniPrefill` repository:

```text
/autodl-fs/data/yjb/
├── UniPrefill/
└── ascendc_uniprefill/
```

The current implementation is an AscendC MVP for:

```text
block_scores -> top-p block mask -> token mask
```

It does not use Triton. The implementation is written as AscendC `.asc` kernels
and exposed to Python through a `torch.ops.npu.*` extension.

## Current Scope

Implemented:

- `topselection_top_p_kernel`: selects important blocks from `block_scores`.
- `topselection_expand_mask_kernel`: expands selected blocks to a per-token mask.
- PyTorch extension registration: `torch.ops.npu.topselection_top_p_mask`.
- PyTorch golden reference and test script.

Not implemented yet:

- `q/k -> attention scores`
- softmax over attention scores
- token/head importance reduction into `block_scores`
- direct vLLM integration

See `PLAN.md` for the full migration plan.

The dynamic hidden-state update problem after top-p selection is discussed in
`PLAN.md` under `Sparse Hidden-State Update Designs`. It compares a block-level
compact operator with an exact token-level prefix-sum compact operator.

## Directory Layout

```text
ascendc_uniprefill/
├── CMakeLists.txt
├── PLAN.md
├── PROJECT_INFO.md
├── README.md
├── run.sh
├── op_kernel/
│   ├── topselection_tiling.h
│   ├── topselection_top_p_kernel.asc
│   └── topselection_expand_mask_kernel.asc
├── op_extension/
│   ├── ops.h
│   ├── register.cpp
│   └── topselection_torch.cpp
└── scripts/
    ├── golden.py
    └── test_torch.py
```

## Build And Run

```bash
cd /autodl-fs/data/yjb/ascendc_uniprefill
bash run.sh
```

`run.sh` performs:

```bash
source "$ASCEND_HOME_PATH/set_env.sh"
cmake -S . -B build
cmake --build build -j4
python3 scripts/test_torch.py
```

Manual execution:

```bash
cd /autodl-fs/data/yjb/ascendc_uniprefill
source "$ASCEND_HOME_PATH/set_env.sh"
cmake -S . -B build
cmake --build build -j4
python3 scripts/test_torch.py
```

The compiled shared library is:

```text
/autodl-fs/data/yjb/ascendc_uniprefill/build/libtopselection_ops.so
```

## Python Usage

```python
import torch
import torch_npu

torch.ops.load_library(
    "/autodl-fs/data/yjb/ascendc_uniprefill/build/libtopselection_ops.so"
)

block_mask, token_mask, new_cu_seqlens, new_max_seq_len = (
    torch.ops.npu.topselection_top_p_mask(
        block_scores.npu(),       # fp32, shape [total_blocks]
        cu_seqlens.npu(),         # int32, shape [batch + 1]
        cu_block_seqlens.npu(),   # int32, shape [batch + 1]
        block_size,
        attention_sink,
        last_q,
        p,
    )
)
```

## API

```python
torch.ops.npu.topselection_top_p_mask(
    block_scores: Tensor,
    cu_seqlens: Tensor,
    cu_block_seqlens: Tensor,
    block_size: int,
    attention_sink: int,
    last_q: int,
    p: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]
```

Inputs:

- `block_scores`: fp32 NPU tensor, shape `[total_blocks]`.
- `cu_seqlens`: int32 NPU tensor, shape `[batch + 1]`.
- `cu_block_seqlens`: int32 NPU tensor, shape `[batch + 1]`.
- `block_size`: token count per block.
- `attention_sink`: number of prefix tokens forced to keep.
- `last_q`: number of tail tokens forced to keep.
- `p`: top-p threshold in `(0, 1]`.

Outputs:

- `block_mask`: uint8 NPU tensor, shape `[total_blocks]`.
- `token_mask`: uint8 NPU tensor, shape `[total_tokens]`.
- `new_cu_seqlens`: int32 NPU tensor, shape `[batch + 1]`.
- `new_max_seq_len`: int32 scalar NPU tensor.

## Notes

- The MVP supports fp32 `block_scores` only.
- `uint8` masks are used at the extension boundary; Python tests compare them
  against boolean golden outputs.
- The current kernels favor clarity and debuggability. They still use scalar
  `GlobalTensor::GetValue/SetValue` patterns and should be optimized after
  correctness is stable.

#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("uniprefill_fixed_topk_compact_out(Tensor hidden_states, Tensor residual, Tensor positions, Tensor slot_mapping, Tensor block_scores, Tensor cu_seqlens, Tensor cu_block_seqlens, Tensor real_cu_seqlens, Tensor keep_middle_blocks, Tensor(a!) hidden_out, Tensor(b!) residual_out, Tensor(c!) positions_out, Tensor(d!) slot_mapping_out, Tensor(e!) kept_block_mask, int block_size, int attention_sink, int last_q) -> ()");
    m.def("uniprefill_fixed_topk_compact_tiled_out(Tensor hidden_states, Tensor residual, Tensor positions, Tensor slot_mapping, Tensor block_scores, Tensor cu_seqlens, Tensor cu_block_seqlens, Tensor kept_block_cu_seqlens, Tensor real_cu_seqlens, Tensor keep_middle_blocks, Tensor(a!) hidden_out, Tensor(b!) residual_out, Tensor(c!) positions_out, Tensor(d!) slot_mapping_out, Tensor(e!) kept_block_mask, Tensor(f!) kept_block_indices, int block_size, int attention_sink, int last_q, int hidden_tile) -> ()");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("uniprefill_fixed_topk_compact_out", TORCH_FN(ascend_kernel::uniprefill_fixed_topk_compact_out_torch));
    m.impl("uniprefill_fixed_topk_compact_tiled_out", TORCH_FN(ascend_kernel::uniprefill_fixed_topk_compact_tiled_out_torch));
}

} // namespace

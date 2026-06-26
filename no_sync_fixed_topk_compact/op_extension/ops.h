#ifndef UNIPREFILL_NO_SYNC_OPS_H
#define UNIPREFILL_NO_SYNC_OPS_H

#include <torch/extension.h>

namespace ascend_kernel {

void uniprefill_fixed_topk_compact_out_torch(
    const at::Tensor& hidden_states,
    const at::Tensor& residual,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& block_scores,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_block_seqlens,
    const at::Tensor& real_cu_seqlens,
    const at::Tensor& keep_middle_blocks,
    at::Tensor& hidden_out,
    at::Tensor& residual_out,
    at::Tensor& positions_out,
    at::Tensor& slot_mapping_out,
    at::Tensor& kept_block_mask,
    int64_t block_size,
    int64_t attention_sink,
    int64_t last_q);

void uniprefill_fixed_topk_compact_tiled_out_torch(
    const at::Tensor& hidden_states,
    const at::Tensor& residual,
    const at::Tensor& positions,
    const at::Tensor& slot_mapping,
    const at::Tensor& block_scores,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_block_seqlens,
    const at::Tensor& kept_block_cu_seqlens,
    const at::Tensor& real_cu_seqlens,
    const at::Tensor& keep_middle_blocks,
    at::Tensor& hidden_out,
    at::Tensor& residual_out,
    at::Tensor& positions_out,
    at::Tensor& slot_mapping_out,
    at::Tensor& kept_block_mask,
    at::Tensor& kept_block_indices,
    int64_t block_size,
    int64_t attention_sink,
    int64_t last_q,
    int64_t hidden_tile);

} // namespace ascend_kernel

#endif // UNIPREFILL_NO_SYNC_OPS_H

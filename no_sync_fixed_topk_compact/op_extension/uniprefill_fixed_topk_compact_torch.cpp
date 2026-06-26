#include <cstdint>

#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "../op_kernel/no_sync_tiling.h"
#include "ops.h"

void uniprefill_fixed_topk_compact_kernel(
    uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *hiddenStates, uint8_t *residual, uint8_t *positions, uint8_t *slotMapping,
    uint8_t *blockScores, uint8_t *cuSeqlens, uint8_t *cuBlockSeqlens,
    uint8_t *realCuSeqlens, uint8_t *keepMiddleBlocks, uint8_t *hiddenOut,
    uint8_t *residualOut, uint8_t *positionsOut, uint8_t *slotMappingOut,
    uint8_t *keptBlockMask, uint8_t *tiling);

void uniprefill_fixed_topk_select_indices_kernel(
    uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *blockScores, uint8_t *cuSeqlens, uint8_t *cuBlockSeqlens,
    uint8_t *keptBlockCuSeqlens, uint8_t *keepMiddleBlocks,
    uint8_t *keptBlockIndices, uint8_t *tiling);

void uniprefill_fixed_topk_write_mask_kernel(
    uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *blockScores, uint8_t *cuSeqlens, uint8_t *cuBlockSeqlens,
    uint8_t *keepMiddleBlocks, uint8_t *keptBlockMask, uint8_t *tiling);

void uniprefill_fixed_topk_compact_copy_tiled_kernel(
    uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
    uint8_t *hiddenStates, uint8_t *residual, uint8_t *positions, uint8_t *slotMapping,
    uint8_t *cuSeqlens, uint8_t *keptBlockCuSeqlens, uint8_t *realCuSeqlens,
    uint8_t *keptBlockIndices, uint8_t *hiddenOut, uint8_t *residualOut,
    uint8_t *positionsOut, uint8_t *slotMappingOut, uint8_t *tiling);

namespace ascend_kernel {
namespace {

at::Tensor copy_tiling_to_device(const void* tiling, size_t size, const at::TensorOptions& options)
{
    at::Tensor tiling_tensor = at::empty({static_cast<int64_t>(size)}, options.dtype(at::kByte));
    auto ret = aclrtMemcpy(tiling_tensor.mutable_data_ptr(), size, tiling, size, ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(ret == ACL_SUCCESS, "failed to copy tiling data to device");
    return tiling_tensor;
}

} // namespace

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
    int64_t last_q)
{
    TORCH_CHECK(hidden_states.is_privateuseone(), "hidden_states must be on NPU");
    TORCH_CHECK(residual.is_privateuseone(), "residual must be on NPU");
    TORCH_CHECK(positions.is_privateuseone(), "positions must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(block_scores.is_privateuseone(), "block_scores must be on NPU");
    TORCH_CHECK(cu_seqlens.is_privateuseone(), "cu_seqlens must be on NPU");
    TORCH_CHECK(cu_block_seqlens.is_privateuseone(), "cu_block_seqlens must be on NPU");
    TORCH_CHECK(real_cu_seqlens.is_privateuseone(), "real_cu_seqlens must be on NPU");
    TORCH_CHECK(keep_middle_blocks.is_privateuseone(), "keep_middle_blocks must be on NPU");
    TORCH_CHECK(hidden_out.is_privateuseone(), "hidden_out must be on NPU");
    TORCH_CHECK(residual_out.is_privateuseone(), "residual_out must be on NPU");
    TORCH_CHECK(positions_out.is_privateuseone(), "positions_out must be on NPU");
    TORCH_CHECK(slot_mapping_out.is_privateuseone(), "slot_mapping_out must be on NPU");
    TORCH_CHECK(kept_block_mask.is_privateuseone(), "kept_block_mask must be on NPU");

    TORCH_CHECK(hidden_states.scalar_type() == at::kFloat, "MVP supports fp32 hidden_states only");
    TORCH_CHECK(residual.scalar_type() == at::kFloat, "MVP supports fp32 residual only");
    TORCH_CHECK(block_scores.scalar_type() == at::kFloat, "MVP supports fp32 block_scores only");
    TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(cu_block_seqlens.scalar_type() == at::kInt, "cu_block_seqlens must be int32");
    TORCH_CHECK(real_cu_seqlens.scalar_type() == at::kInt, "real_cu_seqlens must be int32");
    TORCH_CHECK(keep_middle_blocks.scalar_type() == at::kInt, "keep_middle_blocks must be int32");
    TORCH_CHECK(hidden_out.scalar_type() == at::kFloat, "hidden_out must be fp32");
    TORCH_CHECK(residual_out.scalar_type() == at::kFloat, "residual_out must be fp32");
    TORCH_CHECK(positions_out.scalar_type() == at::kLong, "positions_out must be int64");
    TORCH_CHECK(slot_mapping_out.scalar_type() == at::kInt, "slot_mapping_out must be int32");
    TORCH_CHECK(kept_block_mask.scalar_type() == at::kByte, "kept_block_mask must be uint8");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [T, H]");
    TORCH_CHECK(residual.sizes() == hidden_states.sizes(), "residual must match hidden_states");
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) == hidden_states.size(0), "positions must be [T]");
    TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == hidden_states.size(0), "slot_mapping must be [T]");
    TORCH_CHECK(hidden_out.dim() == 2 && hidden_out.size(1) == hidden_states.size(1), "hidden_out must be [T_out, H]");
    TORCH_CHECK(residual_out.sizes() == hidden_out.sizes(), "residual_out must match hidden_out");
    TORCH_CHECK(positions_out.dim() == 1 && positions_out.size(0) == hidden_out.size(0), "positions_out must match T_out");
    TORCH_CHECK(slot_mapping_out.dim() == 1 && slot_mapping_out.size(0) == hidden_out.size(0), "slot_mapping_out must match T_out");
    TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be 1D");
    TORCH_CHECK(cu_block_seqlens.dim() == 1, "cu_block_seqlens must be 1D");
    TORCH_CHECK(real_cu_seqlens.dim() == 1, "real_cu_seqlens must be 1D");
    TORCH_CHECK(keep_middle_blocks.dim() == 1, "keep_middle_blocks must be 1D");
    TORCH_CHECK(cu_seqlens.numel() == cu_block_seqlens.numel(), "cu_seqlens/cu_block_seqlens length mismatch");
    TORCH_CHECK(cu_seqlens.numel() == real_cu_seqlens.numel(), "cu_seqlens/real_cu_seqlens length mismatch");
    TORCH_CHECK(keep_middle_blocks.numel() + 1 == cu_seqlens.numel(), "keep_middle_blocks must be [batch]");
    TORCH_CHECK(kept_block_mask.numel() == block_scores.numel(), "kept_block_mask must match block_scores");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(attention_sink >= 0, "attention_sink must be non-negative");
    TORCH_CHECK(last_q >= 0, "last_q must be non-negative");

    int64_t batch_i64 = keep_middle_blocks.numel();
    TORCH_CHECK(batch_i64 <= static_cast<int64_t>(UINT32_MAX), "batch too large");
    TORCH_CHECK(hidden_states.size(1) <= static_cast<int64_t>(UINT32_MAX), "hidden size too large");
    uint32_t batch = static_cast<uint32_t>(batch_i64);

    UniPrefillFixedTopKCompactTilingData tiling;
    tiling.batch = batch;
    tiling.hiddenSize = static_cast<uint32_t>(hidden_states.size(1));
    tiling.blockSize = static_cast<uint32_t>(block_size);
    tiling.attentionSink = static_cast<uint32_t>(attention_sink);
    tiling.lastQ = static_cast<uint32_t>(last_q);
    at::Tensor tiling_tensor = copy_tiling_to_device(&tiling, sizeof(tiling), hidden_states.options());

    if (batch == 0) {
        return;
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);
    uniprefill_fixed_topk_compact_kernel(batch, nullptr, acl_stream,
        reinterpret_cast<uint8_t*>(hidden_states.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(residual.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(positions.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(slot_mapping.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(block_scores.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(cu_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(cu_block_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(real_cu_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(keep_middle_blocks.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(hidden_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(residual_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(positions_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(slot_mapping_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kept_block_mask.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tiling_tensor.mutable_data_ptr()));
}


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
    int64_t hidden_tile)
{
    TORCH_CHECK(hidden_states.is_privateuseone(), "hidden_states must be on NPU");
    TORCH_CHECK(residual.is_privateuseone(), "residual must be on NPU");
    TORCH_CHECK(positions.is_privateuseone(), "positions must be on NPU");
    TORCH_CHECK(slot_mapping.is_privateuseone(), "slot_mapping must be on NPU");
    TORCH_CHECK(block_scores.is_privateuseone(), "block_scores must be on NPU");
    TORCH_CHECK(cu_seqlens.is_privateuseone(), "cu_seqlens must be on NPU");
    TORCH_CHECK(cu_block_seqlens.is_privateuseone(), "cu_block_seqlens must be on NPU");
    TORCH_CHECK(kept_block_cu_seqlens.is_privateuseone(), "kept_block_cu_seqlens must be on NPU");
    TORCH_CHECK(real_cu_seqlens.is_privateuseone(), "real_cu_seqlens must be on NPU");
    TORCH_CHECK(keep_middle_blocks.is_privateuseone(), "keep_middle_blocks must be on NPU");
    TORCH_CHECK(hidden_out.is_privateuseone(), "hidden_out must be on NPU");
    TORCH_CHECK(residual_out.is_privateuseone(), "residual_out must be on NPU");
    TORCH_CHECK(positions_out.is_privateuseone(), "positions_out must be on NPU");
    TORCH_CHECK(slot_mapping_out.is_privateuseone(), "slot_mapping_out must be on NPU");
    TORCH_CHECK(kept_block_mask.is_privateuseone(), "kept_block_mask must be on NPU");
    TORCH_CHECK(kept_block_indices.is_privateuseone(), "kept_block_indices must be on NPU");

    TORCH_CHECK(hidden_states.scalar_type() == at::kFloat, "MVP supports fp32 hidden_states only");
    TORCH_CHECK(residual.scalar_type() == at::kFloat, "MVP supports fp32 residual only");
    TORCH_CHECK(block_scores.scalar_type() == at::kFloat, "MVP supports fp32 block_scores only");
    TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kInt, "slot_mapping must be int32");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(cu_block_seqlens.scalar_type() == at::kInt, "cu_block_seqlens must be int32");
    TORCH_CHECK(kept_block_cu_seqlens.scalar_type() == at::kInt, "kept_block_cu_seqlens must be int32");
    TORCH_CHECK(real_cu_seqlens.scalar_type() == at::kInt, "real_cu_seqlens must be int32");
    TORCH_CHECK(keep_middle_blocks.scalar_type() == at::kInt, "keep_middle_blocks must be int32");
    TORCH_CHECK(hidden_out.scalar_type() == at::kFloat, "hidden_out must be fp32");
    TORCH_CHECK(residual_out.scalar_type() == at::kFloat, "residual_out must be fp32");
    TORCH_CHECK(positions_out.scalar_type() == at::kLong, "positions_out must be int64");
    TORCH_CHECK(slot_mapping_out.scalar_type() == at::kInt, "slot_mapping_out must be int32");
    TORCH_CHECK(kept_block_mask.scalar_type() == at::kByte, "kept_block_mask must be uint8");
    TORCH_CHECK(kept_block_indices.scalar_type() == at::kInt, "kept_block_indices must be int32");

    TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be [T, H]");
    TORCH_CHECK(residual.sizes() == hidden_states.sizes(), "residual must match hidden_states");
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) == hidden_states.size(0), "positions must be [T]");
    TORCH_CHECK(slot_mapping.dim() == 1 && slot_mapping.size(0) == hidden_states.size(0), "slot_mapping must be [T]");
    TORCH_CHECK(hidden_out.dim() == 2 && hidden_out.size(1) == hidden_states.size(1), "hidden_out must be [T_out, H]");
    TORCH_CHECK(residual_out.sizes() == hidden_out.sizes(), "residual_out must match hidden_out");
    TORCH_CHECK(positions_out.dim() == 1 && positions_out.size(0) == hidden_out.size(0), "positions_out must match T_out");
    TORCH_CHECK(slot_mapping_out.dim() == 1 && slot_mapping_out.size(0) == hidden_out.size(0), "slot_mapping_out must match T_out");
    TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be 1D");
    TORCH_CHECK(cu_block_seqlens.dim() == 1, "cu_block_seqlens must be 1D");
    TORCH_CHECK(kept_block_cu_seqlens.dim() == 1, "kept_block_cu_seqlens must be 1D");
    TORCH_CHECK(real_cu_seqlens.dim() == 1, "real_cu_seqlens must be 1D");
    TORCH_CHECK(keep_middle_blocks.dim() == 1, "keep_middle_blocks must be 1D");
    TORCH_CHECK(cu_seqlens.numel() == cu_block_seqlens.numel(), "cu_seqlens/cu_block_seqlens length mismatch");
    TORCH_CHECK(cu_seqlens.numel() == real_cu_seqlens.numel(), "cu_seqlens/real_cu_seqlens length mismatch");
    TORCH_CHECK(cu_seqlens.numel() == kept_block_cu_seqlens.numel(), "cu_seqlens/kept_block_cu_seqlens length mismatch");
    TORCH_CHECK(keep_middle_blocks.numel() + 1 == cu_seqlens.numel(), "keep_middle_blocks must be [batch]");
    TORCH_CHECK(kept_block_mask.numel() == block_scores.numel(), "kept_block_mask must match block_scores");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(attention_sink >= 0, "attention_sink must be non-negative");
    TORCH_CHECK(last_q >= 0, "last_q must be non-negative");
    TORCH_CHECK(hidden_tile > 0, "hidden_tile must be positive");

    int64_t batch_i64 = keep_middle_blocks.numel();
    TORCH_CHECK(batch_i64 <= static_cast<int64_t>(UINT32_MAX), "batch too large");
    TORCH_CHECK(hidden_states.size(1) <= static_cast<int64_t>(UINT32_MAX), "hidden size too large");
    TORCH_CHECK(block_scores.numel() <= static_cast<int64_t>(UINT32_MAX), "too many blocks");
    TORCH_CHECK(kept_block_indices.numel() <= static_cast<int64_t>(UINT32_MAX), "too many kept blocks");
    uint32_t batch = static_cast<uint32_t>(batch_i64);
    uint32_t hidden_size = static_cast<uint32_t>(hidden_states.size(1));
    uint32_t hidden_tile_u32 = static_cast<uint32_t>(hidden_tile);
    uint32_t hidden_tile_count = static_cast<uint32_t>((hidden_states.size(1) + hidden_tile - 1) / hidden_tile);
    uint32_t total_blocks = static_cast<uint32_t>(block_scores.numel());
    uint32_t total_kept_blocks = static_cast<uint32_t>(kept_block_indices.numel());

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

    UniPrefillFixedTopKTiledSelectTilingData select_tiling;
    select_tiling.batch = batch;
    select_tiling.blockSize = static_cast<uint32_t>(block_size);
    select_tiling.attentionSink = static_cast<uint32_t>(attention_sink);
    select_tiling.lastQ = static_cast<uint32_t>(last_q);
    at::Tensor select_tiling_tensor = copy_tiling_to_device(&select_tiling, sizeof(select_tiling), hidden_states.options());

    UniPrefillFixedTopKTiledCopyTilingData copy_tiling;
    copy_tiling.batch = batch;
    copy_tiling.hiddenSize = hidden_size;
    copy_tiling.blockSize = static_cast<uint32_t>(block_size);
    copy_tiling.hiddenTile = hidden_tile_u32;
    copy_tiling.hiddenTileCount = hidden_tile_count;
    at::Tensor copy_tiling_tensor = copy_tiling_to_device(&copy_tiling, sizeof(copy_tiling), hidden_states.options());

    if (batch == 0 || total_blocks == 0) {
        return;
    }

    uniprefill_fixed_topk_select_indices_kernel(total_blocks, nullptr, acl_stream,
        reinterpret_cast<uint8_t*>(block_scores.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(cu_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(cu_block_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kept_block_cu_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(keep_middle_blocks.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kept_block_indices.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(select_tiling_tensor.mutable_data_ptr()));

    uniprefill_fixed_topk_write_mask_kernel(total_blocks, nullptr, acl_stream,
        reinterpret_cast<uint8_t*>(block_scores.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(cu_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(cu_block_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(keep_middle_blocks.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kept_block_mask.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(select_tiling_tensor.mutable_data_ptr()));

    if (total_kept_blocks == 0) {
        return;
    }

    uint32_t copy_block_dim = total_kept_blocks * hidden_tile_count;
    uniprefill_fixed_topk_compact_copy_tiled_kernel(copy_block_dim, nullptr, acl_stream,
        reinterpret_cast<uint8_t*>(hidden_states.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(residual.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(positions.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(slot_mapping.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(cu_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kept_block_cu_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(real_cu_seqlens.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kept_block_indices.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(hidden_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(residual_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(positions_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(slot_mapping_out.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(copy_tiling_tensor.mutable_data_ptr()));
}

} // namespace ascend_kernel

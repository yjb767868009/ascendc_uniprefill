#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "acl/acl.h"
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "../op_kernel/topselection_tiling.h"

extern "C" void topselection_top_p_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                           uint8_t *blockScores, uint8_t *cuBlockSeqlens,
                                           uint8_t *blockMask, uint8_t *tiling);

extern "C" void topselection_expand_mask_kernel(uint32_t blockDim, void *l2Ctrl, aclrtStream stream,
                                                uint8_t *blockMask, uint8_t *cuSeqlens,
                                                uint8_t *cuBlockSeqlens, uint8_t *tokenMask,
                                                uint8_t *newSeqLens, uint8_t *tiling);

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

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> topselection_top_p_mask_torch(
    const at::Tensor& block_scores,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_block_seqlens,
    int64_t block_size,
    int64_t attention_sink,
    int64_t last_q,
    double p)
{
    TORCH_CHECK(block_scores.is_privateuseone(), "block_scores must be on NPU");
    TORCH_CHECK(cu_seqlens.is_privateuseone(), "cu_seqlens must be on NPU");
    TORCH_CHECK(cu_block_seqlens.is_privateuseone(), "cu_block_seqlens must be on NPU");
    TORCH_CHECK(block_scores.scalar_type() == at::kFloat, "MVP supports fp32 block_scores only");
    TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
    TORCH_CHECK(cu_block_seqlens.scalar_type() == at::kInt, "cu_block_seqlens must be int32");
    TORCH_CHECK(cu_seqlens.dim() == 1, "cu_seqlens must be 1D");
    TORCH_CHECK(cu_block_seqlens.dim() == 1, "cu_block_seqlens must be 1D");
    TORCH_CHECK(cu_seqlens.numel() == cu_block_seqlens.numel(),
                "cu_seqlens and cu_block_seqlens must have the same length");
    TORCH_CHECK(block_size > 0, "block_size must be positive");
    TORCH_CHECK(attention_sink >= 0, "attention_sink must be non-negative");
    TORCH_CHECK(last_q >= 0, "last_q must be non-negative");
    TORCH_CHECK(p > 0.0 && p <= 1.0, "p must be in (0, 1]");

    int64_t batch_i64 = cu_seqlens.numel() - 1;
    TORCH_CHECK(batch_i64 >= 0, "invalid cu_seqlens");
    TORCH_CHECK(batch_i64 <= static_cast<int64_t>(UINT32_MAX), "batch too large");
    uint32_t batch = static_cast<uint32_t>(batch_i64);

    at::Tensor cu_seqlens_cpu = cu_seqlens.cpu();
    at::Tensor cu_block_seqlens_cpu = cu_block_seqlens.cpu();
    const int32_t* cu_seq_ptr = cu_seqlens_cpu.data_ptr<int32_t>();
    const int32_t* cu_block_ptr = cu_block_seqlens_cpu.data_ptr<int32_t>();

    int32_t total_tokens = cu_seq_ptr[batch];
    int32_t total_blocks = cu_block_ptr[batch];
    TORCH_CHECK(total_tokens >= 0, "total_tokens must be non-negative");
    TORCH_CHECK(total_blocks == block_scores.numel(), "block_scores length must match cu_block_seqlens[-1]");

    uint32_t max_seq_len = 0;
    uint32_t max_block_len = 0;
    for (uint32_t i = 0; i < batch; ++i) {
        int32_t seq_len = cu_seq_ptr[i + 1] - cu_seq_ptr[i];
        int32_t block_len = cu_block_ptr[i + 1] - cu_block_ptr[i];
        TORCH_CHECK(seq_len >= 0, "cu_seqlens must be non-decreasing");
        TORCH_CHECK(block_len >= 0, "cu_block_seqlens must be non-decreasing");
        max_seq_len = std::max(max_seq_len, static_cast<uint32_t>(seq_len));
        max_block_len = std::max(max_block_len, static_cast<uint32_t>(block_len));
    }
    TORCH_CHECK(max_block_len <= TOPSELECTION_MAX_BLOCKS,
                "MVP max block length exceeded: ", max_block_len);
    TORCH_CHECK(max_seq_len <= TOPSELECTION_MAX_TOKENS_PER_REQ,
                "MVP max sequence length exceeded: ", max_seq_len);

    auto byte_options = block_scores.options().dtype(at::kByte);
    at::Tensor block_mask = at::empty({total_blocks}, byte_options);
    at::Tensor token_mask = at::empty({total_tokens}, byte_options);
    at::Tensor new_seq_lens = at::empty({static_cast<int64_t>(batch)}, cu_seqlens.options());
    at::Tensor new_cu_seqlens = at::empty_like(cu_seqlens);
    at::Tensor new_max_seq_len = at::empty({}, cu_seqlens.options());

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

    TopSelectionTopPTilingData top_p_tiling;
    top_p_tiling.batch = batch;
    top_p_tiling.maxBlockLen = max_block_len;
    top_p_tiling.p = static_cast<float>(p);
    at::Tensor top_p_tiling_tensor = copy_tiling_to_device(
        &top_p_tiling, sizeof(top_p_tiling), block_scores.options());

    if (batch > 0) {
        topselection_top_p_kernel(batch, nullptr, acl_stream,
            reinterpret_cast<uint8_t*>(block_scores.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(cu_block_seqlens.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(block_mask.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(top_p_tiling_tensor.mutable_data_ptr()));

        TopSelectionExpandMaskTilingData expand_tiling;
        expand_tiling.batch = batch;
        expand_tiling.maxSeqLen = max_seq_len;
        expand_tiling.blockSize = static_cast<uint32_t>(block_size);
        expand_tiling.attentionSink = static_cast<uint32_t>(attention_sink);
        expand_tiling.lastQ = static_cast<uint32_t>(last_q);
        at::Tensor expand_tiling_tensor = copy_tiling_to_device(
            &expand_tiling, sizeof(expand_tiling), block_scores.options());

        topselection_expand_mask_kernel(batch, nullptr, acl_stream,
            reinterpret_cast<uint8_t*>(block_mask.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(cu_seqlens.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(cu_block_seqlens.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(token_mask.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(new_seq_lens.mutable_data_ptr()),
            reinterpret_cast<uint8_t*>(expand_tiling_tensor.mutable_data_ptr()));
    }

    at::Tensor new_seq_lens_cpu = new_seq_lens.cpu();
    std::vector<int32_t> new_cu_host(batch + 1, 0);
    int32_t max_new_len = 0;
    const int32_t* new_seq_ptr = new_seq_lens_cpu.data_ptr<int32_t>();
    for (uint32_t i = 0; i < batch; ++i) {
        new_cu_host[i + 1] = new_cu_host[i] + new_seq_ptr[i];
        max_new_len = std::max(max_new_len, new_seq_ptr[i]);
    }

    auto ret = aclrtMemcpy(new_cu_seqlens.mutable_data_ptr(), new_cu_host.size() * sizeof(int32_t),
                           new_cu_host.data(), new_cu_host.size() * sizeof(int32_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(ret == ACL_SUCCESS, "failed to copy new_cu_seqlens to device");
    ret = aclrtMemcpy(new_max_seq_len.mutable_data_ptr(), sizeof(int32_t),
                      &max_new_len, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    TORCH_CHECK(ret == ACL_SUCCESS, "failed to copy new_max_seq_len to device");

    return std::make_tuple(block_mask, token_mask, new_cu_seqlens, new_max_seq_len);
}

} // namespace ascend_kernel


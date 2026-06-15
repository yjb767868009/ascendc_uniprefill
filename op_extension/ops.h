#ifndef TOPSELECTION_OPS_H
#define TOPSELECTION_OPS_H

#include <tuple>
#include <torch/extension.h>

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> topselection_top_p_mask_torch(
    const at::Tensor& block_scores,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_block_seqlens,
    int64_t block_size,
    int64_t attention_sink,
    int64_t last_q,
    double p);

} // namespace ascend_kernel

#endif // TOPSELECTION_OPS_H


#include <tuple>
#include <torch/extension.h>
#include <torch/library.h>
#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m)
{
    m.def("topselection_top_p_mask(Tensor block_scores, Tensor cu_seqlens, Tensor cu_block_seqlens, int block_size, int attention_sink, int last_q, float p) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m)
{
    m.impl("topselection_top_p_mask", TORCH_FN(ascend_kernel::topselection_top_p_mask_torch));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> topselection_top_p_mask_meta(
    const at::Tensor& block_scores,
    const at::Tensor& cu_seqlens,
    const at::Tensor& cu_block_seqlens,
    int64_t block_size,
    int64_t attention_sink,
    int64_t last_q,
    double p)
{
    (void)cu_block_seqlens;
    (void)block_size;
    (void)attention_sink;
    (void)last_q;
    (void)p;

    auto token_count = cu_seqlens.sym_size(0) > 0 ? cu_seqlens.sym_size(0) : 0;
    at::Tensor block_mask = at::empty_like(block_scores, block_scores.options().dtype(at::kByte));
    at::Tensor token_mask = at::empty({0}, block_scores.options().dtype(at::kByte));
    at::Tensor new_cu_seqlens = at::empty_like(cu_seqlens);
    at::Tensor new_max_seq_len = at::empty({}, cu_seqlens.options());
    (void)token_count;
    return std::make_tuple(block_mask, token_mask, new_cu_seqlens, new_max_seq_len);
}

TORCH_LIBRARY_IMPL(npu, Meta, m)
{
    m.impl("topselection_top_p_mask", &topselection_top_p_mask_meta);
}

} // namespace


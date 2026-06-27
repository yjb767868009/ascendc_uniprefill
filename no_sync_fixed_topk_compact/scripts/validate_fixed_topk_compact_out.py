from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass

import torch

try:
    import torch_npu  # type: ignore
except ImportError:  # pragma: no cover - only available on Ascend machines.
    torch_npu = None


SO_NAME = "libuniprefill_no_sync_ops.so"
OP_NAME = "uniprefill_fixed_topk_compact_out"
TILED_OP_NAME = "uniprefill_fixed_topk_compact_tiled_out"


@dataclass(frozen=True)
class HostPlan:
    seq_lens: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_block_seqlens: torch.Tensor
    keep_middle_blocks: torch.Tensor
    kept_block_cu_seqlens: torch.Tensor
    real_cu_seqlens: torch.Tensor
    total_tokens: int
    total_blocks: int
    total_real_tokens: int


def parse_seq_lens(raw: str) -> list[int]:
    out = [int(x) for x in raw.split(",") if x.strip()]
    if not out or any(x <= 0 for x in out):
        raise ValueError("--seq-lens must contain positive integers")
    return out


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def classify_blocks(seq_len: int, block_size: int, attention_sink: int, last_q: int) -> tuple[list[bool], list[bool], list[bool]]:
    num_blocks = ceil_div(seq_len, block_size)
    sink = [False] * num_blocks
    tail = [False] * num_blocks
    sink_blocks = ceil_div(min(max(attention_sink, 0), seq_len), block_size) if seq_len > 0 else 0
    for i in range(min(sink_blocks, num_blocks)):
        sink[i] = True
    tail_tokens = min(max(last_q, 0), seq_len)
    if tail_tokens > 0:
        tail_start = max(seq_len - tail_tokens, 0)
        tail_block_start = tail_start // block_size
        for i in range(tail_block_start, num_blocks):
            tail[i] = True
    middle = [not (s or t) for s, t in zip(sink, tail)]
    return sink, middle, tail


def block_real_len(block_idx: int, seq_len: int, block_size: int) -> int:
    lo = block_idx * block_size
    hi = min(lo + block_size, seq_len)
    return max(hi - lo, 0)


def make_host_plan(seq_lens_list: list[int], block_size: int, attention_sink: int, last_q: int, drop_ratio: float) -> HostPlan:
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    cu_seqlens = torch.zeros(len(seq_lens_list) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
    block_lens = torch.div(seq_lens + block_size - 1, block_size, rounding_mode="floor")
    cu_block_seqlens = torch.zeros_like(cu_seqlens)
    cu_block_seqlens[1:] = torch.cumsum(block_lens, dim=0)
    keep_middle = torch.zeros(len(seq_lens_list), dtype=torch.int32)
    kept_block_lens = torch.zeros(len(seq_lens_list), dtype=torch.int32)
    real_lens = torch.zeros(len(seq_lens_list), dtype=torch.int32)
    for req, seq_len in enumerate(seq_lens_list):
        sink, middle, tail = classify_blocks(seq_len, block_size, attention_sink, last_q)
        keep = middle.count(True) - int(middle.count(True) * drop_ratio)
        keep_middle[req] = keep
        forced_blocks = sum(1 for s, t in zip(sink, tail) if s or t)
        kept_block_lens[req] = forced_blocks + keep
        real = 0
        for block_idx, is_sink in enumerate(sink):
            if is_sink:
                real += block_real_len(block_idx, seq_len, block_size)
        for block_idx, is_tail in enumerate(tail):
            if is_tail and not sink[block_idx]:
                real += block_real_len(block_idx, seq_len, block_size)
        real += keep * block_size
        real_lens[req] = real
    kept_block_cu = torch.zeros_like(cu_seqlens)
    kept_block_cu[1:] = torch.cumsum(kept_block_lens, dim=0)
    real_cu = torch.zeros_like(cu_seqlens)
    real_cu[1:] = torch.cumsum(real_lens, dim=0)
    return HostPlan(seq_lens, cu_seqlens, cu_block_seqlens, keep_middle, kept_block_cu, real_cu,
                    int(cu_seqlens[-1]), int(cu_block_seqlens[-1]), int(real_cu[-1]))


def cpu_golden(hidden_states, residual, positions, slot_mapping, block_scores, plan, block_size, attention_sink, last_q):
    hidden_out = torch.empty((plan.total_real_tokens, hidden_states.shape[1]), dtype=hidden_states.dtype)
    residual_out = torch.empty_like(hidden_out)
    positions_out = torch.empty((plan.total_real_tokens,), dtype=positions.dtype)
    slot_mapping_out = torch.empty((plan.total_real_tokens,), dtype=slot_mapping.dtype)
    kept_mask = torch.zeros(plan.total_blocks, dtype=torch.uint8)
    cu_seq = plan.cu_seqlens.tolist()
    cu_blk = plan.cu_block_seqlens.tolist()
    real_cu = plan.real_cu_seqlens.tolist()
    keep_middle = plan.keep_middle_blocks.tolist()
    for req, seq_len in enumerate(plan.seq_lens.tolist()):
        token_start = cu_seq[req]
        block_start = cu_blk[req]
        block_end = cu_blk[req + 1]
        sink, middle, tail = classify_blocks(seq_len, block_size, attention_sink, last_q)
        for local, keep in enumerate(s or t for s, t in zip(sink, tail)):
            if keep:
                kept_mask[block_start + local] = 1
        middle_indices = [i for i, keep in enumerate(middle) if keep]
        if middle_indices and keep_middle[req] > 0:
            idx = torch.tensor(middle_indices, dtype=torch.long)
            scores = block_scores[block_start + idx].float()
            order = torch.argsort(scores, descending=True, stable=True)
            for selected in order[:keep_middle[req]].tolist():
                kept_mask[block_start + middle_indices[selected]] = 1
        write = real_cu[req]
        for local in range(block_end - block_start):
            if kept_mask[block_start + local].item() == 0:
                continue
            real_len = block_real_len(local, seq_len, block_size)
            src = token_start + local * block_size
            hidden_out[write:write + real_len] = hidden_states[src:src + real_len]
            residual_out[write:write + real_len] = residual[src:src + real_len]
            positions_out[write:write + real_len] = positions[src:src + real_len]
            slot_mapping_out[write:write + real_len] = slot_mapping[src:src + real_len]
            write += real_len
    return hidden_out, residual_out, positions_out, slot_mapping_out, kept_mask


def make_inputs(plan: HostPlan, hidden_size: int, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    hidden = torch.randn(plan.total_tokens, hidden_size, generator=g, dtype=torch.float32)
    residual = torch.randn(plan.total_tokens, hidden_size, generator=g, dtype=torch.float32)
    positions = torch.arange(plan.total_tokens, dtype=torch.int64)
    slot_mapping = torch.arange(plan.total_tokens, dtype=torch.int32) + 1000
    scores = torch.rand(plan.total_blocks, generator=g, dtype=torch.float32)
    return hidden, residual, positions, slot_mapping, scores


def require_npu() -> None:
    if torch_npu is None or not hasattr(torch.Tensor, "npu"):
        raise RuntimeError("run this script on an Ascend NPU machine with torch_npu installed")


def sync_npu() -> None:
    torch.npu.synchronize()


def load_op() -> None:
    so_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", SO_NAME)
    if not os.path.exists(so_path):
        raise FileNotFoundError(f"{so_path} not found. Build with cmake first.")
    torch.ops.load_library(so_path)
    getattr(torch.ops.npu, OP_NAME)
    getattr(torch.ops.npu, TILED_OP_NAME)


def run_out_op(tensors_npu, meta_npu, outputs_npu, block_size, attention_sink, last_q) -> None:
    hidden, residual, positions, slot_mapping, scores = tensors_npu
    cu, cu_blk, real_cu, keep_middle = meta_npu
    hidden_out, residual_out, positions_out, slot_mapping_out, kept_mask = outputs_npu
    torch.ops.npu.uniprefill_fixed_topk_compact_out(
        hidden, residual, positions, slot_mapping, scores, cu, cu_blk, real_cu, keep_middle,
        hidden_out, residual_out, positions_out, slot_mapping_out, kept_mask,
        block_size, attention_sink, last_q)


def run_tiled_out_op(tensors_npu, meta_npu, outputs_npu, kept_block_indices, block_size, attention_sink, last_q, hidden_tile) -> None:
    hidden, residual, positions, slot_mapping, scores = tensors_npu
    cu, cu_blk, kept_block_cu, real_cu, keep_middle = meta_npu
    hidden_out, residual_out, positions_out, slot_mapping_out, kept_mask = outputs_npu
    torch.ops.npu.uniprefill_fixed_topk_compact_tiled_out(
        hidden, residual, positions, slot_mapping, scores, cu, cu_blk, kept_block_cu, real_cu, keep_middle,
        hidden_out, residual_out, positions_out, slot_mapping_out, kept_mask, kept_block_indices,
        block_size, attention_sink, last_q, hidden_tile)


def make_kept_block_indices_npu(plan: HostPlan):
    return torch.full((int(plan.kept_block_cu_seqlens[-1]),), -1, device="npu", dtype=torch.int32)


def expected_kept_block_indices(plan: HostPlan, kept_mask: torch.Tensor) -> torch.Tensor:
    indices: list[int] = []
    cu_blk = plan.cu_block_seqlens.tolist()
    for req in range(len(plan.seq_lens)):
        block_start = cu_blk[req]
        block_end = cu_blk[req + 1]
        for local in range(block_end - block_start):
            if kept_mask[block_start + local].item() != 0:
                indices.append(local)
    return torch.tensor(indices, dtype=torch.int32)


def prepare_variant_state(args, plan: HostPlan):
    if args.variant == "scalar":
        meta_npu = (
            plan.cu_seqlens.npu(),
            plan.cu_block_seqlens.npu(),
            plan.real_cu_seqlens.npu(),
            plan.keep_middle_blocks.npu(),
        )
        return meta_npu, None

    meta_npu = (
        plan.cu_seqlens.npu(),
        plan.cu_block_seqlens.npu(),
        plan.kept_block_cu_seqlens.npu(),
        plan.real_cu_seqlens.npu(),
        plan.keep_middle_blocks.npu(),
    )
    kept_block_indices = make_kept_block_indices_npu(plan)
    return meta_npu, kept_block_indices


def run_variant(args, tensors_npu, meta_npu, outputs_npu, kept_block_indices=None):
    if args.variant == "scalar":
        run_out_op(tensors_npu, meta_npu, outputs_npu, args.block_size, args.attention_sink, args.last_q)
        return

    if kept_block_indices is None:
        raise ValueError("kept_block_indices is required for tiled variant")
    run_tiled_out_op(
        tensors_npu, meta_npu, outputs_npu, kept_block_indices,
        args.block_size, args.attention_sink, args.last_q, args.hidden_tile)


def make_outputs_npu(plan: HostPlan, hidden_size: int):
    return (
        torch.empty((plan.total_real_tokens, hidden_size), device="npu", dtype=torch.float32),
        torch.empty((plan.total_real_tokens, hidden_size), device="npu", dtype=torch.float32),
        torch.empty((plan.total_real_tokens,), device="npu", dtype=torch.int64),
        torch.empty((plan.total_real_tokens,), device="npu", dtype=torch.int32),
        torch.empty((plan.total_blocks,), device="npu", dtype=torch.uint8),
    )


def correctness_case(args, name, seq_lens) -> bool:
    plan = make_host_plan(seq_lens, args.block_size, args.attention_sink, args.last_q, args.drop_ratio)
    cpu_inputs = make_inputs(plan, args.hidden_size, args.seed)
    expected = cpu_golden(*cpu_inputs, plan, args.block_size, args.attention_sink, args.last_q)
    tensors_npu = tuple(x.npu() for x in cpu_inputs)
    outputs_npu = make_outputs_npu(plan, args.hidden_size)
    meta_npu, kept_block_indices = prepare_variant_state(args, plan)
    run_variant(args, tensors_npu, meta_npu, outputs_npu, kept_block_indices)
    sync_npu()
    actual = tuple(x.cpu() for x in outputs_npu)
    labels = ["hidden_out", "residual_out", "positions_out", "slot_mapping_out", "kept_block_mask"]
    ok = True
    for label, got, exp in zip(labels, actual, expected):
        same = torch.equal(got, exp) if got.dtype in (torch.int32, torch.int64, torch.uint8) else torch.allclose(got, exp, atol=0, rtol=0)
        print(f"{name}.{label}: {'PASSED' if same else 'FAILED'}")
        ok = ok and same
    if args.variant == "tiled":
        assert kept_block_indices is not None
        got_indices = kept_block_indices.cpu()
        exp_indices = expected_kept_block_indices(plan, expected[-1])
        same = torch.equal(got_indices, exp_indices)
        print(f"{name}.kept_block_indices: {'PASSED' if same else 'FAILED'}")
        if not same:
            print(f"  actual={got_indices}")
            print(f"  expect={exp_indices}")
        ok = ok and same
    print(f"{name}.metadata: total_tokens={plan.total_tokens} total_blocks={plan.total_blocks} total_kept_blocks={int(plan.kept_block_cu_seqlens[-1])} total_real={plan.total_real_tokens} real_cu={plan.real_cu_seqlens.tolist()} kept_block_cu={plan.kept_block_cu_seqlens.tolist()} keep_middle={plan.keep_middle_blocks.tolist()}")
    return ok


def make_expected_token_mask(plan: HostPlan, kept_mask_cpu: torch.Tensor, block_size: int) -> torch.Tensor:
    mask = torch.zeros(plan.total_tokens, dtype=torch.bool)
    cu_seq = plan.cu_seqlens.tolist()
    cu_blk = plan.cu_block_seqlens.tolist()
    for req, seq_len in enumerate(plan.seq_lens.tolist()):
        token_start = cu_seq[req]
        block_start = cu_blk[req]
        block_end = cu_blk[req + 1]
        for local in range(block_end - block_start):
            if kept_mask_cpu[block_start + local].item() == 0:
                continue
            lo = local * block_size
            hi = min(lo + block_size, seq_len)
            mask[token_start + lo:token_start + hi] = True
    return mask


def python_mask_baseline(tensors_npu, expected_mask_npu):
    hidden, residual, positions, slot_mapping, _scores = tensors_npu
    return hidden[expected_mask_npu], residual[expected_mask_npu], positions[expected_mask_npu], slot_mapping[expected_mask_npu]


def median_us(samples: list[float]) -> float:
    return statistics.median(samples) * 1_000_000.0


def benchmark(args) -> None:
    seq_lens = parse_seq_lens(args.seq_lens)
    plan = make_host_plan(seq_lens, args.block_size, args.attention_sink, args.last_q, args.drop_ratio)
    cpu_inputs = make_inputs(plan, args.hidden_size, args.seed)
    expected = cpu_golden(*cpu_inputs, plan, args.block_size, args.attention_sink, args.last_q)
    expected_token_mask = make_expected_token_mask(plan, expected[-1], args.block_size)
    tensors_npu = tuple(x.npu() for x in cpu_inputs)
    expected_token_mask_npu = expected_token_mask.bool().npu()
    outputs_npu = make_outputs_npu(plan, args.hidden_size)
    meta_npu, kept_block_indices = prepare_variant_state(args, plan)
    sync_npu()
    for _ in range(args.warmup):
        run_variant(args, tensors_npu, meta_npu, outputs_npu, kept_block_indices)
    sync_npu()
    out_samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        run_variant(args, tensors_npu, meta_npu, outputs_npu, kept_block_indices)
        sync_npu()
        out_samples.append(time.perf_counter() - t0)
    for _ in range(args.warmup):
        python_mask_baseline(tensors_npu, expected_token_mask_npu)
    sync_npu()
    mask_samples = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        python_mask_baseline(tensors_npu, expected_token_mask_npu)
        sync_npu()
        mask_samples.append(time.perf_counter() - t0)
    out_us = median_us(out_samples)
    mask_us = median_us(mask_samples)
    print("Benchmark:")
    print(f"  seq_lens: {seq_lens}")
    print(f"  hidden_size: {args.hidden_size}")
    print(f"  total_tokens: {plan.total_tokens}")
    print(f"  total_real_tokens: {plan.total_real_tokens}")
    print(f"  python_mask_baseline_median_us: {mask_us:.3f}")
    print(f"  variant: {args.variant}")
    print(f"  fixed_topk_compact_out_median_us: {out_us:.3f}")
    print(f"  speedup_vs_python_mask: {(mask_us / out_us) if out_us > 0 else float('inf'):.3f}x")


def run_profile(args) -> None:
    from torch.profiler import ProfilerActivity, profile
    seq_lens = parse_seq_lens(args.seq_lens)
    plan = make_host_plan(seq_lens, args.block_size, args.attention_sink, args.last_q, args.drop_ratio)
    tensors_npu = tuple(x.npu() for x in make_inputs(plan, args.hidden_size, args.seed))
    outputs_npu = make_outputs_npu(plan, args.hidden_size)
    meta_npu, kept_block_indices = prepare_variant_state(args, plan)
    sync_npu()
    os.makedirs(args.profile_dir, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU], record_shapes=True) as prof:
        for _ in range(args.iters):
            run_variant(args, tensors_npu, meta_npu, outputs_npu, kept_block_indices)
        sync_npu()
    trace_path = os.path.join(args.profile_dir, "fixed_topk_compact_out_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"Profiler trace written to {trace_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["correctness", "benchmark"], default="correctness")
    parser.add_argument("--seq-lens", default="8192,8192")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--variant", choices=["scalar", "tiled"], default="tiled")
    parser.add_argument("--hidden-tile", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--attention-sink", type=int, default=128)
    parser.add_argument("--last-q", type=int, default=128)
    parser.add_argument("--drop-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="./prof_fixed_topk_compact")
    args = parser.parse_args()
    require_npu()
    load_op()
    if args.mode == "correctness":
        cases = [("short_dense", [256]), ("single_8k", [8192]), ("double_8k", [8192, 8192]), ("varlen_batch", [1024, 4096, 8192]), ("non_multiple", [1000, 8193])]
        results = [correctness_case(args, name, seq) for name, seq in cases]
        passed = sum(1 for x in results if x)
        print(f"Total: {len(results)}, Passed: {passed}, Failed: {len(results) - passed}")
        sys.exit(0 if passed == len(results) else 1)
    benchmark(args)
    if args.profile:
        run_profile(args)


if __name__ == "__main__":
    main()

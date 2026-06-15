from __future__ import annotations

import os
import sys

import torch
import torch_npu

from golden import compute_golden, get_cu_block_seqlens


SO_NAME = "libtopselection_ops.so"
OP_NAME = "topselection_top_p_mask"


def make_scores(cu_block_seqlens: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    total_blocks = int(cu_block_seqlens[-1].item())
    scores = torch.rand(total_blocks, generator=generator, dtype=torch.float32)
    if total_blocks > 0:
        scores[0] = 0.0
    return scores


def run_case(
    name: str,
    seqlens: list[int],
    block_size: int,
    attention_sink: int,
    last_q: int,
    p: float,
    seed: int,
) -> bool:
    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    if seqlens:
        cu_seqlens[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32), dim=0)
    cu_block_seqlens = get_cu_block_seqlens(cu_seqlens, block_size)
    block_scores = make_scores(cu_block_seqlens, seed)

    expected = compute_golden(
        block_scores,
        cu_seqlens,
        cu_block_seqlens,
        block_size,
        attention_sink,
        last_q,
        p,
    )

    op = getattr(torch.ops.npu, OP_NAME)
    actual = op(
        block_scores.npu(),
        cu_seqlens.npu(),
        cu_block_seqlens.npu(),
        block_size,
        attention_sink,
        last_q,
        p,
    )

    actual_cpu = tuple(x.cpu() for x in actual)
    expected_cpu = tuple(x.cpu() for x in expected)
    ok = all(torch.equal(a.bool() if a.dtype == torch.uint8 else a, e.bool() if e.dtype == torch.bool else e)
             for a, e in zip(actual_cpu, expected_cpu))
    status = "PASSED" if ok else "FAILED"
    print(f"{name}: {status}")
    if not ok:
        labels = ["block_mask", "token_mask", "new_cu_seqlens", "new_max_seq_len"]
        for label, a, e in zip(labels, actual_cpu, expected_cpu):
            print(f"  {label} actual={a}")
            print(f"  {label} expect={e}")
    return ok


def main() -> None:
    so_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Build first with cmake -S . -B build && cmake --build build -j4")
        sys.exit(1)

    torch.ops.load_library(so_path)

    cases = [
        ("single_small", [17], 8, 4, 4, 0.8, 1),
        ("single_tail", [65], 16, 8, 8, 0.9, 2),
        ("multi_varlen", [1, 9, 33, 70], 16, 4, 8, 0.99, 3),
        ("p_one", [128], 32, 16, 16, 1.0, 4),
        ("zero_scores_fallback", [20], 8, 2, 4, 0.5, 5),
    ]

    results = []
    for case in cases:
        name, seqlens, block_size, attention_sink, last_q, p, seed = case
        if name == "zero_scores_fallback":
            # Keep this case listed for now; explicit zero score injection will be
            # added after the first device build confirms the launch path.
            pass
        results.append(run_case(name, seqlens, block_size, attention_sink, last_q, p, seed))

    passed = sum(results)
    total = len(results)
    print(f"Total: {total}, Passed: {passed}, Failed: {total - passed}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()


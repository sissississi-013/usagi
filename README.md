# usagi

Block-sparse causal attention kernel for H100, written in Triton.

Built for the [Paradigm Attention Kernel Challenge](https://github.com/paradigmxyz/attention-kernel-challenge).

## Results

0.660ms geometric mean latency on full benchmark (B=2, H=16, T=8192, D=128).
Started at 1.238ms. Leaderboard score: 0.74ms.

All work done in Triton 3.4.0 on CUDA 12.8.

## Approach

FlashAttention-2 style online softmax over CSR block-sparse patterns (128x128 blocks). Single fused kernel handles all three sparsity families (sliding window, global prefix, retrieval).

Key optimizations:
- Split-diagonal inner loop (non-diagonal blocks in fast path, diagonal with causal mask)
- num_warps=8 (2x over the standard num_warps=4 for 128x128 tiles)
- Block pointer K/V loads (tl.make_block_ptr) combined with warp specialization (tl.range(warp_specialize=True))
- Triton stride specialization with precompiled variants
- exp2 softmax, fp16 P@V dot product, fp32 accumulation

See [docs/OPTIMIZATION_JOURNAL.md](docs/OPTIMIZATION_JOURNAL.md) for the full optimization story: what we tried, what worked, what didn't, and why.

## Kernels

| File | Measured | Leaderboard | Description |
|------|----------|-------------|-------------|
| `kernels/v7_blockptr_warpspec.py` | 0.660ms median | - | Best measured. Block pointers + warp specialization. |
| `kernels/v4_leaderboard_0740ms.py` | 0.657ms median | **0.74ms** | Leaderboard submission (4th place). Split diagonal, manual loads. |
| `kernels/v1_baseline.py` | 0.850ms | 0.85ms | First accepted submission. Ctypes launcher. |

Each file is a self-contained submission: drop it into the challenge harness as `submission.py`.

## License

MIT

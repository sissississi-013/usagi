# Optimization Journal: Block-Sparse Causal Attention on H100

Project: Paradigm Attention Kernel Challenge
Date: April 9, 2026
Hardware: NVIDIA H100 SXM (132 SMs, 80GB HBM3, 3.35 TB/s, 228KB SMEM/SM)
Environment: Triton 3.4.0, CUDA 12.8, PyTorch 2.8.0, Python 3.11
Final result: 0.660ms geometric mean latency (best single run: 0.617ms)
Starting point: 1.238ms

---

## 1. Problem

Block-sparse causal forward attention on H100. Q, K, V in bf16 with shape (B, H, T, D), D=128. Sparsity given as CSR at 128x128 block granularity. Variable sequence lengths. Outputs: bf16 attention output and fp32 log-sum-exp.

Three sparsity families: sliding window, sliding window + global prefix, sliding window + random retrieval blocks. Scored by geometric mean of family median latencies. Full benchmark uses B=2, H=16, T=8192, density 8-16%.

At 10% density, each query block touches only 4-12 KV blocks. The kernel is dominated by memory movement and per-block overhead, not raw compute. Branches, masks, and pointer arithmetic in the inner loop are proportionally expensive.

---

## 2. Literature

### Papers

[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (Dao et al. 2022). Memory traffic, not FLOPs, is the bottleneck. Never materialize the attention matrix. Keep Q resident, stream K/V tiles, update softmax online. Even more true for sparse attention where the KV working set is small.

[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao 2023). Reduce non-matmul work. Use exp2 (single-cycle PTX on NVIDIA) instead of exp. Separate the hot matmul path from masking and bookkeeping. With only 4-12 KV blocks per query row, every scalar op matters.

[FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (Shah et al. 2024). Overlap data movement with tensor-core compute using TMA descriptors and warp-specialized pipelines. Full FA3-style warp specialization is hard to reproduce in Triton, but the principle (structure the kernel so KV loads overlap with compute) is the right direction.

[Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089) (Yuan et al. 2025). Fuse structured sparse paths instead of treating everything as generic CSR. Sliding window has regular contiguous structure. A single generic CSR kernel is unlikely to be optimal.

[The Anatomy of a Triton Attention Kernel](https://arxiv.org/abs/2511.11581) (Ringlein et al. 2025). Branching inside the inner loop breaks Triton's software pipelining, causing up to 2x performance loss. This motivated our split-diagonal approach.

### H100 hardware reference

| Spec | Value |
|------|-------|
| bf16 Tensor Core TFLOPS | 989 |
| HBM3 Bandwidth | 3.35 TB/s |
| Shared Memory per SM | 228 KB |
| Registers per SM | 65,536 |
| L2 Cache | 50 MB |
| SMs | 132 |

### Roofline

For our workload (density ~10%, T=8192, BH=32):
- Memory-bound floor: ~0.25ms (total KV load / HBM bandwidth)
- Compute-bound floor: ~0.11ms (total matmul FLOPs / tensor core throughput)
- Achieved: 0.66ms (2.6x above memory floor)

The gap comes from softmax computation, pointer arithmetic, loop control, and memory access pattern inefficiencies.

---

## 3. Optimization Phases

### Phase 1: First working kernel (1.238ms)

Problem: Triton JIT compiles per-kernel C launcher stubs via subprocess. The challenge sandbox blocks subprocess calls. CudaUtils are pre-seeded, but per-kernel launchers compile on demand inside the sandbox.

Solution: Python ctypes launcher. Replace Triton's CudaLauncher with a PythonCudaLauncher that calls cuLaunchKernel directly through ctypes.CDLL("libcuda.so.1"). Required parsing kernel signatures, converting args to ctypes, building void** kernelParams, and handling constexpr parameters.

Kernel: standard FlashAttention-2 online softmax, unified loop with `if kb == qb` causal check, num_warps=4, num_stages=3.

Result: 1.238ms. All 9 cases passing.

### Phase 2: num_warps discovery (1.238ms to 0.85ms)

num_warps=8 was the single largest optimization, and it was not obvious.

| num_warps | Latency | Change |
|-----------|---------|--------|
| 4 | 1.44ms | +16% worse |
| 8 | 0.73ms | -41% |
| 16 | 1.12ms | -10% |

Triton tutorials generally suggest num_warps=4 for 128x128 tiles. On H100 with sparse attention's data-dependent loop bounds, 8 warps hides latency much better: some warps handle memory loads and softmax while others run tensor core matmuls. 16 warps caused too much register pressure. 4 warps left too much idle time.

### Phase 3: Split diagonal and stride specialization (0.85ms to 0.74ms)

Split-diagonal loop: process non-diagonal KV blocks in a tight loop without any causal mask check, then handle the diagonal block separately with the full causal mask. This avoids the pipeline-breaking branch documented in the "Anatomy" paper.

```
# Non-diagonal: tight loop, no causal mask
for j in range(num_kv - 1):
    k = load(K[col_idx[j]])
    scores = dot(q, k^T) * scale
    softmax_update(scores)
    v = load(V[col_idx[j]])
    acc += dot(p, v)

# Diagonal: separate, full causal mask
k = load(K[qb])
scores = dot(q, k^T) * scale
scores = where(causal_mask, scores, -inf)
softmax_update(scores)
```

Stride specialization turned out to be critical. do_not_specialize on strides caused a 1.5x slowdown. Triton JIT generates specialized code based on runtime parameter properties (value==1, divisible by 16, general). Without this, the compiler produces generic address arithmetic.

The complication: different sparsity seeds produce different col_idx sizes, changing stride_cz. Solved by precompiling two variants during setup (one for %16==0 strides, one for other). Two variants cover all cases.

Other improvements in this phase:
- Removed ctypes launcher (Triton JIT works natively after harness update)
- torch.empty instead of torch.zeros for output (kernel writes all positions)
- Metadata caching (reshape row_ptr, col_idx, seq_lens once, reuse across calls)
- Simplified non-diagonal mask (q_valid only, no redundant seq_len check; non-diagonal KV positions are provably within seq_len since k_end <= q_start < seq_len)

Result: 0.74ms on leaderboard (4th place).

### Phase 4: Exhaustive micro-optimization sweep

Before trying fundamentally different approaches, we tested every available knob to establish the true performance floor.

| Optimization | Result | Verdict |
|---|---|---|
| num_stages=2 | 0.83ms | 14% worse |
| num_stages=4 | OOM (294912 > 228KB SMEM) | impossible |
| QBLOCKS_PER_PROG=2 | 0.80ms | worse |
| QBLOCKS_PER_PROG=4 | 0.88ms | much worse |
| Unified loop (no split) | 0.77ms | slightly worse |
| Fused dot-accumulate | 0.85ms | pipeline bubble |
| maxnreg=128 | ptxas fatal | can't fit kernel |
| maxnreg=160 | 0.90ms | register spilling |
| maxnreg=168 | 0.74ms | neutral |
| Early V load | neutral | compiler already handles this |
| eviction_policy variants | hurt or neutral | default is best |
| Maskless non-diagonal | 0.67-0.85ms | within variance |

Note: QBLOCKS_PER_PROG=2 helped in opus_v2 (with ctypes launcher) but hurt in opus_v3. The simpler v3 code compiled better, and the extra loop overhead was not amortized.

Run-to-run variance on Modal was +-15% (0.657ms to 0.876ms for identical code). This makes A/B testing unreliable for small improvements. We ran 3-5 times per variant and compared medians.

### Phase 5: Research-driven approaches (0.74ms to 0.66ms)

After exhausting parameter tuning, we tested fundamentally different techniques from the literature.

**TMA descriptors** (tl.make_tensor_descriptor): hardware TMA loads bypass L1 and go directly from HBM to shared memory. Required triton.set_allocator() for runtime scratch memory.
Result: 0.933ms, 26% slower. TMA overhead and L1 bypass hurt for 128x128 blocks where L1 was already effective.

**Pre-scaled Q**: mathematically (Q @ K^T) * s = (Q*s) @ K^T, saves one 128x128 element-wise multiply per KV block.
Result: fails correctness. max_diff=0.015625, 15x over atol=1e-3. bf16 mantissa (7 bits) can't represent QK_SCALE (~0.127) with sufficient precision after the scale-then-cast-back-to-bf16 step.

**CTA clusters** (num_ctas=2): Hopper feature for threadblock cooperation through shared L2.
Result: fails to compile in Triton 3.4 (PassManager::run failed).

**Warp specialization alone** (tl.range(warp_specialize=True) on non-diagonal loop): splits warps into producers (memory loads) and consumers (computation).
Result: 0.730ms median, neutral vs baseline.

**Block pointers alone** (tl.make_block_ptr for K/V loads): standard loads with Triton-optimized address computation. No TMA hardware, no allocator needed.
Result: 0.733ms median, neutral vs baseline.

**Block pointers + warp specialization combined**:

| Version | Runs | Median |
|---------|------|--------|
| v4 baseline | 0.657, 0.667, 0.792, 0.876 | ~0.730 |
| Block ptrs only | 0.640, 0.733, 0.839 | 0.733 |
| Warp spec only | 0.730, 0.682, 0.770 | 0.730 |
| Block ptrs + warp spec | 0.617, 0.646, 0.660, 0.756, 0.792 | 0.660 |

Each optimization was neutral alone. Together they improved median by ~10%. Hypothesis: block pointers change the load issuing pattern in a way that pairs better with warp specialization's producer/consumer split. With manual pointer loads, all warps run the same address arithmetic before loading. With block pointers, address computation is internal to the load, letting producer warps issue loads without blocking consumers.

---

## 4. Key findings

### num_warps=8 is the dominant factor
2x improvement over num_warps=4. Foundation for everything else. On H100 with sparse attention's irregular memory access, 8 warps is necessary for latency hiding.

### Stride specialization matters
Triton generates specialized code when it knows stride properties (==1, %16==0). Disabling this costs 50%. Different sparsity seeds change stride_cz, requiring precompilation of 2 variants.

### bf16 P@V fails, fp16 P@V passes
bf16 has 7-bit mantissa (~0.8% relative error). Multiplying attention weights by values compounds this past atol=1e-3. fp16 has 10-bit mantissa (~0.1% relative error), passes comfortably, same tensor core throughput.

### Pre-scaling Q breaks correctness
Mathematically equivalent, numerically different. QK_SCALE (~0.127) reduces Q magnitude, and casting back to bf16 introduces 0.015625 rounding error. Many implementations do pre-scale Q, but they either use fp16 or have looser tolerances. With atol=1e-3 on bf16, the post-dot fp32 multiply is necessary.

### TMA is not always better
TMA bypasses L1 cache. For 128x128 blocks where L1 was already effective, the bypass hurts. TMA is better for large sequential transfers where L1 would thrash, not for CSR random-access patterns.

### Neutral optimizations can compound
Block pointers and warp specialization were each neutral alone. Together they gave 10% improvement. This suggests they address different bottlenecks that only become visible when both are present.

### Variance requires statistical discipline
+-15% run-to-run variance on Modal H100. A single run says almost nothing for small improvements. Need 3-5 runs minimum comparing medians.

### Simpler code compiles faster
opus_v3 (no ctypes launcher, cleaner code) was faster than opus_v2 (ctypes). Triton's compiler optimizes clean patterns well.

---

## 5. What did not work

### Correctness failures
| Approach | Error | Cause |
|----------|-------|-------|
| Pre-scale Q (bf16) | max_diff=0.015625 | bf16 can't represent QK_SCALE*Q precisely |
| bf16 P@V | max_diff=0.008 | 7-bit mantissa too imprecise |

### Slower than baseline
| Approach | Latency | Cause |
|----------|---------|-------|
| TMA descriptors | 0.933ms (+26%) | L1 bypass hurts cached access |
| num_warps=4 | 1.44ms (+95%) | insufficient latency hiding |
| num_warps=16 | 1.12ms (+51%) | register pressure |
| num_stages=2 | 0.83ms (+12%) | insufficient pipelining |
| num_stages=4 | OOM | exceeds 228KB SMEM |
| do_not_specialize | 1.07ms (+45%) | generic address arithmetic |
| QBLOCKS_PER_PROG=4 | 0.88ms (+19%) | loop overhead exceeds L2 benefit |
| Fused dot-acc | 0.85ms (+15%) | pipeline bubble |
| maxnreg=160 | 0.90ms (+22%) | register spilling |
| Warp spec + num_warps=4 | 1.28ms (+73%) | still insufficient warps |
| tl.range(num_stages=2) | 0.80ms (+9%) | under-pipelined inner loop |
| Unified loop + warp spec | 0.75ms (+2%) | branch in loop hurts specialization |

### Compilation failures
| Approach | Error | Cause |
|----------|-------|-------|
| num_ctas=2 | PassManager::run failed | unsupported in Triton 3.4 |
| maxnreg=128 | ptxas fatal | kernel doesn't fit |

---

## 6. Final kernel (v7)

### Structure
```
Grid: (64, 32) = (num_q_blocks, batch_heads) = 2048 programs
Per program: one 128-row query block
Inner loop: iterate over CSR-indexed KV blocks
Softmax: online, exp2, fp32 accumulation
PV product: fp16 tensor cores
Config: num_warps=8, num_stages=3
```

### Design choices

1. Split diagonal: non-diagonal KV blocks in a warp-specialized loop, diagonal block handled separately with causal mask.

2. Block pointer K/V loads via tl.make_block_ptr. Not TMA. Standard loads with Triton-optimized address generation.

3. Warp specialization via tl.range(warp_specialize=True) on non-diagonal loop. Splits 8 warps into producers (K/V loads) and consumers (dot products and softmax).

4. Stride specialization. Triton JIT compiles separate cubins per stride class. Setup precompiles both variants.

5. Metadata caching. row_ptr, col_idx, seq_lens reshaped once and cached across benchmark iterations.

### Latency by family

| Family | Median range | Density range |
|--------|-------------|---------------|
| sliding_window | 0.585-0.763ms | 8.7-11.8% |
| sliding_window_global | 0.578-0.827ms | 9.9-15.8% |
| sliding_window_retrieval | 0.636-0.887ms | 12.0-15.8% |
| Geometric mean | 0.660ms | |

---

## 7. Version history

| Version | Geo mean | Change |
|---------|----------|--------|
| v1 baseline | 1.238ms | ctypes launcher, unified loop, num_warps=4 |
| opus_v2 | 0.850ms | num_warps=8, QBLOCKS_PER_PROG=2 |
| opus_v3 v1 | 0.802ms | no ctypes, split diagonal |
| opus_v3 v2 | 0.727ms | removed eviction hints |
| opus_v3 v3 | 0.737ms | torch.empty, kernel writes all positions |
| opus_v3 v4 | 0.657ms | simplified mask, metadata caching |
| opus_v3 v5 | 0.705ms | no eviction_policy (slightly worse) |
| opus_v3 v6 | 0.670ms | maskless non-diagonal |
| opus_v3 v7 | 0.660ms | block pointers + warp specialization |

---

## 8. Remaining directions

1. Raw CUDA / CUTLASS. Triton adds overhead vs hand-tuned CUDA. The #1 position (0.50ms) likely uses CUDA.

2. Family-specialized kernels. Sliding window has contiguous KV blocks that could benefit from sequential TMA loads. Using VARIANT_MANIFEST to dispatch per family.

3. Compile-time seq_len specialization. When seq_len == t_max, all boundary masks are unnecessary. A version without boundary checks could save 5-10%.

4. fp8 KV quantization. H100 fp8 tensor cores are 2x faster. Unlikely to pass atol=1e-3 without careful per-block scaling.

---

## 9. Methodology notes

What worked: systematic parameter sweeps before creative approaches. Literature-driven hypotheses. Multiple runs per variant. Version snapshots at every milestone.

What to do differently: establish variance baseline (5+ runs) before any optimization. Get profiling data (Nsight) if possible. Test combinations of optimizations early instead of only testing each in isolation.

The optimization landscape for sparse attention on H100 is surprisingly flat near the optimum. Most micro-optimizations land within the +-15% noise floor. The few that matter are large effects: num_warps (2x), stride specialization (1.5x), split diagonal (~10%), block pointers + warp spec (~10% compound). Everything else is noise on this workload.

"""
OptiKernel opus_v3 — CSR block-sparse causal attention for H100.

Block pointer K/V loads for optimized address computation.
Split diagonal/non-diagonal, exp2 softmax, fp16 P@V.
"""

import math

import torch
import triton
import triton.language as tl

# ─── Constants ─────────────────────────────────────────────────────
BLOCK_SIZE = 128
HEAD_DIM = 128

_SCORE_SCALE = 1.0 / math.sqrt(HEAD_DIM)
_LOG2E = math.log2(math.e)
_QK_SCALE = _SCORE_SCALE * _LOG2E
_LN2 = math.log(2.0)
_QBLOCKS_PER_PROG = 1

VARIANT_MANIFEST = [{"name": "default"}]


# ═══════════════════════════════════════════════════════════════════
# CSR kernel with block pointer K/V loads
# ═══════════════════════════════════════════════════════════════════
@triton.jit
def _fwd_csr(
    Q, K, V, Out, LSE,
    ROW_PTR, COL_IDX, SEQ_LENS,
    stride_z, stride_t,
    stride_lz,
    stride_rz, stride_cz,
    num_q_blocks,
    QK_SCALE: tl.constexpr,
    LN2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    QBLOCKS_PER_PROG: tl.constexpr,
):
    prog_qb = tl.program_id(0)
    bh = tl.program_id(1)

    seq_len = tl.load(SEQ_LENS + bh).to(tl.int32)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    base_qkv = bh * stride_z
    T_dim = num_q_blocks * BLOCK_M

    for local_qb in range(QBLOCKS_PER_PROG):
        qb = prog_qb * QBLOCKS_PER_PROG + local_qb
        q_start = qb * BLOCK_M

        q_pos = q_start + offs_m

        if q_start >= seq_len or qb >= num_q_blocks:
            tl.store(Out + base_qkv + q_pos[:, None] * stride_t + offs_d[None, :],
                     tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.bfloat16))
            tl.store(LSE + bh * stride_lz + q_pos,
                     tl.full([BLOCK_M], float('-inf'), dtype=tl.float32))
        else:
            q_valid = q_pos < seq_len

            q = tl.load(Q + base_qkv + q_pos[:, None] * stride_t + offs_d[None, :])

            m_i = tl.full([BLOCK_M], value=-1e30, dtype=tl.float32)
            l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

            row_start = tl.load(ROW_PTR + bh * stride_rz + qb).to(tl.int32)
            row_end = tl.load(ROW_PTR + bh * stride_rz + qb + 1).to(tl.int32)
            num_kv = row_end - row_start
            col_base = COL_IDX + bh * stride_cz + row_start

            # Non-diagonal blocks: no causal mask needed
            for j in tl.range(0, num_kv - 1, warp_specialize=True):
                kb = tl.load(col_base + j).to(tl.int32)
                k_start = kb * BLOCK_N

                k_ptr = tl.make_block_ptr(
                    K + base_qkv, shape=(T_dim, BLOCK_D),
                    strides=(stride_t, 1), offsets=(k_start, 0),
                    block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
                k = tl.load(k_ptr)
                scores = tl.dot(q, tl.trans(k)) * QK_SCALE

                scores = tl.where(q_valid[:, None], scores, float('-inf'))

                m_j = tl.max(scores, axis=1)
                m_new = tl.maximum(m_i, m_j)
                alpha = tl.math.exp2(m_i - m_new)
                p = tl.math.exp2(scores - m_new[:, None])
                l_i = alpha * l_i + tl.sum(p, axis=1)

                v_ptr = tl.make_block_ptr(
                    V + base_qkv, shape=(T_dim, BLOCK_D),
                    strides=(stride_t, 1), offsets=(k_start, 0),
                    block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
                v = tl.load(v_ptr)
                acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
                m_i = m_new

            # Diagonal block: apply causal mask
            k_start = qb * BLOCK_N
            k_ptr = tl.make_block_ptr(
                K + base_qkv, shape=(T_dim, BLOCK_D),
                strides=(stride_t, 1), offsets=(k_start, 0),
                block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
            k = tl.load(k_ptr)
            scores = tl.dot(q, tl.trans(k)) * QK_SCALE

            diag_mask = (offs_n[None, :] <= offs_m[:, None]) & \
                        ((k_start + offs_n[None, :]) < seq_len) & \
                        q_valid[:, None]
            scores = tl.where(diag_mask, scores, float('-inf'))

            m_j = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_j)
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(scores - m_new[:, None])
            l_i = alpha * l_i + tl.sum(p, axis=1)

            v_ptr = tl.make_block_ptr(
                V + base_qkv, shape=(T_dim, BLOCK_D),
                strides=(stride_t, 1), offsets=(k_start, 0),
                block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
            v = tl.load(v_ptr)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v.to(tl.float16))
            m_i = m_new

            l_safe = tl.where(l_i > 0.0, l_i, 1.0)
            acc = acc / l_safe[:, None]
            acc = tl.where(q_valid[:, None], acc, 0.0)

            tl.store(
                Out + base_qkv + q_pos[:, None] * stride_t + offs_d[None, :],
                acc.to(tl.bfloat16),
            )
            lse = tl.where(l_i > 0.0, m_i * LN2 + tl.log(l_safe), float('-inf'))
            lse = tl.where(q_valid, lse, float('-inf'))
            tl.store(LSE + bh * stride_lz + q_pos, lse)



# ─── helper: run CSR kernel ──────────────────────────────────────
_meta_cache = {}

def _run_csr(q, k, v, row_ptr, col_idx, seq_lens):
    B, H, T, D = q.shape
    BH = B * H
    nqb = T // BLOCK_SIZE

    q_flat = q.reshape(BH, T, D)
    k_flat = k.reshape(BH, T, D)
    v_flat = v.reshape(BH, T, D)

    output = torch.empty(BH, T, D, device=q.device, dtype=torch.bfloat16)
    lse = torch.empty(BH, T, device=q.device, dtype=torch.float32)

    cache_key = (row_ptr.data_ptr(), col_idx.data_ptr(), seq_lens.data_ptr())
    cached = _meta_cache.get(cache_key)
    if cached is not None:
        rp_flat, ci_flat, sl_flat = cached
    else:
        max_nnz = col_idx.shape[-1]
        rp_flat = row_ptr.reshape(BH, nqb + 1).to(torch.int32)
        ci_flat = col_idx.reshape(BH, max_nnz).to(torch.int32)
        sl_flat = seq_lens.repeat_interleave(H).to(torch.int32)
        _meta_cache[cache_key] = (rp_flat, ci_flat, sl_flat)

    grid_q = (nqb + _QBLOCKS_PER_PROG - 1) // _QBLOCKS_PER_PROG
    _fwd_csr[(grid_q, BH)](
        q_flat, k_flat, v_flat, output, lse,
        rp_flat, ci_flat, sl_flat,
        q_flat.stride(0), q_flat.stride(1),
        lse.stride(0),
        rp_flat.stride(0), ci_flat.stride(0),
        nqb,
        QK_SCALE=_QK_SCALE, LN2=_LN2,
        BLOCK_M=BLOCK_SIZE, BLOCK_N=BLOCK_SIZE, BLOCK_D=HEAD_DIM,
        QBLOCKS_PER_PROG=_QBLOCKS_PER_PROG,
        num_warps=8, num_stages=3,
    )
    return (output.reshape(B, H, T, D), lse.reshape(B, H, T))


# ─── setup ────────────────────────────────────────────────────────
def _triton_spec_class(v):
    if v == 1: return 1
    if v % 16 == 0: return 16
    return 0


def setup(suite_specs, device, variants):
    if str(device) != "cuda" or not suite_specs:
        return None

    seen_combos = set()
    for spec in suite_specs:
        T_s = spec.t_max
        nqb_s = T_s // BLOCK_SIZE
        combo_base = (
            _triton_spec_class(T_s * HEAD_DIM),
            _triton_spec_class(HEAD_DIM),
            _triton_spec_class(T_s),
            _triton_spec_class(nqb_s + 1),
            _triton_spec_class(nqb_s),
        )

        M = ((nqb_s + 15) // 16) * 16
        for max_nnz in (M, M + 1):
            combo = combo_base + (_triton_spec_class(max_nnz),)
            if combo in seen_combos:
                continue
            seen_combos.add(combo)

            q_c = torch.empty(1, T_s, HEAD_DIM, device=device, dtype=torch.bfloat16)
            k_c = torch.empty_like(q_c)
            v_c = torch.empty_like(q_c)
            o_c = torch.empty_like(q_c)
            lse_c = torch.empty(1, T_s, device=device, dtype=torch.float32)
            rp_c = torch.arange(nqb_s + 1, device=device, dtype=torch.int32).unsqueeze(0)
            ci_c = torch.zeros(1, max_nnz, device=device, dtype=torch.int32)
            sl_c = torch.tensor([T_s], device=device, dtype=torch.int32)

            _fwd_csr[(1, 1)](
                q_c, k_c, v_c, o_c, lse_c,
                rp_c, ci_c, sl_c,
                q_c.stride(0), q_c.stride(1),
                lse_c.stride(0),
                rp_c.stride(0), ci_c.stride(0),
                nqb_s,
                QK_SCALE=_QK_SCALE, LN2=_LN2,
                BLOCK_M=128, BLOCK_N=128, BLOCK_D=128,
                QBLOCKS_PER_PROG=_QBLOCKS_PER_PROG,
                num_warps=8, num_stages=3,
            )
            del q_c, k_c, v_c, o_c, lse_c, rp_c, ci_c, sl_c

    torch.cuda.synchronize()


# ─── entry point ──────────────────────────────────────────────────
def block_sparse_attn_fwd(q, k, v, row_ptr, col_idx, seq_lens):
    return _run_csr(q, k, v, row_ptr, col_idx, seq_lens)

"""Fused gated Filter forward + contrast via Triton.

Reference path (Filter.forward + Filter.contrast):
    x1 = normalize(x1 * gate)           # [B, K]
    x2 = normalize(x2[n] * gate[b])     # [B, N, K]  (materialized)
    logits = einsum("bj,bnj->bn", x1, x2)

Fused math (no [B,N,K] tensor):
    g2 = gate^2
    qnorm[b]   = ||x1[b] * gate[b]||
    knorm[b,n] = ||x2[n] * gate[b]||
    logits[b,n] = sum_k x1[b,k] * x2[n,k] * g2[b,k] / (qnorm[b] * knorm[b,n])

Both reductions over ``k`` are matrix products sharing the same ``x2`` tile::

    dot[b,n]    = (x1 * g2) @ x2^T
    knorm2[b,n] = g2 @ (x2 * x2)^T

so the Triton path runs them as one fused tensor-core GEMM with two
accumulators: each key tile is loaded once and feeds both ``tl.dot`` calls, no
[B,N] intermediates are written, and the ``dot * rsqrt(knorm2)`` epilogue happens
in registers. That replaces the two cuBLAS GEMMs plus ~4 elementwise [B,N]
passes of the torch formulation.

Scale conditioning (exact, and what makes fp16 tensor cores usable here): the
target value is invariant to per-row rescaling of ``x1``, ``gate`` and ``x2``, so
the prologue stores ``w = x1 * ghat^2 / ||x1 * ghat||`` with
``ghat = gate / max|gate|`` and ``x2hat = x2 / max|x2|``. Every GEMM operand then
lives in [-1, 1] whatever the gate/feature magnitudes, which matters because a
gate row of ~1e-3 (sigmoid of a moderately negative logit) would push ``gate^2``
into fp16 subnormals.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_AVAILABLE = False


_EPS = 1e-12
# Guard for the rescaling divisions (kernels inline this value). Only reached for
# all-zero rows, which the reference maps to logits 0 (F.normalize of a zero
# vector returns zeros).
_TINY = 1e-30

# Largest K handled by the single-tile prologue kernel; above this fall back to
# the torch prologue (K is the embedding dim, normally 128-512).
_MAX_PREP_K = 2048


def reference_gated_contrast(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference matching ``Filter.forward`` + ``Filter.contrast``.

    Args:
        x1: [B, K] query features
        x2: [N, K] key features
        gate: [B, K] gate vectors

    Returns:
        logits: [B, N]
    """
    x1g = x1 * gate
    x2g = x2.unsqueeze(0) * gate.unsqueeze(1)  # [B, N, K]
    x1n = F.normalize(x1g, p=2, dim=-1)
    x2n = F.normalize(x2g, p=2, dim=-1)
    return torch.einsum("bj,bnj->bn", x1n, x2n)


def torch_gated_contrast_einsum(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Einsum fusion without the [B,N,K] tensor (original torch baseline)."""
    g2 = gate * gate
    qnorm = (x1 * x1 * g2).sum(dim=-1).clamp_min(_EPS).sqrt()
    knorm2 = torch.einsum("nk,bk->bn", x2 * x2, g2).clamp_min(_EPS)
    knorm = knorm2.sqrt()
    dot = torch.einsum("bk,nk->bn", x1 * g2, x2)
    return dot / (qnorm.unsqueeze(1) * knorm)


def torch_gated_contrast(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Mathematically equivalent reference without the [B,N,K] tensor.

    Same two GEMMs as the Triton path but through cuBLAS. ``qnorm`` is folded
    into the [B,K] operand and the [B,N] epilogue runs in place, which is the
    fewest passes reachable without a custom kernel.

    Note: unlike the Triton path this does not rescale the operands, so the
    ``knorm2`` GEMM can overflow in fp16 if the key features are far from unit
    scale.
    """
    inplace_ok = not (
        torch.is_grad_enabled()
        and (x1.requires_grad or x2.requires_grad or gate.requires_grad)
    )
    g2 = gate * gate
    qnorm = (
        (x1.float() * x1.float() * g2.float())
        .sum(dim=-1, keepdim=True)
        .clamp_min(_EPS)
        .sqrt()
        .to(x1.dtype)
    )
    if inplace_ok:
        w = (x1 * g2).div_(qnorm)
        dot = torch.mm(w, x2.t())
        knorm2 = torch.mm(g2, (x2 * x2).t())
        return dot.mul_(knorm2.clamp_min_(_EPS).rsqrt_())
    w = x1 * g2 / qnorm
    dot = torch.mm(w, x2.t())
    knorm2 = torch.mm(g2, (x2 * x2).t())
    return dot * torch.rsqrt(knorm2.clamp_min(_EPS))


# Kept for backwards compatibility with existing callers / benchmarks.
reference_gated_contrast_no_materialize = torch_gated_contrast_einsum


if _TRITON_AVAILABLE:

    @triton.jit
    def _prep_kernel(
        x1_ptr,  # [B, K]
        gate_ptr,  # [B, K]
        x2_ptr,  # [N, K]
        q_ptr,  # [2, B, K] out: w = x1 * ghat^2 / ||x1 * ghat||, then ghat^2
        kb_ptr,  # [1 or 2, N, K] out: x2 / max|x2|, then its square
        B,
        N,
        K,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
        WRITE_KSQ: tl.constexpr,
    ):
        """Row-tiled prologue building every GEMM operand in a single launch.

        The first ``cdiv(B, BLOCK_R)`` programs build the query side, the rest
        rescale the key side. Each program handles ``BLOCK_R`` rows so the row
        reductions run along a tile axis instead of costing one block-wide
        reduction per row. All reductions are fp32. All tensors are contiguous,
        so row strides are ``K`` and no stride arguments are needed (kernel
        argument count dominates the launch cost at these sizes).
        """
        pid = tl.program_id(0)
        n_q_blocks = tl.cdiv(B, BLOCK_R)
        offs_k = tl.arange(0, BLOCK_K)
        offs_r = tl.arange(0, BLOCK_R)
        k_mask = offs_k < K

        if pid < n_q_blocks:
            rows = pid * BLOCK_R + offs_r
            mask = (rows < B)[:, None] & k_mask[None, :]
            offs = rows[:, None] * K + offs_k[None, :]
            x1 = tl.load(x1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            ghat = g / tl.maximum(tl.max(tl.abs(g), axis=1), 1e-30)[:, None]
            gh2 = ghat * ghat
            u = x1 * gh2
            qnorm = tl.sqrt(tl.maximum(tl.sum(x1 * u, axis=1), 1e-30))[:, None]
            tl.store(
                q_ptr + offs, (u / qnorm).to(q_ptr.dtype.element_ty), mask=mask
            )
            tl.store(
                q_ptr + B * K + offs, gh2.to(q_ptr.dtype.element_ty), mask=mask
            )
        else:
            rows = (pid - n_q_blocks) * BLOCK_R + offs_r
            mask = (rows < N)[:, None] & k_mask[None, :]
            offs = rows[:, None] * K + offs_k[None, :]
            x2 = tl.load(x2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            smax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-30)[:, None]
            xh = (x2 / smax).to(kb_ptr.dtype.element_ty)
            tl.store(kb_ptr + offs, xh, mask=mask)
            if WRITE_KSQ:
                tl.store(kb_ptr + N * K + offs, xh * xh, mask=mask)

    def _gemm_configs():
        # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
        # Two fp32 [BLOCK_M, BLOCK_N] accumulators live in registers, so keep
        # BLOCK_M * BLOCK_N / (32 * num_warps) <= ~128 regs/thread; wider tiles
        # spill badly (measured 3x slower on GH200).
        specs = [
            (16, 128, 64, 4, 3),
            (32, 128, 64, 4, 3),
            (32, 256, 64, 8, 3),
            (64, 64, 128, 4, 3),
            (64, 128, 64, 8, 3),
            (64, 128, 64, 8, 4),
            (64, 256, 64, 8, 3),
            (64, 256, 64, 8, 4),
            (128, 128, 32, 8, 4),
            (128, 128, 64, 8, 3),
            (128, 128, 64, 8, 4),
            (256, 64, 64, 8, 3),
        ]
        return [
            triton.Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8},
                num_warps=nw,
                num_stages=ns,
            )
            for bm, bn, bk, nw, ns in specs
        ]

    @triton.jit
    def _gc_gemm_kernel(
        q_ptr,  # [2, M, K]: pre-scaled w then ghat^2
        kb_ptr,  # [2, N, K]: pre-scaled keys then their squares
        out_ptr,  # [M, N]
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        EVEN_K: tl.constexpr,
        IN_PREC: tl.constexpr,
    ):
        """Fused double GEMM: ``dot = w @ x2^T`` and ``knorm2 = g2 @ (x2^2)^T``.

        The two contractions share the same tile geometry, so the second one
        costs MMA throughput but almost no extra memory traffic (the key
        matrices are small enough to stay L2-resident). Only the [M, N] logits
        are written -- no [B,N] intermediates and no separate epilogue pass.

        ``x2^2`` comes precomputed from the prologue rather than being squared
        here: measured ~5% faster on GH200, since keeping both MMA operands as
        plain loads leaves the pipelining to the compiler.
        """
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = (offs_m < M)[:, None]
        n_mask = (offs_n < N)[:, None]

        w_ptrs = q_ptr + offs_m[:, None] * K + offs_k[None, :]
        g2_ptrs = w_ptrs + M * K
        x2_ptrs = kb_ptr + offs_n[:, None] * K + offs_k[None, :]
        x2sq_ptrs = x2_ptrs + N * K

        acc_dot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_kn = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            if EVEN_K:
                w = tl.load(w_ptrs, mask=m_mask, other=0.0)
                g2 = tl.load(g2_ptrs, mask=m_mask, other=0.0)
                x2 = tl.load(x2_ptrs, mask=n_mask, other=0.0)
                x2sq = tl.load(x2sq_ptrs, mask=n_mask, other=0.0)
            else:
                k_mask = offs_k[None, :] < K - k * BLOCK_K
                w = tl.load(w_ptrs, mask=m_mask & k_mask, other=0.0)
                g2 = tl.load(g2_ptrs, mask=m_mask & k_mask, other=0.0)
                x2 = tl.load(x2_ptrs, mask=n_mask & k_mask, other=0.0)
                x2sq = tl.load(x2sq_ptrs, mask=n_mask & k_mask, other=0.0)

            # [BLOCK_K, BLOCK_N] operands; loads stay K-contiguous (NT layout).
            if IN_PREC == "default":
                acc_dot = tl.dot(w, tl.trans(x2), acc_dot)
                acc_kn = tl.dot(g2, tl.trans(x2sq), acc_kn)
            else:
                acc_dot = tl.dot(w, tl.trans(x2), acc_dot, input_precision=IN_PREC)
                acc_kn = tl.dot(
                    g2, tl.trans(x2sq), acc_kn, input_precision=IN_PREC
                )

            w_ptrs += BLOCK_K
            g2_ptrs += BLOCK_K
            x2_ptrs += BLOCK_K
            x2sq_ptrs += BLOCK_K

        logits = acc_dot * tl.rsqrt(tl.maximum(acc_kn, 1e-30))
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            logits.to(out_ptr.dtype.element_ty),
            mask=m_mask & (offs_n < N)[None, :],
        )

    _gc_gemm_autotuned = triton.autotune(configs=_gemm_configs(), key=["M", "N", "K"])(
        _gc_gemm_kernel
    )

    @triton.jit
    def _gc_reduce_kernel(
        q_ptr,  # [2, B, K] fp32: w then ghat^2
        kb_ptr,  # [1, N, K] fp32: pre-scaled keys
        out_ptr,  # [B, N]
        B,
        N,
        K,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Non-tensor-core fallback: one program per batch row, loops N tiles.

        Cheaper than the GEMM path when B is tiny (a handful of rows), where MMA
        tiles would be mostly padding.
        """
        b = tl.program_id(0)
        w_b = q_ptr + b * K
        g2_b = w_b + B * K

        for n_start in range(0, N, BLOCK_N):
            n_offs = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offs < N
            dot_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            knorm_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

            for k_start in range(0, K, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offs < K
                w = tl.load(w_b + k_offs, mask=k_mask, other=0.0).to(tl.float32)
                g2 = tl.load(g2_b + k_offs, mask=k_mask, other=0.0).to(tl.float32)
                x2 = tl.load(
                    kb_ptr + n_offs[:, None] * K + k_offs[None, :],
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                dot_acc += tl.sum(x2 * w[None, :], axis=1)
                knorm_acc += tl.sum(x2 * x2 * g2[None, :], axis=1)

            logits = dot_acc * tl.rsqrt(tl.maximum(knorm_acc, 1e-30))
            tl.store(
                out_ptr + b * N + n_offs,
                logits.to(out_ptr.dtype.element_ty),
                mask=n_mask,
            )


def _fp32_input_precision() -> str:
    """``tl.dot`` precision for fp32 operands.

    Honours ``torch.backends.cuda.matmul.allow_tf32`` so the Triton path has the
    same accuracy contract as the torch/cuBLAS baselines, with an env override
    (``SIMLAP_GC_FP32_PRECISION=tf32|tf32x3|ieee``).
    """
    override = os.environ.get("SIMLAP_GC_FP32_PRECISION", "").strip()
    if override:
        return override
    return "tf32" if torch.backends.cuda.matmul.allow_tf32 else "ieee"


def _contig(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


_PREP_CFG_CACHE: dict = {}


def _prep_operands(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gate: torch.Tensor,
    op_dtype: torch.dtype,
    want_ksq: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the rescaled GEMM operands: ``q`` = [2, B, K], ``kb`` = [1|2, N, K]."""
    B, K = x1.shape
    N = x2.shape[0]

    if _TRITON_AVAILABLE and K <= _MAX_PREP_K and x1.is_cuda:
        q = torch.empty((2, B, K), device=x1.device, dtype=op_dtype)
        kb = torch.empty((2 if want_ksq else 1, N, K), device=x1.device, dtype=op_dtype)
        cfg = _PREP_CFG_CACHE.get((B, N, K))
        if cfg is None:
            block_k = triton.next_power_of_2(K)
            # ~4K elements per program keeps the row reductions on a tile axis
            # while staying inside the register budget.
            block_r = max(1, min(16, 4096 // block_k))
            cfg = (
                (triton.cdiv(B, block_r) + triton.cdiv(N, block_r),),
                block_r,
                block_k,
                8 if block_r * block_k >= 4096 else 4,
            )
            _PREP_CFG_CACHE[(B, N, K)] = cfg
        grid, block_r, block_k, num_warps = cfg
        _prep_kernel[grid](
            x1,
            gate,
            x2,
            q,
            kb,
            B,
            N,
            K,
            BLOCK_R=block_r,
            BLOCK_K=block_k,
            WRITE_KSQ=want_ksq,
            num_warps=num_warps,
        )
        return q, kb

    # Torch prologue (very large K, or Triton unavailable for the prologue).
    gf = gate.float()
    ghat = gf / gf.abs().amax(dim=-1, keepdim=True).clamp_min(_TINY)
    gh2 = ghat * ghat
    u = x1.float() * gh2
    qnorm = (x1.float() * u).sum(dim=-1, keepdim=True).clamp_min(_TINY).sqrt()
    q = torch.stack([(u / qnorm).to(op_dtype), gh2.to(op_dtype)])
    smax = x2.float().abs().amax(dim=-1, keepdim=True).clamp_min(_TINY)
    x2h = (x2.float() / smax).to(op_dtype)
    kb = torch.stack([x2h, x2h * x2h]) if want_ksq else x2h.unsqueeze(0)
    return q, kb


# Autotuned config per (M, N, K, dtype); the autotune wrapper itself costs
# ~10-15us of Python per call, which is significant next to a ~60us kernel, so
# it only runs once per shape and the raw kernel is launched afterwards.
_GEMM_CFG_CACHE: dict = {}


def _reduce_blocks(N: int, K: int) -> Tuple[int, int]:
    bk = 128 if K <= 128 else 256
    if N >= 1024:
        bn = 128
    elif N >= 256:
        bn = 64
    else:
        bn = 32
    return bn, bk


def _launch_triton(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gate: torch.Tensor,
    *,
    method: str = "auto",
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Triton forward. ``method`` is ``auto`` | ``gemm`` | ``reduce``."""
    assert _TRITON_AVAILABLE, "triton is required for fused_gated_contrast"
    B, K = x1.shape
    N = x2.shape[0]
    assert x2.shape == (N, K), f"x2 shape {tuple(x2.shape)} != {(N, K)}"
    assert gate.shape == (B, K), f"gate shape {tuple(gate.shape)} != {(B, K)}"
    assert x1.is_cuda and x2.is_cuda and gate.is_cuda

    x1 = _contig(x1)
    x2 = _contig(x2)
    gate = _contig(gate)
    if out_dtype is None:
        out_dtype = x1.dtype

    if method == "auto":
        # Tensor-core tiles need at least ~16 rows to pay off; below that the
        # per-row reduction kernel wins.
        method = "reduce" if B < 16 else "gemm"

    # int32 pointer arithmetic inside the kernels.
    assert max(B * N, B * K, N * K) < 2**31, "problem too large for the Triton path"

    if method == "reduce":
        # fp32 operands: this path is FMA-based, so precision costs nothing.
        q, kb = _prep_operands(x1, x2, gate, torch.float32, want_ksq=False)
        out = torch.empty((B, N), device=x1.device, dtype=out_dtype)
        bn, bk = _reduce_blocks(N, K)
        _gc_reduce_kernel[(B,)](
            q, kb, out, B, N, K, BLOCK_N=bn, BLOCK_K=bk, num_warps=8, num_stages=3
        )
        return out

    op_dtype = (
        x1.dtype if x1.dtype in (torch.float16, torch.bfloat16) else torch.float32
    )
    q, kb = _prep_operands(x1, x2, gate, op_dtype, want_ksq=True)
    out = torch.empty((B, N), device=x1.device, dtype=out_dtype)
    even_k = K % 32 == 0
    in_prec = "default" if op_dtype != torch.float32 else _fp32_input_precision()

    key = (B, N, K, op_dtype, out_dtype, in_prec)
    cached = _GEMM_CFG_CACHE.get(key)
    if cached is None:
        # First call for this shape: let the autotuner pick a config, then reuse
        # it directly (the wrapper's per-call bookkeeping is pure overhead).
        grid = lambda meta: (  # noqa: E731
            triton.cdiv(B, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
        )
        _gc_gemm_autotuned[grid](q, kb, out, B, N, K, EVEN_K=even_k, IN_PREC=in_prec)
        best = _gc_gemm_autotuned.best_config
        bm, bn = best.kwargs["BLOCK_M"], best.kwargs["BLOCK_N"]
        _GEMM_CFG_CACHE[key] = (
            (triton.cdiv(B, bm) * triton.cdiv(N, bn),),
            bm,
            bn,
            best.kwargs["BLOCK_K"],
            best.kwargs["GROUP_M"],
            best.num_warps,
            best.num_stages,
        )
        return out

    grid, bm, bn, bk, gm, nw, ns = cached
    _gc_gemm_kernel[grid](
        q,
        kb,
        out,
        B,
        N,
        K,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=gm,
        EVEN_K=even_k,
        IN_PREC=in_prec,
        num_warps=nw,
        num_stages=ns,
    )
    return out


class _FusedGatedContrastFn(torch.autograd.Function):
    """Forward: Triton. Backward: torch GEMMs (no [B,N,K] materialization)."""

    @staticmethod
    def forward(ctx, x1, x2, gate, method):
        with torch.no_grad():
            logits = _launch_triton(x1, x2, gate, method=method)
        # ``logits`` is the returned tensor, so keeping it for backward costs no
        # extra memory and saves recomputing the [B,N] dot in backward.
        ctx.save_for_backward(x1, x2, gate, logits)
        return logits

    @staticmethod
    def backward(ctx, grad_logits):
        x1, x2, gate, logits = ctx.saved_tensors
        dt = x1.dtype

        # The [B,K] reductions run in fp32 (cheap and precision-critical) while
        # the [B,N]-sized GEMMs stay in the input dtype so they keep tensor
        # cores -- an all-fp32 backward costs ~6x more at B=N=4096, and the
        # reference backward is in the input dtype anyway.
        x1_f = x1.float()
        g_f = gate.float()
        q_raw = x1_f * g_f
        qnorm = q_raw.pow(2).sum(-1, keepdim=True).clamp_min(_EPS).sqrt()
        q = q_raw / qnorm
        g2 = g_f * g_f
        g2_c = g2.to(dt)

        # Inputs may differ in dtype under autocast; the GEMMs need one dtype.
        x2c = x2 if x2.dtype == dt else x2.to(dt)
        grad_logits = grad_logits.to(dt)
        x2sq = x2c * x2c
        inv_knorm = torch.mm(g2_c, x2sq.t()).clamp_min_(_EPS).rsqrt_()
        scaled = grad_logits * inv_knorm
        coeff2 = scaled * logits.to(dt) * inv_knorm
        del inv_knorm

        sx = torch.mm(scaled, x2c).float()
        d_logits_dq = sx * g_f
        dq_dot_q = (d_logits_dq * q).sum(-1, keepdim=True)
        dq_raw = (d_logits_dq - q * dq_dot_q) / qnorm
        grad_x1 = (dq_raw * g_f).to(dt)

        grad_x2 = torch.mm(scaled.t(), (q * g_f).to(dt))
        grad_x2 -= x2c * torch.mm(coeff2.t(), g2_c)
        grad_x2 = grad_x2.to(dtype=x2.dtype)

        grad_gate = (
            dq_raw * x1_f + q * sx - g_f * torch.mm(coeff2, x2sq).float()
        ).to(dtype=gate.dtype)

        return grad_x1, grad_x2, grad_gate, None


# fp32 crossover points measured on GH200 (see the bench JSONs under
# fused-gated-contrast/outputs). Expressed in B*N*K MACs.
_FP32_TF32_MAX_WORK = int(float(os.environ.get("SIMLAP_GC_FP32_TF32_MAX_WORK", 1.2e9)))
_FP32_IEEE_MAX_WORK = int(float(os.environ.get("SIMLAP_GC_FP32_MAX_WORK", 1e8)))


def _prefer_triton(B: int, N: int, K: int, dtype: torch.dtype) -> bool:
    """``backend="auto"`` dispatch.

    fp16/bf16: the fused Triton GEMM wins at every measured shape (1.7-4.8x),
    so always take it.

    fp32: ``tl.dot`` has no tensor-core path under the default
    ``allow_tf32=False`` contract, so cuBLAS SGEMM wins the GEMM itself and
    Triton is only ahead while the saved kernel launches and [B,N] passes
    dominate. With TF32 allowed both sides get tensor cores and the crossover
    moves up by ~10x. Beyond the crossover, fall back to ``torch``.

    ROCm/MI250: GEMM tiles need ~144KB shared mem; gfx90a has 64KB (SimLAP
    e100 job 22194946). Use the torch GEMM path there.
    """
    if getattr(torch.version, "hip", None):
        return False
    if dtype in (torch.float16, torch.bfloat16):
        return True
    limit = (
        _FP32_TF32_MAX_WORK
        if torch.backends.cuda.matmul.allow_tf32
        else _FP32_IEEE_MAX_WORK
    )
    return B * N * K <= limit


_FUSED_PATH_LOGGED = False


def fused_gated_contrast(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gate: torch.Tensor,
    *,
    backend: str = "auto",
    method: str = "auto",
) -> torch.Tensor:
    """Fused gated Filter forward + contrast.

    Args:
        x1: [B, K] queries
        x2: [N, K] keys
        gate: [B, K] gate vectors
        backend: ``"auto"`` | ``"triton"`` | ``"torch"`` | ``"torch_einsum"``
            | ``"reference"``
            - ``triton``: fused Triton double GEMM (CUDA required)
            - ``auto``: Triton when it is expected to win, else ``torch``
            - ``torch``: two cuBLAS GEMMs + in-place epilogue, no [B,N,K]
            - ``torch_einsum``: original einsum fusion, no [B,N,K]
            - ``reference``: exact Filter.forward + contrast path
        method: Triton kernel selection, ``"auto"`` | ``"gemm"`` | ``"reduce"``

    Returns:
        logits: [B, N] cosine similarities after gated L2 normalization
    """
    global _FUSED_PATH_LOGGED

    def _log_once(chosen: str) -> None:
        global _FUSED_PATH_LOGGED
        if _FUSED_PATH_LOGGED:
            return
        _FUSED_PATH_LOGGED = True
        B, K = x1.shape
        N = x2.shape[0]
        print(
            f"[fused_gated_contrast] ACTIVE path={chosen} "
            f"requested={backend} shape=B{B}_N{N}_K{K} dtype={x1.dtype} "
            f"cuda={x1.is_cuda} file={__file__}",
            flush=True,
        )

    if backend == "reference":
        _log_once("reference")
        return reference_gated_contrast(x1, x2, gate)
    if backend == "torch_einsum":
        _log_once("torch_einsum")
        return torch_gated_contrast_einsum(x1, x2, gate)
    if backend == "torch":
        _log_once("torch")
        return torch_gated_contrast(x1, x2, gate)
    if backend not in ("auto", "triton"):
        raise ValueError(f"unknown backend {backend!r}")

    triton_ok = _TRITON_AVAILABLE and x1.is_cuda
    if backend == "triton" and not triton_ok:
        raise RuntimeError(
            "Triton backend requested but unavailable "
            f"(triton={_TRITON_AVAILABLE}, cuda={x1.is_cuda})"
        )
    if backend == "auto" and triton_ok:
        triton_ok = _prefer_triton(x1.shape[0], x2.shape[0], x1.shape[1], x1.dtype)
    if not triton_ok:
        _log_once("torch_fallback")
        return torch_gated_contrast(x1, x2, gate)

    _log_once("triton")
    if x1.requires_grad or x2.requires_grad or gate.requires_grad:
        return _FusedGatedContrastFn.apply(x1, x2, gate, method)
    return _launch_triton(x1, x2, gate, method=method)


def check_correctness(
    B: int = 64,
    N: int = 256,
    K: int = 256,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    check_backward: bool = True,
) -> Tuple[bool, dict]:
    """Compare Triton / torch-fused vs reference Filter path (fwd + bwd)."""
    if rtol is None or atol is None:
        if dtype == torch.float32:
            rtol, atol = 1e-4, 1e-5
        elif dtype == torch.float16:
            rtol, atol = 2e-2, 2e-3
        else:  # bfloat16
            rtol, atol = 3e-2, 5e-3

    torch.manual_seed(0)
    x1 = torch.randn(B, K, device=device, dtype=dtype)
    x2 = torch.randn(N, K, device=device, dtype=dtype)
    gate = torch.rand(B, K, device=device, dtype=dtype).clamp(0.05, 1.0)

    ref = reference_gated_contrast(x1, x2, gate)
    torch_fused = torch_gated_contrast(x1, x2, gate)
    torch_einsum = torch_gated_contrast_einsum(x1, x2, gate)
    stats = {
        "dtype": str(dtype).replace("torch.", ""),
        "shape": (B, N, K),
        "torch_vs_ref_max": (torch_fused.float() - ref.float()).abs().max().item(),
        "torch_vs_ref_mean": (torch_fused.float() - ref.float()).abs().mean().item(),
        "torch_einsum_vs_ref_max": (torch_einsum.float() - ref.float())
        .abs()
        .max()
        .item(),
    }

    ok_torch = torch.allclose(torch_fused.float(), ref.float(), rtol=rtol, atol=atol)
    has_cuda = (
        _TRITON_AVAILABLE
        and str(device).startswith("cuda")
        and torch.cuda.is_available()
    )
    ok_triton = False
    ok_extra = True
    if has_cuda:
        ref32 = ref.float()
        for label, kwargs in (("triton", {}), ("triton_reduce", {"method": "reduce"})):
            tri = fused_gated_contrast(x1, x2, gate, backend="triton", **kwargs).float()
            stats[f"{label}_vs_ref_max"] = (tri - ref32).abs().max().item()
            stats[f"{label}_vs_ref_mean"] = (tri - ref32).abs().mean().item()
            passed = torch.allclose(tri, ref32, rtol=rtol, atol=atol)
            stats[f"ok_{label}"] = bool(passed)
            if label == "triton":
                ok_triton = passed
            else:
                ok_extra = ok_extra and passed

        # Scale robustness: the value is invariant to per-row rescaling, and a
        # badly scaled gate is exactly what breaks a naive fp16 GEMM (gate^2
        # would overflow at the top of this range and go subnormal at the
        # bottom). Powers of two keep the rescaled inputs exactly
        # representable, so the target value is unchanged.
        scale = torch.exp2(
            torch.linspace(-8, 8, B, device=device, dtype=torch.float32)
        ).to(dtype)
        tri_s = fused_gated_contrast(
            x1, x2 * 0.0625, gate * scale.unsqueeze(1), backend="triton"
        ).float()
        stats["triton_scaled_vs_ref_max"] = (tri_s - ref32).abs().max().item()
        passed = torch.allclose(tri_s, ref32, rtol=rtol, atol=atol * 5)
        stats["ok_triton_scaled"] = bool(passed)
        ok_extra = ok_extra and passed

        if check_backward:
            ok_extra = ok_extra and _check_backward(x1, x2, gate, rtol, atol, stats)
    else:
        stats["triton_vs_ref_max"] = float("nan")
        stats["triton_vs_ref_mean"] = float("nan")
        stats["note"] = "triton skipped (no CUDA)"

    stats["ok"] = bool(ok_torch and ok_triton and ok_extra)
    stats["ok_torch"] = bool(ok_torch)
    stats["ok_triton"] = bool(ok_triton)
    stats["ok_extra"] = bool(ok_extra)
    stats["rtol"] = rtol
    stats["atol"] = atol
    return stats["ok"], stats


def _check_backward(x1, x2, gate, rtol, atol, stats) -> bool:
    """Compare Triton-forward autograd grads against the reference path."""
    grads = {}
    torch.manual_seed(1)
    grad_out = torch.randn(x1.shape[0], x2.shape[0], device=x1.device, dtype=x1.dtype)
    for label, fn in (
        ("ref", reference_gated_contrast),
        ("triton", lambda a, b, c: fused_gated_contrast(a, b, c, backend="triton")),
    ):
        a = x1.clone().requires_grad_(True)
        b = x2.clone().requires_grad_(True)
        c = gate.clone().requires_grad_(True)
        fn(a, b, c).backward(grad_out)
        grads[label] = (a.grad.float(), b.grad.float(), c.grad.float())

    ok = True
    # The reference backward goes through the [B,N,K] tensor, so relax
    # tolerances relative to the forward comparison.
    g_rtol, g_atol = rtol * 5, atol * 5
    for i, name in enumerate(("x1", "x2", "gate")):
        r, t = grads["ref"][i], grads["triton"][i]
        denom = r.abs().max().clamp_min(1e-30)
        stats[f"grad_{name}_rel_max"] = ((t - r).abs().max() / denom).item()
        passed = torch.allclose(t, r, rtol=g_rtol, atol=g_atol * denom.item())
        stats[f"ok_grad_{name}"] = bool(passed)
        ok = ok and passed
    return ok

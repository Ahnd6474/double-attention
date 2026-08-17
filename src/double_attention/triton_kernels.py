from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except (ImportError, OSError):  # OSError also covers unsupported binary builds.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


MAX_PRECOMPUTED_DP_BYTES = 16 * 1024 * 1024


if TRITON_AVAILABLE:

    @triton.jit
    def _row_l2_normalize_kernel(
        input_ptr,
        output_ptr,
        input_row_stride,
        output_row_stride,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_cols
        values = tl.load(input_ptr + row * input_row_stride + columns, mask=mask, other=0.0).to(tl.float32)
        inverse_norm = tl.rsqrt(tl.sum(values * values, axis=0) + eps)
        tl.store(output_ptr + row * output_row_stride + columns, values * inverse_norm, mask=mask)


    @triton.jit
    def _row_scaled_softmax_kernel(
        input_ptr,
        beta_ptr,
        output_ptr,
        input_row_stride,
        output_row_stride,
        n_cols: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_cols
        beta = tl.load(beta_ptr).to(tl.float32)
        values = tl.load(
            input_ptr + row * input_row_stride + columns,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        values = values * beta
        values = values - tl.max(values, axis=0)
        numerator = tl.exp(values)
        denominator = tl.sum(numerator, axis=0)
        tl.store(output_ptr + row * output_row_stride + columns, numerator / denominator, mask=mask)


    @triton.jit
    def _row_l2_normalize_backward_kernel(
        input_ptr,
        grad_output_ptr,
        grad_input_ptr,
        input_row_stride,
        grad_output_row_stride,
        grad_input_row_stride,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_cols
        values = tl.load(
            input_ptr + row * input_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr + row * grad_output_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        inverse_norm = tl.rsqrt(tl.sum(values * values, axis=0) + eps)
        projection = tl.sum(grad_output * values, axis=0)
        inverse_norm_cubed = inverse_norm * inverse_norm * inverse_norm
        grad_input = grad_output * inverse_norm - values * projection * inverse_norm_cubed
        tl.store(
            grad_input_ptr + row * grad_input_row_stride + columns,
            grad_input,
            mask=mask,
        )


    @triton.jit
    def _row_scaled_softmax_backward_kernel(
        logits_ptr,
        weights_ptr,
        grad_weights_ptr,
        beta_ptr,
        grad_logits_ptr,
        grad_beta_ptr,
        logits_row_stride,
        weights_row_stride,
        grad_weights_row_stride,
        grad_logits_row_stride,
        n_cols: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        mask = columns < n_cols
        logits = tl.load(
            logits_ptr + row * logits_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weights = tl.load(
            weights_ptr + row * weights_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        grad_weights = tl.load(
            grad_weights_ptr + row * grad_weights_row_stride + columns,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        beta = tl.load(beta_ptr).to(tl.float32)
        centered = grad_weights - tl.sum(grad_weights * weights, axis=0)
        grad_scaled_logits = weights * centered
        tl.store(
            grad_logits_ptr + row * grad_logits_row_stride + columns,
            beta * grad_scaled_logits,
            mask=mask,
        )
        tl.store(
            grad_beta_ptr + row,
            tl.sum(grad_scaled_logits * logits, axis=0),
        )


    @triton.jit
    def _routed_attention_forward_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        scale_ptr,
        output_ptr,
        lse_ptr,
        stride_qb,
        stride_qs,
        stride_qt,
        stride_qr,
        stride_kb,
        stride_ks,
        stride_kt,
        stride_kr,
        stride_vb,
        stride_vt,
        stride_vd,
        stride_ob,
        stride_os,
        stride_ot,
        stride_od,
        stride_lb,
        stride_ls,
        stride_lt,
        N_MAPS: tl.constexpr,
        T: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        CAUSAL: tl.constexpr,
        IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_map = tl.program_id(1)
        value_block = tl.program_id(2)
        batch = batch_map // N_MAPS
        score_map = batch_map - batch * N_MAPS

        offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tl.arange(0, BLOCK_N)
        offsets_r = tl.arange(0, BLOCK_R)
        offsets_d = value_block * BLOCK_D + tl.arange(0, BLOCK_D)

        query_pointers = (
            query_ptr
            + batch * stride_qb
            + score_map * stride_qs
            + offsets_m[:, None] * stride_qt
            + offsets_r[None, :] * stride_qr
        )
        query = tl.load(
            query_pointers,
            mask=(offsets_m[:, None] < T) & (offsets_r[None, :] < R),
            other=0.0,
        )
        scale = tl.load(scale_ptr + score_map).to(tl.float32)

        running_max = tl.full([BLOCK_M], -float("inf"), tl.float32)
        running_sum = tl.zeros([BLOCK_M], tl.float32)
        accumulator = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

        key_end = T
        if CAUSAL:
            key_end = (query_block + 1) * BLOCK_M
        for key_start in tl.range(0, key_end, BLOCK_N):
            current_n = key_start + offsets_n
            key_pointers = (
                key_ptr
                + batch * stride_kb
                + score_map * stride_ks
                + current_n[:, None] * stride_kt
                + offsets_r[None, :] * stride_kr
            )
            key = tl.load(
                key_pointers,
                mask=(current_n[:, None] < T) & (offsets_r[None, :] < R),
                other=0.0,
            )
            scores = tl.dot(query, key.T) * scale
            valid = (offsets_m[:, None] < T) & (current_n[None, :] < T)
            if CAUSAL:
                valid = valid & (offsets_m[:, None] >= current_n[None, :])
            scores = tl.where(valid, scores, -1.0e6)

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            correction = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            block_sum = tl.sum(probabilities, axis=1)

            value_pointers = (
                value_ptr
                + batch * stride_vb
                + current_n[:, None] * stride_vt
                + offsets_d[None, :] * stride_vd
            )
            value = tl.load(
                value_pointers,
                mask=(current_n[:, None] < T) & (offsets_d[None, :] < D),
                other=0.0,
            )
            accumulator = accumulator * correction[:, None]
            if IS_BF16:
                accumulator += tl.dot(probabilities.to(tl.bfloat16), value)
            else:
                accumulator += tl.dot(probabilities.to(tl.float16), value)
            running_sum = running_sum * correction + block_sum
            running_max = new_max

        result = accumulator / running_sum[:, None]
        output_pointers = (
            output_ptr
            + batch * stride_ob
            + score_map * stride_os
            + offsets_m[:, None] * stride_ot
            + offsets_d[None, :] * stride_od
        )
        tl.store(
            output_pointers,
            result,
            mask=(offsets_m[:, None] < T) & (offsets_d[None, :] < D),
        )
        lse_pointers = (
            lse_ptr
            + batch * stride_lb
            + score_map * stride_ls
            + offsets_m * stride_lt
        )
        tl.store(
            lse_pointers,
            running_max + tl.log(running_sum),
            mask=(offsets_m < T) & (value_block == 0),
        )


    @triton.jit
    def _attention_backward_preprocess_kernel(
        output_ptr,
        grad_output_ptr,
        delta_ptr,
        stride_ob,
        stride_os,
        stride_ot,
        stride_od,
        stride_gob,
        stride_gos,
        stride_got,
        stride_god,
        stride_db,
        stride_ds,
        stride_dt,
        N_MAPS: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        batch_map = tl.program_id(1)
        batch = batch_map // N_MAPS
        score_map = batch_map - batch * N_MAPS
        offsets_d = tl.arange(0, BLOCK_D)
        output = tl.load(
            output_ptr
            + batch * stride_ob
            + score_map * stride_os
            + token * stride_ot
            + offsets_d * stride_od,
            mask=offsets_d < D,
            other=0.0,
        ).to(tl.float32)
        grad_output = tl.load(
            grad_output_ptr
            + batch * stride_gob
            + score_map * stride_gos
            + token * stride_got
            + offsets_d * stride_god,
            mask=offsets_d < D,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            delta_ptr + batch * stride_db + score_map * stride_ds + token * stride_dt,
            tl.sum(output * grad_output),
        )


    @triton.jit
    def _routed_attention_dq_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        grad_output_ptr,
        grad_probabilities_ptr,
        lse_ptr,
        delta_ptr,
        scale_ptr,
        grad_query_ptr,
        grad_scale_ptr,
        stride_qb,
        stride_qs,
        stride_qt,
        stride_qr,
        stride_kb,
        stride_ks,
        stride_kt,
        stride_kr,
        stride_vb,
        stride_vt,
        stride_vd,
        stride_gob,
        stride_gos,
        stride_got,
        stride_god,
        stride_dpb,
        stride_dps,
        stride_dpt,
        stride_dpn,
        stride_lb,
        stride_ls,
        stride_lt,
        stride_db,
        stride_ds,
        stride_dt,
        stride_dqb,
        stride_dqs,
        stride_dqt,
        stride_dqr,
        N_MAPS: tl.constexpr,
        T: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        CAUSAL: tl.constexpr,
        USE_PRECOMPUTED_DP: tl.constexpr,
        IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_map = tl.program_id(1)
        batch = batch_map // N_MAPS
        score_map = batch_map - batch * N_MAPS
        offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = tl.arange(0, BLOCK_N)
        offsets_r = tl.arange(0, BLOCK_R)

        query = tl.load(
            query_ptr
            + batch * stride_qb
            + score_map * stride_qs
            + offsets_m[:, None] * stride_qt
            + offsets_r[None, :] * stride_qr,
            mask=(offsets_m[:, None] < T) & (offsets_r[None, :] < R),
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + batch * stride_lb + score_map * stride_ls + offsets_m * stride_lt,
            mask=offsets_m < T,
            other=0.0,
        ).to(tl.float32)
        delta = tl.load(
            delta_ptr + batch * stride_db + score_map * stride_ds + offsets_m * stride_dt,
            mask=offsets_m < T,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + score_map).to(tl.float32)
        grad_query = tl.zeros([BLOCK_M, BLOCK_R], tl.float32)
        grad_scale = 0.0

        key_end = T
        if CAUSAL:
            key_end = (query_block + 1) * BLOCK_M
        for key_start in tl.range(0, key_end, BLOCK_N):
            current_n = key_start + offsets_n
            key = tl.load(
                key_ptr
                + batch * stride_kb
                + score_map * stride_ks
                + current_n[:, None] * stride_kt
                + offsets_r[None, :] * stride_kr,
                mask=(current_n[:, None] < T) & (offsets_r[None, :] < R),
                other=0.0,
            )
            unscaled_scores = tl.dot(query, key.T)
            valid = (offsets_m[:, None] < T) & (current_n[None, :] < T)
            if CAUSAL:
                valid = valid & (offsets_m[:, None] >= current_n[None, :])
            probabilities = tl.where(
                valid,
                tl.exp(unscaled_scores * scale - lse[:, None]),
                0.0,
            )
            if USE_PRECOMPUTED_DP:
                grad_probabilities = tl.load(
                    grad_probabilities_ptr
                    + batch * stride_dpb
                    + score_map * stride_dps
                    + offsets_m[:, None] * stride_dpt
                    + current_n[None, :] * stride_dpn,
                    mask=(offsets_m[:, None] < T) & (current_n[None, :] < T),
                    other=0.0,
                ).to(tl.float32)
            else:
                grad_probabilities = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                for value_start in tl.range(0, D, BLOCK_D):
                    offsets_d = value_start + tl.arange(0, BLOCK_D)
                    grad_output = tl.load(
                        grad_output_ptr
                        + batch * stride_gob
                        + score_map * stride_gos
                        + offsets_m[:, None] * stride_got
                        + offsets_d[None, :] * stride_god,
                        mask=(offsets_m[:, None] < T) & (offsets_d[None, :] < D),
                        other=0.0,
                    )
                    value = tl.load(
                        value_ptr
                        + batch * stride_vb
                        + current_n[:, None] * stride_vt
                        + offsets_d[None, :] * stride_vd,
                        mask=(current_n[:, None] < T) & (offsets_d[None, :] < D),
                        other=0.0,
                    )
                    grad_probabilities += tl.dot(grad_output, value.T)
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            if IS_BF16:
                grad_query += tl.dot(grad_scores.to(tl.bfloat16), key) * scale
            else:
                grad_query += tl.dot(grad_scores.to(tl.float16), key) * scale
            grad_scale += tl.sum(grad_scores * unscaled_scores)

        tl.atomic_add(grad_scale_ptr + score_map, grad_scale)

        tl.store(
            grad_query_ptr
            + batch * stride_dqb
            + score_map * stride_dqs
            + offsets_m[:, None] * stride_dqt
            + offsets_r[None, :] * stride_dqr,
            grad_query,
            mask=(offsets_m[:, None] < T) & (offsets_r[None, :] < R),
        )


    @triton.jit
    def _routed_attention_dk_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        grad_output_ptr,
        grad_probabilities_ptr,
        lse_ptr,
        delta_ptr,
        scale_ptr,
        grad_key_ptr,
        stride_qb,
        stride_qs,
        stride_qt,
        stride_qr,
        stride_kb,
        stride_ks,
        stride_kt,
        stride_kr,
        stride_vb,
        stride_vt,
        stride_vd,
        stride_gob,
        stride_gos,
        stride_got,
        stride_god,
        stride_dpb,
        stride_dps,
        stride_dpt,
        stride_dpn,
        stride_lb,
        stride_ls,
        stride_lt,
        stride_db,
        stride_ds,
        stride_dt,
        stride_dkb,
        stride_dks,
        stride_dkt,
        stride_dkr,
        N_MAPS: tl.constexpr,
        T: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        CAUSAL: tl.constexpr,
        USE_PRECOMPUTED_DP: tl.constexpr,
        IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        batch_map = tl.program_id(1)
        batch = batch_map // N_MAPS
        score_map = batch_map - batch * N_MAPS
        offsets_m = tl.arange(0, BLOCK_M)
        offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_r = tl.arange(0, BLOCK_R)
        key = tl.load(
            key_ptr
            + batch * stride_kb
            + score_map * stride_ks
            + offsets_n[:, None] * stride_kt
            + offsets_r[None, :] * stride_kr,
            mask=(offsets_n[:, None] < T) & (offsets_r[None, :] < R),
            other=0.0,
        )
        scale = tl.load(scale_ptr + score_map).to(tl.float32)
        grad_key = tl.zeros([BLOCK_N, BLOCK_R], tl.float32)

        query_begin = 0
        if CAUSAL:
            query_begin = key_block * BLOCK_N
        for query_start in tl.range(query_begin, T, BLOCK_M):
            current_m = query_start + offsets_m
            query = tl.load(
                query_ptr
                + batch * stride_qb
                + score_map * stride_qs
                + current_m[:, None] * stride_qt
                + offsets_r[None, :] * stride_qr,
                mask=(current_m[:, None] < T) & (offsets_r[None, :] < R),
                other=0.0,
            )
            unscaled_scores = tl.dot(query, key.T)
            valid = (current_m[:, None] < T) & (offsets_n[None, :] < T)
            if CAUSAL:
                valid = valid & (current_m[:, None] >= offsets_n[None, :])
            lse = tl.load(
                lse_ptr + batch * stride_lb + score_map * stride_ls + current_m * stride_lt,
                mask=current_m < T,
                other=0.0,
            ).to(tl.float32)
            delta = tl.load(
                delta_ptr + batch * stride_db + score_map * stride_ds + current_m * stride_dt,
                mask=current_m < T,
                other=0.0,
            ).to(tl.float32)
            probabilities = tl.where(
                valid,
                tl.exp(unscaled_scores * scale - lse[:, None]),
                0.0,
            )
            if USE_PRECOMPUTED_DP:
                grad_probabilities = tl.load(
                    grad_probabilities_ptr
                    + batch * stride_dpb
                    + score_map * stride_dps
                    + current_m[:, None] * stride_dpt
                    + offsets_n[None, :] * stride_dpn,
                    mask=(current_m[:, None] < T) & (offsets_n[None, :] < T),
                    other=0.0,
                ).to(tl.float32)
            else:
                grad_probabilities = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                for value_start in tl.range(0, D, BLOCK_D):
                    offsets_d = value_start + tl.arange(0, BLOCK_D)
                    grad_output = tl.load(
                        grad_output_ptr
                        + batch * stride_gob
                        + score_map * stride_gos
                        + current_m[:, None] * stride_got
                        + offsets_d[None, :] * stride_god,
                        mask=(current_m[:, None] < T) & (offsets_d[None, :] < D),
                        other=0.0,
                    )
                    value = tl.load(
                        value_ptr
                        + batch * stride_vb
                        + offsets_n[:, None] * stride_vt
                        + offsets_d[None, :] * stride_vd,
                        mask=(offsets_n[:, None] < T) & (offsets_d[None, :] < D),
                        other=0.0,
                    )
                    grad_probabilities += tl.dot(grad_output, value.T)
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            if IS_BF16:
                grad_key += tl.dot(grad_scores.T.to(tl.bfloat16), query) * scale
            else:
                grad_key += tl.dot(grad_scores.T.to(tl.float16), query) * scale

        tl.store(
            grad_key_ptr
            + batch * stride_dkb
            + score_map * stride_dks
            + offsets_n[:, None] * stride_dkt
            + offsets_r[None, :] * stride_dkr,
            grad_key,
            mask=(offsets_n[:, None] < T) & (offsets_r[None, :] < R),
        )


    @triton.jit
    def _routed_attention_dv_kernel(
        query_ptr,
        key_ptr,
        grad_output_ptr,
        lse_ptr,
        scale_ptr,
        grad_value_ptr,
        stride_qb,
        stride_qs,
        stride_qt,
        stride_qr,
        stride_kb,
        stride_ks,
        stride_kt,
        stride_kr,
        stride_gob,
        stride_gos,
        stride_got,
        stride_god,
        stride_lb,
        stride_ls,
        stride_lt,
        stride_dvb,
        stride_dvt,
        stride_dvd,
        N_MAPS: tl.constexpr,
        T: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        CAUSAL: tl.constexpr,
        IS_BF16: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        batch = tl.program_id(1)
        value_block = tl.program_id(2)
        offsets_m = tl.arange(0, BLOCK_M)
        offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_r = tl.arange(0, BLOCK_R)
        offsets_d = value_block * BLOCK_D + tl.arange(0, BLOCK_D)
        grad_value = tl.zeros([BLOCK_N, BLOCK_D], tl.float32)

        for score_map in tl.static_range(0, N_MAPS):
            key = tl.load(
                key_ptr
                + batch * stride_kb
                + score_map * stride_ks
                + offsets_n[:, None] * stride_kt
                + offsets_r[None, :] * stride_kr,
                mask=(offsets_n[:, None] < T) & (offsets_r[None, :] < R),
                other=0.0,
            )
            scale = tl.load(scale_ptr + score_map).to(tl.float32)
            query_begin = 0
            if CAUSAL:
                query_begin = key_block * BLOCK_N
            for query_start in tl.range(query_begin, T, BLOCK_M):
                current_m = query_start + offsets_m
                query = tl.load(
                    query_ptr
                    + batch * stride_qb
                    + score_map * stride_qs
                    + current_m[:, None] * stride_qt
                    + offsets_r[None, :] * stride_qr,
                    mask=(current_m[:, None] < T) & (offsets_r[None, :] < R),
                    other=0.0,
                )
                scores = tl.dot(query, key.T) * scale
                valid = (current_m[:, None] < T) & (offsets_n[None, :] < T)
                if CAUSAL:
                    valid = valid & (current_m[:, None] >= offsets_n[None, :])
                lse = tl.load(
                    lse_ptr + batch * stride_lb + score_map * stride_ls + current_m * stride_lt,
                    mask=current_m < T,
                    other=0.0,
                ).to(tl.float32)
                probabilities = tl.where(valid, tl.exp(scores - lse[:, None]), 0.0)
                grad_output = tl.load(
                    grad_output_ptr
                    + batch * stride_gob
                    + score_map * stride_gos
                    + current_m[:, None] * stride_got
                    + offsets_d[None, :] * stride_god,
                    mask=(current_m[:, None] < T) & (offsets_d[None, :] < D),
                    other=0.0,
                )
                if IS_BF16:
                    grad_value += tl.dot(probabilities.T.to(tl.bfloat16), grad_output)
                else:
                    grad_value += tl.dot(probabilities.T.to(tl.float16), grad_output)

        tl.store(
            grad_value_ptr
            + batch * stride_dvb
            + offsets_n[:, None] * stride_dvt
            + offsets_d[None, :] * stride_dvd,
            grad_value,
            mask=(offsets_n[:, None] < T) & (offsets_d[None, :] < D),
        )


def _require_triton() -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed; use backend='torch' or backend='auto'")


def _row_l2_normalize(input: Tensor, eps: float) -> Tensor:
    _require_triton()
    if input.ndim != 2 or not input.is_cuda:
        raise ValueError("Triton row normalization expects a CUDA matrix")
    rows, columns = input.shape
    if columns > 4096:
        raise ValueError("Triton row normalization currently supports at most 4096 columns")
    output = torch.empty_like(input)
    block = triton.next_power_of_2(columns)
    warps = 8 if block >= 2048 else 4
    _row_l2_normalize_kernel[(rows,)](
        input,
        output,
        input.stride(0),
        output.stride(0),
        n_cols=columns,
        eps=eps,
        BLOCK=block,
        num_warps=warps,
    )
    return output


def _row_scaled_softmax(input: Tensor, beta: Tensor) -> Tensor:
    _require_triton()
    if input.ndim != 2 or not input.is_cuda:
        raise ValueError("Triton softmax expects a CUDA matrix")
    rows, columns = input.shape
    if columns > 4096:
        raise ValueError("Triton softmax currently supports at most 4096 columns")
    output = torch.empty_like(input)
    block = triton.next_power_of_2(columns)
    warps = 8 if block >= 2048 else 4
    _row_scaled_softmax_kernel[(rows,)](
        input,
        beta,
        output,
        input.stride(0),
        output.stride(0),
        n_cols=columns,
        BLOCK=block,
        num_warps=warps,
    )
    return output


def _row_l2_normalize_backward(input: Tensor, grad_output: Tensor, eps: float) -> Tensor:
    _require_triton()
    input = input.contiguous()
    grad_output = grad_output.contiguous()
    if input.ndim != 2 or input.shape != grad_output.shape or not input.is_cuda:
        raise ValueError("Triton row normalization backward expects matching CUDA matrices")
    rows, columns = input.shape
    if columns > 4096:
        raise ValueError("Triton row normalization backward supports at most 4096 columns")
    grad_input = torch.empty_like(input)
    block = triton.next_power_of_2(columns)
    warps = 8 if block >= 2048 else 4
    _row_l2_normalize_backward_kernel[(rows,)](
        input,
        grad_output,
        grad_input,
        input.stride(0),
        grad_output.stride(0),
        grad_input.stride(0),
        n_cols=columns,
        eps=eps,
        BLOCK=block,
        num_warps=warps,
    )
    return grad_input


def _row_scaled_softmax_backward(
    logits: Tensor,
    weights: Tensor,
    grad_weights: Tensor,
    beta: Tensor,
) -> tuple[Tensor, Tensor]:
    _require_triton()
    logits = logits.contiguous()
    weights = weights.contiguous()
    grad_weights = grad_weights.contiguous()
    if (
        logits.ndim != 2
        or logits.shape != weights.shape
        or logits.shape != grad_weights.shape
        or not logits.is_cuda
    ):
        raise ValueError("Triton softmax backward expects matching CUDA matrices")
    rows, columns = logits.shape
    if columns > 4096:
        raise ValueError("Triton softmax backward supports at most 4096 columns")
    grad_logits = torch.empty_like(logits)
    grad_beta_rows = torch.empty(rows, dtype=torch.float32, device=logits.device)
    block = triton.next_power_of_2(columns)
    warps = 8 if block >= 2048 else 4
    _row_scaled_softmax_backward_kernel[(rows,)](
        logits,
        weights,
        grad_weights,
        beta,
        grad_logits,
        grad_beta_rows,
        logits.stride(0),
        weights.stride(0),
        grad_weights.stride(0),
        grad_logits.stride(0),
        n_cols=columns,
        BLOCK=block,
        num_warps=warps,
    )
    return grad_logits, grad_beta_rows.sum().to(beta.dtype)


def _dictionary_route_triton_forward(
    input: Tensor,
    dictionary_key: Tensor,
    dictionary_value: Tensor,
    beta: Tensor,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    original_shape = input.shape
    flat = input.contiguous().view(-1, original_shape[-1])
    normalized = _row_l2_normalize(flat, eps)
    logits = normalized @ dictionary_key.contiguous()
    weights = _row_scaled_softmax(logits, beta)
    reconstructed = weights @ dictionary_value.transpose(0, 1).contiguous()
    output = _row_l2_normalize(reconstructed, eps)
    return output.view(original_shape), normalized, logits, weights, reconstructed


class _DictionaryRouteFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: Tensor,
        dictionary_key: Tensor,
        dictionary_value: Tensor,
        beta: Tensor,
        eps: float,
    ) -> Tensor:
        output, normalized, logits, weights, reconstructed = _dictionary_route_triton_forward(
            input, dictionary_key, dictionary_value, beta, eps
        )
        ctx.save_for_backward(
            input,
            dictionary_key,
            dictionary_value,
            beta,
            normalized,
            logits,
            weights,
            reconstructed,
        )
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor):
        (
            input,
            dictionary_key,
            dictionary_value,
            beta,
            normalized,
            logits,
            weights,
            reconstructed,
        ) = ctx.saved_tensors
        flat_input = input.contiguous().view(-1, input.shape[-1])
        flat_grad_output = grad_output.contiguous().view(-1, grad_output.shape[-1])
        grad_reconstructed = _row_l2_normalize_backward(
            reconstructed, flat_grad_output, ctx.eps
        )
        grad_weights = grad_reconstructed @ dictionary_value.contiguous()
        grad_dictionary_value = grad_reconstructed.transpose(0, 1) @ weights
        grad_logits, grad_beta = _row_scaled_softmax_backward(
            logits, weights, grad_weights, beta
        )
        grad_normalized = grad_logits @ dictionary_key.transpose(0, 1).contiguous()
        grad_dictionary_key = normalized.transpose(0, 1) @ grad_logits
        grad_input = _row_l2_normalize_backward(
            flat_input, grad_normalized, ctx.eps
        ).view_as(input)
        return grad_input, grad_dictionary_key, grad_dictionary_value, grad_beta, None


def dictionary_route_triton(
    input: Tensor,
    dictionary_key: Tensor,
    dictionary_value: Tensor,
    beta: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Triton forward and backward for the dictionary feature map.

    GEMMs remain with PyTorch/cuBLAS; Triton implements row normalization and
    temperature-scaled softmax in both directions without autograd graph
    recomputation.
    """

    return _DictionaryRouteFunction.apply(input, dictionary_key, dictionary_value, beta, eps)


def _routed_attention_triton_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scales: Tensor,
    causal: bool,
) -> tuple[Tensor, Tensor]:
    _require_triton()
    if query.ndim != 4 or key.shape != query.shape:
        raise ValueError("query and key must have matching [B, S, T, R] shapes")
    if value.ndim != 3 or value.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError("value must have shape [B, T, D] matching query")
    if scales.shape != (query.shape[1],):
        raise ValueError(f"scales must have shape [{query.shape[1]}]")
    tensors = (query, key, value, scales)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("Triton routed attention requires CUDA tensors")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have one matching dtype")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("Triton routed attention requires fp16 or bf16 inputs")
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    scales = scales.contiguous()
    batch, maps, length, routing_dim = query.shape
    model_dim = value.shape[-1]
    if routing_dim > 512:
        raise ValueError("Triton routed attention currently supports routing widths up to 512")
    block_r = max(16, triton.next_power_of_2(routing_dim))
    block_m, block_n = 16, 32
    block_d = 32 if length <= 128 else 64
    output = torch.empty(
        (batch, maps, length, model_dim),
        dtype=value.dtype,
        device=value.device,
    )
    lse = torch.empty(
        (batch, maps, length),
        dtype=torch.float32,
        device=value.device,
    )
    grid = (triton.cdiv(length, block_m), batch * maps, triton.cdiv(model_dim, block_d))
    _routed_attention_forward_kernel[grid](
        query,
        key,
        value,
        scales,
        output,
        lse,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *lse.stride(),
        N_MAPS=maps,
        T=length,
        R=routing_dim,
        D=model_dim,
        CAUSAL=causal,
        IS_BF16=value.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return output, lse


def _routed_attention_triton_backward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scales: Tensor,
    output: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    causal: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _require_triton()
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    scales = scales.contiguous()
    output = output.contiguous()
    lse = lse.contiguous()
    grad_output = grad_output.contiguous()
    batch, maps, length, routing_dim = query.shape
    model_dim = value.shape[-1]
    grad_probabilities_bytes = batch * maps * length * length * 4
    expanded_value_bytes = (
        batch * maps * length * model_dim * value.element_size()
    )
    use_precomputed_grad_probabilities = (
        grad_probabilities_bytes + expanded_value_bytes <= MAX_PRECOMPUTED_DP_BYTES
    )
    if use_precomputed_grad_probabilities:
        expanded_value = value[:, None].expand(-1, maps, -1, -1)
        try:
            grad_probabilities = torch.bmm(
                grad_output.flatten(0, 1),
                expanded_value.flatten(0, 1).transpose(1, 2),
                out_dtype=torch.float32,
            ).view(batch, maps, length, length)
        except TypeError:
            # Older supported PyTorch releases do not expose CUDA bmm's FP32
            # output. Preserve gradient fidelity by using the recompute path.
            use_precomputed_grad_probabilities = False
            grad_probabilities = grad_output
    else:
        # The pointer is never read when the constexpr branch is disabled.
        grad_probabilities = grad_output
    block_m = 16
    if length <= 128:
        block_n, block_d = 16, 32
    else:
        block_n, block_d = 32, 64
    block_r = max(16, triton.next_power_of_2(routing_dim))
    qk_warps = 8 if routing_dim >= 256 else 4
    delta_block_d = triton.next_power_of_2(model_dim)
    if delta_block_d > 4096:
        raise ValueError("Triton routed attention backward supports model widths up to 4096")

    delta = torch.empty((batch, maps, length), dtype=torch.float32, device=query.device)
    _attention_backward_preprocess_kernel[(length, batch * maps)](
        output,
        grad_output,
        delta,
        *output.stride(),
        *grad_output.stride(),
        *delta.stride(),
        N_MAPS=maps,
        T=length,
        D=model_dim,
        BLOCK_D=delta_block_d,
        num_warps=8 if delta_block_d >= 2048 else 4,
    )

    grad_query = torch.empty_like(query)
    grad_key = torch.empty_like(key)
    grad_value = torch.empty_like(value)
    grad_scales = torch.zeros_like(scales, dtype=torch.float32)
    grid_q = (triton.cdiv(length, block_m), batch * maps)
    _routed_attention_dq_kernel[grid_q](
        query,
        key,
        value,
        grad_output,
        grad_probabilities,
        lse,
        delta,
        scales,
        grad_query,
        grad_scales,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *grad_output.stride(),
        *grad_probabilities.stride(),
        *lse.stride(),
        *delta.stride(),
        *grad_query.stride(),
        N_MAPS=maps,
        T=length,
        R=routing_dim,
        D=model_dim,
        CAUSAL=causal,
        USE_PRECOMPUTED_DP=use_precomputed_grad_probabilities,
        IS_BF16=query.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=qk_warps,
        num_stages=2,
    )
    grid_k = (triton.cdiv(length, block_n), batch * maps)
    _routed_attention_dk_kernel[grid_k](
        query,
        key,
        value,
        grad_output,
        grad_probabilities,
        lse,
        delta,
        scales,
        grad_key,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *grad_output.stride(),
        *grad_probabilities.stride(),
        *lse.stride(),
        *delta.stride(),
        *grad_key.stride(),
        N_MAPS=maps,
        T=length,
        R=routing_dim,
        D=model_dim,
        CAUSAL=causal,
        USE_PRECOMPUTED_DP=use_precomputed_grad_probabilities,
        IS_BF16=query.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=qk_warps,
        num_stages=2,
    )
    grid_v = (
        triton.cdiv(length, block_n),
        batch,
        triton.cdiv(model_dim, block_d),
    )
    _routed_attention_dv_kernel[grid_v](
        query,
        key,
        grad_output,
        lse,
        scales,
        grad_value,
        *query.stride(),
        *key.stride(),
        *grad_output.stride(),
        *lse.stride(),
        *grad_value.stride(),
        N_MAPS=maps,
        T=length,
        R=routing_dim,
        D=model_dim,
        CAUSAL=causal,
        IS_BF16=query.dtype == torch.bfloat16,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return grad_query, grad_key, grad_value, grad_scales.to(scales.dtype)


class _RoutedAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scales: Tensor,
        causal: bool,
    ) -> Tensor:
        output, lse = _routed_attention_triton_forward(query, key, value, scales, causal)
        ctx.save_for_backward(query, key, value, scales, output, lse)
        ctx.causal = causal
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor):
        query, key, value, scales, output, lse = ctx.saved_tensors
        gradients = _routed_attention_triton_backward(
            query, key, value, scales, output, lse, grad_output, ctx.causal
        )
        return (*gradients, None)


def routed_attention_triton(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scales: Tensor,
    causal: bool = True,
) -> Tensor:
    """Flash-style forward and backward for routed Q/K and a shared dense value.

    Neither direction writes scores or probabilities to global memory.
    Backward reuses forward log-sum-exp statistics and, within a bounded
    workspace budget, shares one FP32 ``dO @ V.T`` buffer between dQ and dK.
    Larger cases recompute those tiles directly in Triton.
    """

    return _RoutedAttentionFunction.apply(query, key, value, scales, causal)

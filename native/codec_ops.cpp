// Derived from ggml.c at 57ea0bc119d722d74594196cc5b494a34dd87be4
// (conv_1d lines 4483-4500, conv_1d_dw lines 4515-4531).
// Keep the im2col payload in F32.
// GGML source copyright/license remains in upstream/s2.cpp/ggml/LICENSE.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>

// Called only by our generated, opt-in donor overlay. The codec handle mutex
// serializes use; the overlay detaches/frees this owned pool during reset.
void drama3_codec_set_threads(ggml_backend_t backend, ggml_threadpool_t * pool,
                             int * pool_threads, int threads) {
    if (!backend || !pool || !pool_threads || threads < 1 || threads > 256) {
        throw std::runtime_error("invalid persistent codec CPU threadpool inputs");
    }
    if (!*pool || *pool_threads != threads) {
        ggml_backend_cpu_set_threadpool(backend, nullptr);
        if (*pool) ggml_threadpool_free(*pool);
        *pool = nullptr;
        *pool_threads = 0;
        auto parameters = ggml_threadpool_params_default(threads);
        // Sleeping workers must not compete with model generation between
        // reference encoding and waveform decoding.
        parameters.poll = 0;
        *pool = ggml_threadpool_new(&parameters);
        if (!*pool) throw std::runtime_error("cannot allocate codec CPU threadpool");
        *pool_threads = threads;
        ggml_backend_cpu_set_threadpool(backend, *pool);
    }
    ggml_backend_cpu_set_n_threads(backend, threads);
}

void drama3_codec_free_threads(ggml_backend_t backend, ggml_threadpool_t & pool) {
    if (!pool) return;
    ggml_backend_cpu_set_threadpool(backend, nullptr);
    ggml_threadpool_free(pool);
    pool = nullptr;
}

namespace {
// Fish Speech 1ff3c3ed2f80d3106a83909dd25b83129c3d816f,
// models/dac/modded_dac.py apply_rotary_emb: adjacent pairs with stored,
// BF16-rounded cosine/sine, upcast to F32 before multiplying F32 inputs.
void cached_rope(ggml_tensor * output, const ggml_tensor * input,
                 const ggml_tensor * positions, const ggml_tensor * cache,
                 int worker, int workers, void *) {
    const auto * x = static_cast<const float *>(input->data);
    auto * result = static_cast<float *>(output->data);
    const auto * ids = static_cast<const int32_t *>(positions->data);
    const auto * values = static_cast<const ggml_bf16_t *>(cache->data);
    const int64_t width = input->ne[0], heads = input->ne[1];
    for (int64_t row = worker; row < heads * input->ne[2]; row += workers) {
        const int64_t position = ids[row / heads];
        // Emit nonfinite output on invalid runtime position; bridge finite gate
        // reports the failure without indexing outside the cache.
        for (int64_t pair = 0; pair < width / 2; ++pair) {
            const int64_t offset = row * width + pair * 2;
            if (position < 0 || position >= cache->ne[2]) {
                result[offset] = result[offset + 1] = NAN;
                continue;
            }
            const int64_t cache_offset = position * width + pair * 2;
            const float cosine = ggml_bf16_to_fp32(values[cache_offset]);
            const float sine = ggml_bf16_to_fp32(values[cache_offset + 1]);
            result[offset] = x[offset] * cosine - x[offset + 1] * sine;
            result[offset + 1] = x[offset] * sine + x[offset + 1] * cosine;
        }
    }
}
} // namespace

ggml_tensor * drama3_codec_rope(ggml_context * context,
    ggml_tensor * input, ggml_tensor * positions, ggml_tensor * cache) {
    if (!cache || input->type != GGML_TYPE_F32 || positions->type != GGML_TYPE_I32 ||
        cache->type != GGML_TYPE_BF16 || !ggml_is_contiguous(input) ||
        !ggml_is_contiguous(positions) || !ggml_is_contiguous(cache) ||
        input->ne[0] % 2 || input->ne[3] != 1 || cache->ne[0] != 2 ||
        cache->ne[1] != input->ne[0] / 2 || cache->ne[3] != 1 ||
        ggml_nelements(positions) != input->ne[2]) {
        throw std::runtime_error("invalid F32 input/I32 positions/BF16 codec RoPE cache geometry");
    }
    return ggml_map_custom3(context, input, positions, cache, cached_rope, GGML_N_TASKS_MAX, nullptr);
}

extern "C" ggml_tensor * drama3_conv_1d(ggml_context * context,
    ggml_tensor * kernel, ggml_tensor * input, int stride, int padding, int dilation) {
    auto * columns = ggml_im2col(context, kernel, input, stride, 0, padding, 0,
                                dilation, 0, false, GGML_TYPE_F32);
    auto * result = ggml_mul_mat(context,
        ggml_reshape_2d(context, columns, columns->ne[0], columns->ne[2] * columns->ne[1]),
        ggml_reshape_2d(context, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    return ggml_reshape_3d(context, result, columns->ne[1], kernel->ne[2], columns->ne[2]);
}

extern "C" ggml_tensor * drama3_conv_1d_dw(ggml_context * context,
    ggml_tensor * kernel, ggml_tensor * input, int stride, int padding, int dilation) {
    auto * reshaped = ggml_reshape_4d(context, input, input->ne[0], 1, input->ne[1], input->ne[2]);
    auto * columns = ggml_im2col(context, kernel, reshaped, stride, 0, padding, 0,
                                dilation, 0, false, GGML_TYPE_F32);
    auto * result = ggml_mul_mat(context, columns, kernel);
    return ggml_reshape_3d(context, result, result->ne[0], result->ne[2], 1);
}

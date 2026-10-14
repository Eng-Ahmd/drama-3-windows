// F32 attention primitives only: no projection, persistent cache, or synthesis.
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

template<typename T>
std::vector<T> read_values(const std::filesystem::path & path, size_t count) {
    if (std::filesystem::file_size(path) != count * sizeof(T)) {
        throw std::runtime_error("incorrect input size: " + path.string());
    }
    std::vector<T> values(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(values.data()),
                static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream) throw std::runtime_error("input read failed: " + path.string());
    for (const auto value : values) {
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::runtime_error("nonfinite input: " + path.string());
        }
    }
    return values;
}

ggml_tensor * load_tensor(ggml_context * context, const std::filesystem::path & path,
                         int64_t width, int64_t heads, int64_t tokens) {
    auto * tensor = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, heads, tokens);
    const auto values = read_values<float>(path, static_cast<size_t>(width * heads * tokens));
    std::memcpy(tensor->data, values.data(), values.size() * sizeof(float));
    return tensor;
}

ggml_tensor * normalized_rope(ggml_context * context, ggml_tensor * input,
                             ggml_tensor * norm, ggml_tensor * positions, int mode) {
    auto * ones = ggml_new_tensor_1d(context, GGML_TYPE_F32, 8);
    auto * values = static_cast<float *>(ones->data);
    for (int index = 0; index < 8; ++index) values[index] = 1.0f;
    auto * effective_norm = ggml_add(context, norm, ones);
    auto * normalized = ggml_mul(context, ggml_rms_norm(context, input, 1e-6f), effective_norm);
    return ggml_rope_ext(context, normalized, positions, nullptr, 4, mode, 3,
                         1e7f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

ggml_tensor * attention(ggml_context * context, ggml_tensor * q, ggml_tensor * k,
                       ggml_tensor * v, ggml_tensor * gate) {
    // [D,H,T] -> [D,T,H]. KV head count 1 broadcasts to query head count 2.
    auto * query = ggml_cont(context, ggml_permute(context, q, 0, 2, 1, 3));
    auto * key = ggml_cont(context, ggml_permute(context, k, 0, 2, 1, 3));
    auto * scores = ggml_mul_mat(context, key, query); // [Tkey,Tquery,H]
    auto * causal = ggml_diag_mask_inf(context, scores, 0);
    auto * probabilities = ggml_soft_max_ext(context, causal, nullptr, 1.0f / std::sqrt(8.0f), 0.0f);
    auto * value = ggml_cont(context, ggml_permute(context, v, 1, 2, 0, 3)); // [T,D,Hkv]
    auto * attended = ggml_mul_mat(context, value, probabilities); // [D,T,Hq]
    auto * result = ggml_cont(context, ggml_permute(context, attended, 0, 2, 1, 3));
    return ggml_mul(context, result, ggml_sigmoid(context, gate));
}

void write_output(const std::filesystem::path & path, const ggml_tensor * output) {
    const auto * values = static_cast<const float *>(output->data);
    for (int index = 0; index < 48; ++index) {
        if (!std::isfinite(values[index])) throw std::runtime_error("nonfinite output");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(values), 48 * sizeof(float));
    if (!stream) throw std::runtime_error("output write failed");
}

void run(const std::filesystem::path & directory, int mode) {
    Context compute(ggml_init({8 * 1024 * 1024, nullptr, false}), ggml_free);
    if (!compute) throw std::runtime_error("compute context allocation failed");
    auto * context = compute.get();
    auto * q = load_tensor(context, directory / "q.f32", 8, 2, 3);
    auto * k = load_tensor(context, directory / "k.f32", 8, 1, 3);
    auto * v = load_tensor(context, directory / "v.f32", 8, 1, 3);
    auto * gate = load_tensor(context, directory / "gate.f32", 8, 2, 3);
    auto * qnorm = load_tensor(context, directory / "qnorm.f32", 8, 1, 1);
    auto * knorm = load_tensor(context, directory / "knorm.f32", 8, 1, 1);
    const auto ids = read_values<int32_t>(directory / "positions.i32", 3);
    if (ids != std::vector<int32_t>{0, 1, 2}) throw std::runtime_error("positions must be [0,1,2]");
    auto * positions = ggml_new_tensor_1d(context, GGML_TYPE_I32, 3);
    std::memcpy(positions->data, ids.data(), ids.size() * sizeof(int32_t));
    auto * rotated_q = normalized_rope(context, q, qnorm, positions, mode);
    auto * rotated_k = normalized_rope(context, k, knorm, positions, mode);
    auto * output = attention(context, rotated_q, rotated_k, v, gate);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(context, graph, 8) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("attention graph execution failed");
    }
    write_output(directory / "output.f32", output);
    std::cout << "PASS: F32 attention primitives, rotary mode " << mode << ", 48 outputs\n";
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: drama3-attention-fixture INPUTDIR neox|interleaved");
        const std::string mode = argv[2];
        if (mode != "neox" && mode != "interleaved") throw std::runtime_error("invalid rotary mode");
        run(argv[1], mode == "neox" ? 2 : 0);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

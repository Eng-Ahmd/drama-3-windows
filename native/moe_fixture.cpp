// CPU routed-expert fixture only: no router, shared expert, or speech pipeline.
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using File = std::unique_ptr<gguf_context, decltype(&gguf_free)>;

template<typename T>
std::vector<T> read_input(const char * path, size_t count) {
    if (std::filesystem::file_size(path) != count * sizeof(T)) {
        throw std::runtime_error(std::string("incorrect input length: ") + path);
    }
    std::vector<T> values(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(values.data()),
                static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream) {
        throw std::runtime_error("cannot read fixture input");
    }
    return values;
}

ggml_tensor * bank(ggml_context * context, const char * projection) {
    const std::string name = "model.layers.4.mlp.experts_bank." +
                             std::string(projection) + ".weight";
    auto * tensor = ggml_get_tensor(context, name.c_str());
    const bool down = std::string(projection) == "down_proj";
    if (!tensor || tensor->type != GGML_TYPE_Q4_K ||
        tensor->ne[0] != (down ? 512 : 2048) ||
        tensor->ne[1] != (down ? 2048 : 512) ||
        tensor->ne[2] != 256 || tensor->ne[3] != 1) {
        throw std::runtime_error("invalid layer-4 Q4_K bank: " + name);
    }
    return tensor;
}

void run(char ** argv) {
    const auto input = read_input<float>(argv[2], 2048);
    const auto ids = read_input<int32_t>(argv[3], 8);
    const auto weights = read_input<float>(argv[4], 8);
    for (const auto value : input) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite input");
    }
    std::set<int32_t> unique_ids;
    for (size_t index = 0; index < ids.size(); ++index) {
        if (ids[index] < 0 || ids[index] >= 256 || !unique_ids.insert(ids[index]).second) {
            throw std::runtime_error("expert IDs must be unique and in [0,255]");
        }
        if (!std::isfinite(weights[index]) || weights[index] < 0) {
            throw std::runtime_error("mixing weights must be finite and nonnegative");
        }
    }
    // Refuse the full artifact: this runner deliberately loads one bounded layer.
    if (std::filesystem::file_size(argv[1]) > 512ULL * 1024 * 1024) {
        throw std::runtime_error("fixture GGUF exceeds 512 MiB limit");
    }
    ggml_context * raw = nullptr;
    File file(gguf_init_from_file(argv[1], {false, &raw}), gguf_free);
    Context model(raw, ggml_free);
    if (!file || !model) throw std::runtime_error("cannot load layer GGUF");
    auto * gate_bank = bank(model.get(), "gate_proj");
    auto * up_bank = bank(model.get(), "up_proj");
    auto * down_bank = bank(model.get(), "down_proj");
    Context compute(ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
    if (!compute) throw std::runtime_error("cannot allocate compute context");
    auto * context = compute.get();
    auto * x = ggml_new_tensor_3d(context, GGML_TYPE_F32, 2048, 1, 1);
    auto * selected = ggml_new_tensor_2d(context, GGML_TYPE_I32, 8, 1);
    auto * mixing = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, 8, 1);
    std::memcpy(x->data, input.data(), input.size() * sizeof(float));
    std::memcpy(selected->data, ids.data(), ids.size() * sizeof(int32_t));
    std::memcpy(mixing->data, weights.data(), weights.size() * sizeof(float));
    auto * gate = ggml_mul_mat_id(context, gate_bank, x, selected);
    auto * up = ggml_mul_mat_id(context, up_bank, x, selected);
    auto * hidden = ggml_mul(context, ggml_silu(context, gate), up);
    auto * down = ggml_mul_mat_id(context, down_bank, hidden, selected);
    auto * weighted = ggml_mul(context, down, mixing);
    auto * rows = ggml_cont(context, ggml_transpose(context, weighted));
    auto * output = ggml_sum_rows(context, rows);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(context, graph, 8) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("CPU graph execution failed");
    }
    const auto * values = static_cast<const float *>(output->data);
    for (size_t index = 0; index < input.size(); ++index) {
        if (!std::isfinite(values[index])) throw std::runtime_error("nonfinite output");
    }
    std::ofstream stream(argv[5], std::ios::binary | std::ios::trunc);
    stream.write(reinterpret_cast<const char *>(values), 2048 * sizeof(float));
    if (!stream) throw std::runtime_error("cannot write output");
    std::cout << "PASS: layer 4 routed MoE, 8 experts, CPU 8 threads, 2048 F32 outputs\n";
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 6) {
            std::cerr << "usage: drama3-moe-fixture MODEL.gguf INPUT.f32 IDS.i32 MIX.f32 OUTPUT.f32\n";
            return 1;
        }
        run(argv);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

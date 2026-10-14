// CPU component fixture, not a model runtime. All files are little-endian F32.
// q/k/v: GGML [D,H,T,1], equivalent numpy C-order [T,H,D].
// q/k must already be L2-normalized; GGML applies 1/sqrt(D) to output.
// g/beta: [1,H,T,1], numpy [T,H]. g is log decay, not exp(decay).
// state: [D,D,H,1], numpy [H,value,key] (reference state transposed).
// result: activations [T,H,value], then final state [H,value,key].
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
constexpr int64_t kDim = 8;
constexpr int64_t kHeads = 2;
constexpr int64_t kTokens = 3;
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

void read_tensor(const std::filesystem::path &path, ggml_tensor *tensor) {
    const size_t bytes = ggml_nbytes(tensor);
    if (std::filesystem::file_size(path) != bytes) {
        throw std::runtime_error("Incorrect input size: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.read(static_cast<char *>(tensor->data), bytes)) {
        throw std::runtime_error("Cannot read: " + path.string());
    }
    const auto *values = static_cast<const float *>(tensor->data);
    for (int64_t index = 0; index < ggml_nelements(tensor); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error("Nonfinite input: " + path.string());
        }
    }
}

ggml_tensor *load(ggml_context *context, const std::filesystem::path &directory,
                  const char *name, int64_t dim0, int64_t dim1,
                  int64_t dim2) {
    auto *tensor = ggml_new_tensor_4d(context, GGML_TYPE_F32, dim0, dim1,
                                    dim2, 1);
    ggml_set_name(tensor, name);
    read_tensor(directory / (std::string(name) + ".f32"), tensor);
    return tensor;
}

void write_result(const std::filesystem::path &directory, ggml_tensor *result) {
    const auto *values = static_cast<const float *>(result->data);
    for (int64_t index = 0; index < ggml_nelements(result); ++index) {
        if (!std::isfinite(values[index])) {
            throw std::runtime_error("Nonfinite GDN output");
        }
    }
    std::ofstream output(directory / "result.f32", std::ios::binary);
    if (!output.write(static_cast<const char *>(result->data),
                      ggml_nbytes(result))) {
        throw std::runtime_error("Cannot write result.f32");
    }
    output.close();
    if (!output) {
        throw std::runtime_error("Cannot flush result.f32");
    }
}

void run_fixture(const std::filesystem::path &directory) {
    const uint32_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1) {
        throw std::runtime_error("Fixture requires little-endian host");
    }
    ggml_cpu_init();
    Context context(ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
    if (!context) {
        throw std::runtime_error("Cannot initialize GGML context");
    }
    auto *q = load(context.get(), directory, "q", kDim, kHeads, kTokens);
    auto *k = load(context.get(), directory, "k", kDim, kHeads, kTokens);
    auto *v = load(context.get(), directory, "v", kDim, kHeads, kTokens);
    auto *g = load(context.get(), directory, "g", 1, kHeads, kTokens);
    auto *beta = load(context.get(), directory, "beta", 1, kHeads, kTokens);
    auto *state = load(context.get(), directory, "state", kDim, kDim, kHeads);
    auto *result = ggml_gated_delta_net(context.get(), q, k, v, g, beta, state);
    auto *graph = ggml_new_graph(context.get());
    ggml_build_forward_expand(graph, result);
    if (ggml_graph_compute_with_ctx(context.get(), graph, 8) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GDN graph computation failed");
    }
    write_result(directory, result);
    std::cout << "D=8 H=2 T=3: 48 activation + 128 state floats written\n";
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: drama3-gdn-fixture INPUT_DIRECTORY\n";
        return 2;
    }
    try {
        run_fixture(std::filesystem::path(argv[1]));
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "GDN fixture: " << error.what() << '\n';
        return 1;
    }
}

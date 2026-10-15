// Windows CPU expert bridge. Model data stays in a read-only file mapping.
#define NOMINMAX
#include <windows.h>
#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef DRAMA3_CUDA
#include "ggml-cuda.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Context = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using File = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ThreadPool = std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)>;

#ifdef DRAMA3_CUDA
struct CudaGraph {
    Context context{nullptr, ggml_free};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator{nullptr, ggml_gallocr_free};
    ggml_cgraph * graph = nullptr;
    ggml_tensor * output = nullptr;
};
#endif

struct Model {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    const unsigned char * view = nullptr;
    uint64_t size = 0;
    File metadata{nullptr, gguf_free};
    Context tensors{nullptr, ggml_free};
    std::map<int, std::map<std::string, ggml_tensor *>> banks;
    std::mutex mutex;
    ThreadPool cpu_pool{nullptr, ggml_threadpool_free};
    int cpu_threads = 0;
    std::vector<uint8_t> cpu_scratch;
#ifdef DRAMA3_CUDA
    ggml_backend_t cuda = nullptr;
    ggml_backend_buffer_t gpu_weights = nullptr;
    Context gpu_context{nullptr, ggml_free};
    std::set<int> gpu_layers;
    std::map<int, std::unique_ptr<CudaGraph>> decode_graphs;
#endif

    ~Model() {
#ifdef DRAMA3_CUDA
        decode_graphs.clear();
        if (gpu_weights) ggml_backend_buffer_free(gpu_weights);
        if (cuda) ggml_backend_free(cuda);
#endif
        if (view) UnmapViewOfFile(view);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
    }
};

void error_text(char * destination, size_t capacity, const char * text) noexcept {
    if (!destination || !capacity) return;
    const size_t length = (std::strlen(text) < capacity - 1) ? std::strlen(text) : capacity - 1;
    std::memcpy(destination, text, length);
    destination[length] = '\0';
}

std::wstring unicode_path(const char * path) {
    if (!path) throw std::runtime_error("null path");
    const int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, nullptr, 0);
    if (!length) throw std::runtime_error("invalid UTF-8 path");
    std::wstring result(static_cast<size_t>(length), L'\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, result.data(), length)) {
        throw std::runtime_error("path conversion failed");
    }
    return result;
}

void map_file(Model & model, const char * path) {
    model.file = CreateFileW(unicode_path(path).c_str(), GENERIC_READ, FILE_SHARE_READ,
                             nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (model.file == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open GGUF");
    LARGE_INTEGER size;
    if (!GetFileSizeEx(model.file, &size) || size.QuadPart <= 0) {
        throw std::runtime_error("cannot determine GGUF size");
    }
    model.size = static_cast<uint64_t>(size.QuadPart);
    model.mapping = CreateFileMappingW(model.file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!model.mapping) throw std::runtime_error("cannot create GGUF mapping");
    model.view = static_cast<const unsigned char *>(
        MapViewOfFile(model.mapping, FILE_MAP_READ, 0, 0, 0));
    if (!model.view) throw std::runtime_error("cannot map GGUF");
}

void metadata_string(const gguf_context * file, const char * key, const char * expected) {
    const auto index = gguf_find_key(file, key);
    if (index < 0 || gguf_get_kv_type(file, index) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(file, index)) != expected) {
        throw std::runtime_error(std::string("invalid GGUF metadata: ") + key);
    }
}

void validate_bank(ggml_tensor * tensor, const std::string & projection) {
    const bool down = projection == "down_proj";
    if (tensor->type != GGML_TYPE_Q4_K || tensor->ne[0] != (down ? 512 : 2048) ||
        tensor->ne[1] != (down ? 2048 : 512) || tensor->ne[2] != 256 || tensor->ne[3] != 1) {
        throw std::runtime_error("invalid Q4_K expert bank geometry");
    }
}

void load_descriptors(Model & model, const char * path) {
    ggml_context * raw = nullptr;
    model.metadata.reset(gguf_init_from_file(path, {true, &raw}));
    model.tensors.reset(raw);
    if (!model.metadata || !model.tensors) throw std::runtime_error("invalid GGUF descriptors");
    metadata_string(model.metadata.get(), "general.architecture", "fish_drama3");
    metadata_string(model.metadata.get(), "fish_drama3.recipe", "Q4_K-experts_Q8_0-backbone");
    const uint64_t start = gguf_get_data_offset(model.metadata.get());
    if (start > model.size) throw std::runtime_error("invalid GGUF data start");
    for (int64_t index = 0; index < gguf_get_n_tensors(model.metadata.get()); ++index) {
        const char * name = gguf_get_tensor_name(model.metadata.get(), index);
        auto * tensor = ggml_get_tensor(model.tensors.get(), name);
        const uint64_t offset = gguf_get_tensor_offset(model.metadata.get(), index);
        const uint64_t bytes = gguf_get_tensor_size(model.metadata.get(), index);
        if (!tensor || offset > model.size - start || bytes > model.size - start - offset ||
            ggml_nbytes(tensor) != bytes) throw std::runtime_error("GGUF payload out of bounds");
        tensor->data = const_cast<unsigned char *>(model.view + start + offset);
    }
}

void collect_banks(Model & model) {
    std::set<std::string> recognized;
    for (int layer = 0; layer < 40; ++layer) {
        std::map<std::string, ggml_tensor *> projections;
        for (const std::string projection : {"gate_proj", "up_proj", "down_proj"}) {
            const std::string name = "model.layers." + std::to_string(layer) +
                                    ".mlp.experts_bank." + projection + ".weight";
            auto * tensor = ggml_get_tensor(model.tensors.get(), name.c_str());
            if (!tensor) continue;
            validate_bank(tensor, projection);
            recognized.insert(name);
            projections.emplace(projection, tensor);
        }
        if (!projections.empty() && projections.size() != 3) {
            throw std::runtime_error("incomplete expert layer");
        }
        if (!projections.empty()) model.banks.emplace(layer, std::move(projections));
    }
    if (model.banks.empty()) throw std::runtime_error("no expert banks");
    for (int64_t index = 0; index < gguf_get_n_tensors(model.metadata.get()); ++index) {
        const std::string name = gguf_get_tensor_name(model.metadata.get(), index);
        if (name.find("experts_bank") != std::string::npos && recognized.count(name) == 0) {
            throw std::runtime_error("unrecognized expert bank name");
        }
    }
}

void validate_inputs(int tokens, const float * x, const int32_t * ids,
                     const float * mix, float * out, int threads) {
    if (tokens < 1 || tokens > 128 || threads < 1 || threads > 256 ||
        !x || !ids || !mix || !out) throw std::runtime_error("invalid call dimensions/pointers");
    for (int index = 0; index < tokens * 2048; ++index) {
        if (!std::isfinite(x[index])) throw std::runtime_error("nonfinite input");
    }
    for (int token = 0; token < tokens; ++token) {
        std::set<int32_t> unique;
        double total = 0.0;
        for (int expert = 0; expert < 8; ++expert) {
            const int index = token * 8 + expert;
            if (ids[index] < 0 || ids[index] >= 256 || !unique.insert(ids[index]).second ||
                !std::isfinite(mix[index]) || mix[index] < 0) {
                throw std::runtime_error("invalid expert IDs or mixing weights");
            }
            total += mix[index];
        }
        if (!(total > 0.0) || !std::isfinite(total)) throw std::runtime_error("invalid mixing total");
    }
}

ggml_tensor * build_graph(ggml_context * context, std::map<std::string, ggml_tensor *> & banks,
                         int tokens, const float * input, const int32_t * ids, const float * mix) {
    auto * x = ggml_new_tensor_3d(context, GGML_TYPE_F32, 2048, 1, tokens);
    auto * selected = ggml_new_tensor_2d(context, GGML_TYPE_I32, 8, tokens);
    auto * mixing = ggml_new_tensor_3d(context, GGML_TYPE_F32, 1, 8, tokens);
    ggml_set_name(x, "input"); ggml_set_name(selected, "selected"); ggml_set_name(mixing, "mixing");
    ggml_set_input(x); ggml_set_input(selected); ggml_set_input(mixing);
    if (!ggml_get_no_alloc(context)) {
        std::memcpy(x->data, input, static_cast<size_t>(tokens) * 2048 * sizeof(float));
        std::memcpy(selected->data, ids, static_cast<size_t>(tokens) * 8 * sizeof(int32_t));
        std::memcpy(mixing->data, mix, static_cast<size_t>(tokens) * 8 * sizeof(float));
    }
    auto * gate = ggml_mul_mat_id(context, banks.at("gate_proj"), x, selected);
    auto * up = ggml_mul_mat_id(context, banks.at("up_proj"), x, selected);
    auto * hidden = ggml_mul(context, ggml_silu(context, gate), up);
    auto * down = ggml_mul_mat_id(context, banks.at("down_proj"), hidden, selected);
    auto * weighted = ggml_mul(context, down, mixing);
    return ggml_sum_rows(context, ggml_cont(context, ggml_transpose(context, weighted)));
}

#ifdef DRAMA3_CUDA
void cuda_weights(Model & model, int device, int count, uint64_t reserve_bytes) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count() || count < 1 ||
        count > static_cast<int>(model.banks.size())) throw std::runtime_error("invalid CUDA device/layer count");
    model.cuda = ggml_backend_cuda_init(device);
    if (!model.cuda) throw std::runtime_error("CUDA initialization failed");
    model.gpu_context.reset(ggml_init({1024 * 1024, nullptr, true}));
    if (!model.gpu_context) throw std::runtime_error("GPU descriptor allocation failed");
    uint64_t required = 0;
    for (auto entry = model.banks.rbegin(); count > 0; ++entry, --count) {
        model.gpu_layers.insert(entry->first);
        for (auto & projection : entry->second) {
            auto * tensor = ggml_dup_tensor(model.gpu_context.get(), projection.second);
            ggml_set_name(tensor, projection.second->name);
            required += ggml_nbytes(tensor);
        }
    }
    size_t available = 0, total = 0;
    ggml_backend_cuda_get_device_memory(device, &available, &total);
    if (reserve_bytes == 0 || reserve_bytes > total) {
        throw std::runtime_error("invalid CUDA memory reserve");
    }
    if (required > available || reserve_bytes > available - required) {
        throw std::runtime_error("CUDA weights would breach the requested free-memory reserve; reduce GPU layers");
    }
    model.gpu_weights = ggml_backend_alloc_ctx_tensors(model.gpu_context.get(), model.cuda);
    if (!model.gpu_weights) throw std::runtime_error("CUDA weight allocation failed");
    ggml_backend_buffer_set_usage(model.gpu_weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (int layer : model.gpu_layers) {
        for (auto & projection : model.banks.at(layer)) {
            auto * source = projection.second;
            auto * target = ggml_get_tensor(model.gpu_context.get(), source->name);
            ggml_backend_tensor_set(target, source->data, 0, ggml_nbytes(source));
            projection.second = target;
        }
    }
}

std::unique_ptr<CudaGraph> create_cuda_graph(Model & model, int layer, int tokens) {
    auto state = std::make_unique<CudaGraph>();
    state->context.reset(ggml_init({4 * 1024 * 1024, nullptr, true}));
    if (!state->context) throw std::runtime_error("CUDA graph descriptor allocation failed");
    auto * context = state->context.get();
    auto * output = build_graph(context, model.banks.at(layer), tokens, nullptr, nullptr, nullptr);
    ggml_set_output(output);
    auto * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);
    for (int index = 0; index < ggml_graph_n_nodes(graph); ++index) {
        if (!ggml_backend_supports_op(model.cuda, ggml_graph_node(graph, index))) {
            throw std::runtime_error("unsupported CUDA graph operator; no CPU fallback");
        }
    }
    state->allocator.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.cuda)));
    if (!state->allocator || !ggml_gallocr_alloc_graph(state->allocator.get(), graph)) {
        throw std::runtime_error("CUDA graph allocation failed");
    }
    state->graph = graph;
    state->output = output;
    return state;
}

void execute_cuda(Model & model, int layer, int tokens, const float * x,
                  const int32_t * ids, const float * mix, float * out) {
    std::unique_ptr<CudaGraph> temporary;
    CudaGraph * state = nullptr;
    if (tokens == 1) {
        auto & cached = model.decode_graphs[layer];
        if (!cached) cached = create_cuda_graph(model, layer, tokens);
        state = cached.get();
    } else {
        temporary = create_cuda_graph(model, layer, tokens);
        state = temporary.get();
    }
    auto * context = state->context.get();
    ggml_backend_tensor_set(ggml_get_tensor(context, "input"), x, 0, static_cast<size_t>(tokens) * 2048 * sizeof(float));
    ggml_backend_tensor_set(ggml_get_tensor(context, "selected"), ids, 0, static_cast<size_t>(tokens) * 8 * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_get_tensor(context, "mixing"), mix, 0, static_cast<size_t>(tokens) * 8 * sizeof(float));
    if (ggml_backend_graph_compute(model.cuda, state->graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("CUDA expert graph execution failed");
    }
    ggml_backend_tensor_get(state->output, out, 0, static_cast<size_t>(tokens) * 2048 * sizeof(float));
    for (int index = 0; index < tokens * 2048; ++index) {
        if (!std::isfinite(out[index])) throw std::runtime_error("nonfinite CUDA output");
    }
}
#endif

void compute_cpu(Model & model, ggml_cgraph * graph, int threads) {
    if (!model.cpu_pool || model.cpu_threads != threads) {
        auto parameters = ggml_threadpool_params_default(threads);
        model.cpu_pool.reset(ggml_threadpool_new(&parameters));
        if (!model.cpu_pool) throw std::runtime_error("CPU threadpool allocation failed");
        model.cpu_threads = threads;
    }
    auto plan = ggml_graph_plan(graph, threads, model.cpu_pool.get());
    model.cpu_scratch.resize(plan.work_size);
    plan.work_data = model.cpu_scratch.data();
    if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("expert CPU graph execution failed");
    }
}

void execute(Model & model, int layer, int tokens, const float * x, const int32_t * ids,
             const float * mix, float * out, int threads) {
    const auto found = model.banks.find(layer);
    if (found == model.banks.end()) throw std::runtime_error("layer banks unavailable");
#ifdef DRAMA3_CUDA
    if (model.gpu_layers.count(layer)) {
        execute_cuda(model, layer, tokens, x, ids, mix, out);
        return;
    }
#endif
    Context compute(ggml_init({64 * 1024 * 1024, nullptr, false}), ggml_free);
    if (!compute) throw std::runtime_error("compute allocation failed");
    auto * output = build_graph(compute.get(), found->second, tokens, x, ids, mix);
    auto * graph = ggml_new_graph(compute.get());
    ggml_build_forward_expand(graph, output);
    compute_cpu(model, graph, threads);
    const auto * values = static_cast<const float *>(output->data);
    for (int index = 0; index < tokens * 2048; ++index) {
        if (!std::isfinite(values[index])) throw std::runtime_error("nonfinite expert output");
    }
    std::memcpy(out, values, static_cast<size_t>(tokens) * 2048 * sizeof(float));
}
} // namespace

extern "C" __declspec(dllexport) void * drama3_open(const char * path, char * err, size_t capacity) {
    try {
        auto model = std::make_unique<Model>();
        map_file(*model, path);
        load_descriptors(*model, path);
        collect_banks(*model);
        error_text(err, capacity, "");
        return model.release();
    } catch (const std::exception & error) {
        error_text(err, capacity, error.what());
    } catch (...) {
        error_text(err, capacity, "unknown open failure");
    }
    return nullptr;
}

extern "C" __declspec(dllexport) void drama3_close(void * handle) {
    // Caller must finish every in-flight call before closing this handle.
    delete static_cast<Model *>(handle);
}

#ifdef DRAMA3_CUDA
extern "C" __declspec(dllexport) void * drama3_open_cuda(const char * path, int device,
    int gpu_layer_count, char * err, size_t capacity) {
    try {
        auto model = std::make_unique<Model>();
        map_file(*model, path);
        load_descriptors(*model, path);
        collect_banks(*model);
        cuda_weights(*model, device, gpu_layer_count, 1024ULL * 1024 * 1024);
        error_text(err, capacity, "");
        return model.release();
    } catch (const std::exception & error) {
        error_text(err, capacity, error.what());
    } catch (...) {
        error_text(err, capacity, "unknown CUDA open failure");
    }
    return nullptr;
}

extern "C" __declspec(dllexport) void * drama3_open_cuda_budget(const char * path, int device,
    int gpu_layer_count, uint64_t reserve_bytes, char * err, size_t capacity) {
    try {
        if (reserve_bytes == 0) throw std::runtime_error("invalid CUDA memory reserve");
        auto model = std::make_unique<Model>();
        map_file(*model, path);
        load_descriptors(*model, path);
        collect_banks(*model);
        cuda_weights(*model, device, gpu_layer_count, reserve_bytes);
        error_text(err, capacity, "");
        return model.release();
    } catch (const std::exception & error) {
        error_text(err, capacity, error.what());
    } catch (...) {
        error_text(err, capacity, "unknown CUDA open failure");
    }
    return nullptr;
}
#endif

extern "C" __declspec(dllexport) int drama3_experts(void * handle, int layer, int tokens,
    const float * x, const int32_t * ids, const float * mix, float * out,
    int threads, char * err, size_t capacity) {
    try {
        if (!handle) throw std::runtime_error("null model handle");
        validate_inputs(tokens, x, ids, mix, out, threads);
        auto & model = *static_cast<Model *>(handle);
        std::lock_guard<std::mutex> lock(model.mutex);
        execute(model, layer, tokens, x, ids, mix, out, threads);
        error_text(err, capacity, "");
        return 0;
    } catch (const std::exception & error) {
        error_text(err, capacity, error.what());
    } catch (...) {
        error_text(err, capacity, "unknown expert failure");
    }
    return -1;
}

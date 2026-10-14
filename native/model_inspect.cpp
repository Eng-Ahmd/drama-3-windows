// Descriptor-only contract validation. This executable does not synthesize audio.
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Ggml = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

int64_t require_key(const gguf_context * context, const char * name, gguf_type type) {
    const auto key = gguf_find_key(context, name);
    if (key < 0 || gguf_get_kv_type(context, key) != type) {
        throw std::runtime_error(std::string("missing/wrong-type metadata: ") + name);
    }
    return key;
}

void require_string(const gguf_context * context, const char * name, const char * value) {
    const auto key = require_key(context, name, GGUF_TYPE_STRING);
    if (std::string(gguf_get_val_str(context, key)) != value) {
        throw std::runtime_error(std::string("unsupported metadata: ") + name);
    }
}

void check_embedding(ggml_context * context, const char * name, bool complete) {
    const auto * tensor = ggml_get_tensor(context, name);
    if (!tensor && !complete) {
        return;
    }
    if (!tensor || tensor->type != GGML_TYPE_Q8_0 || tensor->ne[0] != 2048 ||
        tensor->ne[1] != 252288 || tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        throw std::runtime_error(std::string("invalid/missing Q8_0 embedding/head: ") + name);
    }
}

void inspect(const std::string & path, bool allow_partial) {
    const auto file_size = std::filesystem::file_size(path);
    ggml_context * raw_context = nullptr;
    Gguf file(gguf_init_from_file(path.c_str(), {true, &raw_context}), gguf_free);
    Ggml tensors(raw_context, ggml_free);
    if (!file || !tensors) {
        throw std::runtime_error("GGUF metadata could not be loaded");
    }
    require_string(file.get(), "general.architecture", "fish_drama3");
    require_string(file.get(), "fish_drama3.recipe", "Q4_K-experts_Q8_0-backbone");
    const bool complete = gguf_get_val_bool(file.get(),
        require_key(file.get(), "fish_drama3.complete", GGUF_TYPE_BOOL));
    if (!complete && !allow_partial) {
        throw std::runtime_error("component artifact requires --allow-partial");
    }
    const auto data_start = gguf_get_data_offset(file.get());
    if (data_start > file_size) {
        throw std::runtime_error("data start is past end of file");
    }
    const std::regex bank_pattern(
        R"(^model\.layers\.(\d+)\.mlp\.experts_bank\.(gate_proj|up_proj|down_proj)\.weight$)");
    std::set<std::pair<int, std::string>> banks;
    std::vector<std::pair<uintmax_t, uintmax_t>> intervals;
    for (int64_t index = 0; index < gguf_get_n_tensors(file.get()); ++index) {
        const std::string name = gguf_get_tensor_name(file.get(), index);
        const auto * tensor = ggml_get_tensor(tensors.get(), name.c_str());
        if (!tensor) {
            throw std::runtime_error("missing descriptor: " + name);
        }
        const auto offset = gguf_get_tensor_offset(file.get(), index);
        const auto bytes = gguf_get_tensor_size(file.get(), index);
        const auto remaining = file_size - data_start;
        if (offset > remaining || bytes > remaining - offset || ggml_nbytes(tensor) != bytes) {
            throw std::runtime_error("invalid payload bounds: " + name);
        }
        intervals.emplace_back(offset, offset + bytes);
        std::smatch match;
        if (!std::regex_match(name, match, bank_pattern)) {
            if (name.find("experts_bank") != std::string::npos) {
                throw std::runtime_error("invalid expert bank name: " + name);
            }
            continue;
        }
        const int layer = std::stoi(match[1].str());
        const auto projection = match[2].str();
        const bool down = projection == "down_proj";
        if (layer < 0 || layer >= 40 || !banks.emplace(layer, projection).second ||
            tensor->type != GGML_TYPE_Q4_K || tensor->ne[0] != (down ? 512 : 2048) ||
            tensor->ne[1] != (down ? 2048 : 512) || tensor->ne[2] != 256 ||
            tensor->ne[3] != 1) {
            throw std::runtime_error("invalid Q4_K expert bank: " + name);
        }
    }
    std::sort(intervals.begin(), intervals.end());
    for (size_t index = 1; index < intervals.size(); ++index) {
        if (intervals[index].first < intervals[index - 1].second) {
            throw std::runtime_error("tensor payloads overlap");
        }
    }
    if (complete && banks.size() != 120) {
        throw std::runtime_error("complete artifact requires all 40 x 3 expert banks");
    }
    check_embedding(tensors.get(), "model.embed_tokens.weight", complete);
    check_embedding(tensors.get(), "lm_head.weight", complete);
    std::cout << "PASS: " << (complete ? "complete" : "component") << ", "
              << gguf_get_n_tensors(file.get()) << " tensors, " << banks.size()
              << " Q4_K banks, " << file_size << " bytes; descriptors/bounds only\n";
}
} // namespace

int main(int argc, char ** argv) {
    try {
        bool allow_partial = false;
        std::string path;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument == "--help" && argc == 2) {
                std::cout << "usage: drama3-inspect [--allow-partial] MODEL.gguf\n";
                return 0;
            } else if (argument == "--allow-partial") {
                allow_partial = true;
            } else if (path.empty() && argument != "--help") {
                path = argument;
            } else {
                throw std::runtime_error("usage: drama3-inspect [--allow-partial] MODEL.gguf");
            }
        }
        if (path.empty()) {
            std::cout << "usage: drama3-inspect [--allow-partial] MODEL.gguf\n";
            return argc == 1 ? 1 : 0;
        }
        inspect(path, allow_partial);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

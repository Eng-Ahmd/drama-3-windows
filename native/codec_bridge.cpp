// CPU bridge to the pinned donor codec. Invalid codes are rejected, never clamped.
#include "s2_codec.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {
struct Codec {
    s2::AudioCodec model;
    std::mutex mutex;
};

void error_text(char * destination, size_t capacity, const char * text) noexcept {
    if (!destination || !capacity) return;
    const size_t size = std::strlen(text);
    const size_t copied = size < capacity - 1 ? size : capacity - 1;
    std::memcpy(destination, text, copied);
    destination[copied] = '\0';
}

void require_u32(const gguf_context * context, const char * key, uint32_t expected) {
    const auto index = gguf_find_key(context, key);
    if (index < 0 || gguf_get_kv_type(context, index) != GGUF_TYPE_UINT32 ||
        gguf_get_val_u32(context, index) != expected) {
        throw std::runtime_error(std::string("unsupported codec metadata: ") + key);
    }
}

void validate_metadata(const char * path) {
    if (!path) throw std::runtime_error("null codec path");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
        gguf_init_from_file(path, {true, nullptr}), gguf_free);
    if (!metadata) throw std::runtime_error("cannot read codec GGUF metadata");
    const auto index = gguf_find_key(metadata.get(), "general.architecture");
    if (index < 0 || gguf_get_kv_type(metadata.get(), index) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(metadata.get(), index)) != "fish-speech-codec") {
        throw std::runtime_error("codec requires fish-speech-codec architecture");
    }
    require_u32(metadata.get(), "fish_speech.codec.sample_rate", 44100);
    require_u32(metadata.get(), "fish_speech.codec.frame_length", 2048);
    require_u32(metadata.get(), "fish_speech.codec.quantizer_residual_codebooks", 9);
    require_u32(metadata.get(), "fish_speech.codec.quantizer_semantic_codebook_size", 4096);
    require_u32(metadata.get(), "fish_speech.codec.quantizer_residual_codebook_size", 1024);
}

void validate_caches(s2::AudioCodec & model) {
    for (const auto & entry : {std::make_pair("encoder.block.4.block.5.freqs_cis", 16384),
                              std::make_pair("quantizer.pre_module.freqs_cis", 4096),
                              std::make_pair("quantizer.post_module.freqs_cis", 4096)}) {
        const auto * tensor = ggml_get_tensor(model.weights_ctx(), entry.first);
        if (!tensor || tensor->type != GGML_TYPE_BF16 || tensor->ne[0] != 2 ||
            tensor->ne[1] != 32 || tensor->ne[2] != entry.second || tensor->ne[3] != 1) {
            throw std::runtime_error("missing/invalid source BF16 codec RoPE cache");
        }
    }
}

void validate_inputs(const int32_t * codes, int frames, int threads,
                     float * output, int capacity, int * count) {
    if (!codes || !output || !count || frames < 1 || frames > 4096 ||
        frames > std::numeric_limits<int>::max() / 2048 ||
        threads < 1 || threads > 256 || capacity < frames * 2048) {
        throw std::runtime_error("invalid codec buffers, frames, threads, or capacity");
    }
    for (int book = 0; book < 10; ++book) {
        const int limit = book == 0 ? 4096 : 1024;
        for (int frame = 0; frame < frames; ++frame) {
            const int code = codes[static_cast<size_t>(book) * frames + frame];
            if (code < 0 || code >= limit) {
                throw std::runtime_error("codec index outside semantic/residual codebook range");
            }
        }
    }
}

void decode(Codec & codec, const int32_t * codes, int frames, int threads,
            float * output, int capacity, int * count) {
    std::vector<float> audio;
    if (!codec.model.decode(codes, frames, threads, audio)) {
        throw std::runtime_error("CPU codec decode failed");
    }
    if (audio.empty() || audio.size() > static_cast<size_t>(capacity) ||
        audio.size() > static_cast<size_t>(frames) * 2048) {
        throw std::runtime_error("unexpected codec output sample count");
    }
    for (const float value : audio) {
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite codec audio");
    }
    std::memcpy(output, audio.data(), audio.size() * sizeof(float));
    *count = static_cast<int>(audio.size());
}

void validate_encode(const float * audio, int samples, int threads,
                     int32_t * output, int capacity, int * frames) {
    if (!audio || !output || !frames || samples < 1 || samples > 16384 * 512 ||
        samples > std::numeric_limits<int>::max() - 2048 ||
        threads < 1 || threads > 256) throw std::runtime_error("invalid encoder inputs");
    const int required = ((samples + 2047) / 2048) * 10;
    if (capacity < required) throw std::runtime_error("insufficient encoder code capacity");
    for (int index = 0; index < samples; ++index) {
        if (!std::isfinite(audio[index])) throw std::runtime_error("nonfinite encoder audio");
    }
}

void encode(Codec & codec, const float * audio, int samples, int threads,
            int32_t * output, int capacity, int * frames) {
    std::vector<int32_t> codes;
    int32_t actual_frames = 0;
    if (!codec.model.encode(audio, samples, threads, codes, actual_frames)) {
        throw std::runtime_error("CPU codec encode failed");
    }
    if (actual_frames < 1 || actual_frames > (samples + 2047) / 2048 ||
        codes.size() != static_cast<size_t>(actual_frames) * 10 ||
        codes.size() > static_cast<size_t>(capacity)) {
        throw std::runtime_error("unexpected encoder frame/code count");
    }
    for (int book = 0; book < 10; ++book) {
        const int limit = book == 0 ? 4096 : 1024;
        for (int frame = 0; frame < actual_frames; ++frame) {
            const int code = codes[static_cast<size_t>(book) * actual_frames + frame];
            if (code < 0 || code >= limit) throw std::runtime_error("invalid encoder code range");
        }
    }
    std::memcpy(output, codes.data(), codes.size() * sizeof(int32_t));
    *frames = actual_frames;
}
} // namespace

extern "C" __declspec(dllexport) void * drama3_codec_open(const char * path, char * err, size_t capacity) {
    try {
        validate_metadata(path);
        auto codec = std::make_unique<Codec>();
        if (!codec->model.load(path, -1, s2::BackendType::CPU)) {
            throw std::runtime_error("CPU codec load failed");
        }
        if (codec->model.sample_rate() != 44100 || codec->model.num_codebooks() != 10 ||
            codec->model.samples_per_code_frame() != 2048) {
            throw std::runtime_error("unsupported loaded codec geometry");
        }
        validate_caches(codec->model);
        error_text(err, capacity, "");
        return codec.release();
    } catch (const std::exception & error) {
        error_text(err, capacity, error.what());
    } catch (...) {
        error_text(err, capacity, "unknown codec open failure");
    }
    return nullptr;
}

extern "C" __declspec(dllexport) void drama3_codec_close(void * handle) {
    // Caller must finish in-flight decode calls before closing this handle.
    delete static_cast<Codec *>(handle);
}

extern "C" __declspec(dllexport) int drama3_codec_decode(void * handle, const int32_t * codes,
    int frames, int threads, float * output, int capacity, int * count,
    char * err, size_t error_capacity) {
    if (count) *count = 0;
    try {
        if (!handle) throw std::runtime_error("null codec handle");
        validate_inputs(codes, frames, threads, output, capacity, count);
        auto & codec = *static_cast<Codec *>(handle);
        std::lock_guard<std::mutex> lock(codec.mutex);
        decode(codec, codes, frames, threads, output, capacity, count);
        error_text(err, error_capacity, "");
        return 0;
    } catch (const std::exception & error) {
        error_text(err, error_capacity, error.what());
    } catch (...) {
        error_text(err, error_capacity, "unknown codec decode failure");
    }
    return -1;
}

extern "C" __declspec(dllexport) int drama3_codec_encode(void * handle, const float * audio,
    int samples, int threads, int32_t * output, int capacity, int * frames,
    char * err, size_t error_capacity) {
    if (frames) *frames = 0;
    try {
        if (!handle) throw std::runtime_error("null codec handle");
        validate_encode(audio, samples, threads, output, capacity, frames);
        auto & codec = *static_cast<Codec *>(handle);
        std::lock_guard<std::mutex> lock(codec.mutex);
        encode(codec, audio, samples, threads, output, capacity, frames);
        error_text(err, error_capacity, "");
        return 0;
    } catch (const std::exception & error) {
        error_text(err, error_capacity, error.what());
    } catch (...) {
        error_text(err, error_capacity, "unknown codec encode failure");
    }
    return -1;
}

cmake_minimum_required(VERSION 3.20)
project(drama3_inspect LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "This target requires Windows MSVC and the pinned donor build")
endif()
get_filename_component(DRAMA3_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(DONOR_BUILD "${DRAMA3_ROOT}/builds/s2-cpu" CACHE PATH "Pinned CPU donor build")
add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES
    IMPORTED_IMPLIB "${DONOR_BUILD}/ggml/src/ggml-base.lib"
    IMPORTED_LOCATION "${DONOR_BUILD}/bin/ggml-base.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${DRAMA3_ROOT}/upstream/s2.cpp/ggml/include")
add_executable(drama3-inspect model_inspect.cpp)
target_compile_features(drama3-inspect PRIVATE cxx_std_17)
target_compile_options(drama3-inspect PRIVATE /W4 /WX /EHsc)
target_link_libraries(drama3-inspect PRIVATE ggml-base)
add_custom_command(TARGET drama3-inspect POST_BUILD COMMAND ${CMAKE_COMMAND} -E
    copy_if_different "$<TARGET_FILE:ggml-base>" "$<TARGET_FILE_DIR:drama3-inspect>")
add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES
    IMPORTED_IMPLIB "${DONOR_BUILD}/ggml/src/ggml-cpu.lib"
    IMPORTED_LOCATION "${DONOR_BUILD}/bin/ggml-cpu.dll")
add_executable(drama3-moe-fixture moe_fixture.cpp)
target_compile_features(drama3-moe-fixture PRIVATE cxx_std_17)
target_compile_options(drama3-moe-fixture PRIVATE /W4 /WX /EHsc)
target_link_libraries(drama3-moe-fixture PRIVATE ggml-base ggml-cpu)
foreach(runtime ggml-base ggml-cpu)
    add_custom_command(TARGET drama3-moe-fixture POST_BUILD COMMAND ${CMAKE_COMMAND} -E
        copy_if_different "$<TARGET_FILE:${runtime}>" "$<TARGET_FILE_DIR:drama3-moe-fixture>")
endforeach()
add_library(ggml-core SHARED IMPORTED)
set_target_properties(ggml-core PROPERTIES
    IMPORTED_IMPLIB "${DONOR_BUILD}/ggml/src/ggml.lib"
    IMPORTED_LOCATION "${DONOR_BUILD}/bin/ggml.dll")
# Compile pristine donor files separately so their existing warning policy does
# not weaken /W4 /WX on our bridge code.
set(codec_source "${DRAMA3_ROOT}/upstream/s2.cpp/src/s2_codec.cpp")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${codec_source}")
file(READ "${codec_source}" codec_overlay)
string(REPLACE "\r" "" codec_overlay "${codec_overlay}")
option(DRAMA3_CODEC_CPU_FLASH_ATTENTION "Use F32 CPU fused codec attention (validate before release)" OFF)
if(DRAMA3_CODEC_CPU_FLASH_ATTENTION)
    # Preserve the donor's causal/local support. F16 mask conversion maps its
    # 0/-1e9 entries to 0/-inf, enabling the pinned CPU kernel's masked-tile skip.
    set(old_causal_mask "const bool use_full_causal_mask = force_explicit_causal_mask && !use_window_mask;")
    set(old_attention [=[        ggml_tensor * k_rep = repeat_interleave_heads(ctx, k, n_head / n_local_heads);
        ggml_tensor * v_rep = repeat_interleave_heads(ctx, v, n_head / n_local_heads);

        ggml_tensor * Q  = ggml_permute(ctx, q,     0, 2, 1, 3);
        ggml_tensor * K  = ggml_permute(ctx, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ = mul_mat_checked(ctx, K, Q, "mul_mat:codec_kq");
        ggml_tensor * KQs = ggml_scale(ctx, KQ, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        ggml_tensor * KQm;
        if (inp.mask) {
            KQm = ggml_add(ctx, KQs, repeat_checked(ctx, inp.mask, KQs, "repeat:attn_mask"));
        } else {
            KQm = ggml_diag_mask_inf(ctx, KQs, 0);
        }
        ggml_tensor * KQf = ggml_soft_max(ctx, KQm);

        ggml_tensor * V       = ggml_cont(ctx, ggml_permute(ctx, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(ctx, V, KQf, "mul_mat:codec_kqv");
        ggml_tensor * KQVm    = ggml_permute(ctx, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(ctx, KQVm, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_size, seq_len));]=])
    foreach(fragment IN ITEMS old_causal_mask old_attention)
        string(LENGTH "${codec_overlay}" before_length)
        string(REPLACE "${${fragment}}" "" unmatched "${codec_overlay}")
        string(LENGTH "${unmatched}" after_length)
        string(LENGTH "${${fragment}}" fragment_length)
        math(EXPR matches "(${before_length} - ${after_length}) / ${fragment_length}")
        if(NOT matches EQUAL 1)
            message(FATAL_ERROR "Pinned codec fused-attention overlay expected one ${fragment}, got ${matches}")
        endif()
    endforeach()
    string(REPLACE "${old_causal_mask}" "const bool use_full_causal_mask = !use_window_mask;"
        codec_overlay "${codec_overlay}")
    set(new_attention [=[        ggml_tensor * Q = ggml_permute(ctx, q, 0, 2, 1, 3);
        ggml_tensor * K = ggml_permute(ctx, k, 0, 2, 1, 3);
        ggml_tensor * V = ggml_permute(ctx, v, 0, 2, 1, 3);
        ggml_tensor * mask = ggml_cast(ctx, inp.mask, GGML_TYPE_F16);
        ggml_tensor * fused = ggml_flash_attn_ext(ctx, Q, K, V, mask,
            1.0f / std::sqrt(static_cast<float>(head_dim)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fused, GGML_PREC_F32);
        ggml_tensor * attn_cur = ggml_reshape_2d(ctx, fused, q_size, seq_len);]=])
    string(REPLACE "${old_attention}" "${new_attention}" codec_overlay "${codec_overlay}")
endif()
option(DRAMA3_CODEC_CPU_GRAPH_CACHE "Reuse the pinned decoder graph on CPU (benchmark before release)" OFF)
if(DRAMA3_CODEC_CPU_GRAPH_CACHE)
    set(old_decode_dispatch "if (!ggml_backend_is_cpu(impl_->backend) &&\n        run_cached_decode_graph(*impl_, codes, n_frames, n_threads, audio_out))")
    string(FIND "${codec_overlay}" "${old_decode_dispatch}" decode_dispatch_match)
    if(decode_dispatch_match EQUAL -1)
        message(FATAL_ERROR "Pinned codec decode dispatch no longer matches the CPU-cache overlay")
    endif()
    string(LENGTH "${old_decode_dispatch}" decode_dispatch_length)
    math(EXPR after_decode_dispatch "${decode_dispatch_match} + ${decode_dispatch_length}")
    string(SUBSTRING "${codec_overlay}" ${after_decode_dispatch} -1 decode_dispatch_remainder)
    string(FIND "${decode_dispatch_remainder}" "${old_decode_dispatch}" duplicate_decode_dispatch)
    if(NOT duplicate_decode_dispatch EQUAL -1)
        message(FATAL_ERROR "Unexpected duplicate codec decode dispatch")
    endif()
    string(REPLACE "${old_decode_dispatch}"
        "if (run_cached_decode_graph(*impl_, codes, n_frames, n_threads, audio_out))"
        codec_overlay "${codec_overlay}")
endif()
option(DRAMA3_CODEC_CPU_THREADPOOL "Reuse a nonspinning CPU codec threadpool (benchmark before release)" OFF)
if(DRAMA3_CODEC_CPU_THREADPOOL)
    function(drama3_codec_replace_checked old new expected)
        string(LENGTH "${codec_overlay}" before_length)
        string(REPLACE "${old}" "" without_matches "${codec_overlay}")
        string(LENGTH "${without_matches}" after_length)
        string(LENGTH "${old}" old_length)
        math(EXPR matches "(${before_length} - ${after_length}) / ${old_length}")
        if(NOT matches EQUAL expected)
            message(FATAL_ERROR "Pinned codec threadpool overlay expected ${expected} matches, got ${matches}: ${old}")
        endif()
        string(REPLACE "${old}" "${new}" replacement "${codec_overlay}")
        set(codec_overlay "${replacement}" PARENT_SCOPE)
    endfunction()
    drama3_codec_replace_checked("codec_decode_cache decode_cache;"
        "codec_decode_cache decode_cache;\n    ggml_threadpool_t cpu_pool = nullptr;\n    int cpu_pool_threads = 0;" 1)
    drama3_codec_replace_checked("ggml_backend_cpu_set_n_threads(impl_->backend, n_threads);"
        "drama3_codec_set_threads(impl_->backend, &impl_->cpu_pool, &impl_->cpu_pool_threads, n_threads);" 4)
    drama3_codec_replace_checked("ggml_backend_cpu_set_n_threads(impl.backend, n_threads);"
        "drama3_codec_set_threads(impl.backend, &impl.cpu_pool, &impl.cpu_pool_threads, n_threads);" 1)
    drama3_codec_replace_checked("reset_decode_cache(impl.decode_cache, false);"
        "reset_decode_cache(impl.decode_cache, false);\n    drama3_codec_free_threads(impl.backend, impl.cpu_pool);" 1)
    drama3_codec_replace_checked("#include <stdexcept>"
        "#include <stdexcept>\nvoid drama3_codec_set_threads(ggml_backend_t, ggml_threadpool_t *, int *, int);\nvoid drama3_codec_free_threads(ggml_backend_t, ggml_threadpool_t &);" 1)
endif()
foreach(kind q k)
    set(old_rope "${kind} = ggml_rope_ext(ctx, ${kind}, inp.positions, nullptr, head_dim, 0,\n                          block_size, rope_base, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f);")
    string(FIND "${codec_overlay}" "${old_rope}" match)
    if(match EQUAL -1)
        message(FATAL_ERROR "Pinned codec ${kind} RoPE call no longer matches")
    endif()
    string(LENGTH "${old_rope}" match_length)
    math(EXPR after_match "${match} + ${match_length}")
    string(SUBSTRING "${codec_overlay}" ${after_match} -1 remainder)
    string(FIND "${remainder}" "${old_rope}" duplicate)
    if(NOT duplicate EQUAL -1)
        message(FATAL_ERROR "Unexpected duplicate codec ${kind} RoPE call")
    endif()
    string(REPLACE "${old_rope}" "${kind} = drama3_codec_rope(ctx, ${kind}, inp.positions, req(prefix + \".freqs_cis\"));" codec_overlay "${codec_overlay}")
endforeach()
string(REPLACE "../include/" "${DRAMA3_ROOT}/upstream/s2.cpp/include/" codec_overlay "${codec_overlay}")
string(REPLACE "#include <stdexcept>" "#include <stdexcept>\nggml_tensor * drama3_codec_rope(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *);" codec_overlay "${codec_overlay}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/s2_codec_overlay.cpp" "${codec_overlay}")
add_library(drama3-codec-donor OBJECT
    "${CMAKE_CURRENT_BINARY_DIR}/s2_codec_overlay.cpp"
    "${DRAMA3_ROOT}/upstream/s2.cpp/src/s2_log.cpp")
target_compile_features(drama3-codec-donor PRIVATE cxx_std_17)
target_compile_options(drama3-codec-donor PRIVATE /W3 /EHsc)
target_compile_definitions(drama3-codec-donor PRIVATE
    ggml_conv_1d=drama3_conv_1d ggml_conv_1d_dw=drama3_conv_1d_dw)
target_include_directories(drama3-codec-donor PRIVATE
    "${DRAMA3_ROOT}/upstream/s2.cpp/include" "${DRAMA3_ROOT}/upstream/s2.cpp/src")
target_link_libraries(drama3-codec-donor PRIVATE ggml-base ggml-cpu ggml-core)
add_library(drama3-codec-bridge SHARED codec_bridge.cpp codec_ops.cpp $<TARGET_OBJECTS:drama3-codec-donor>)
target_compile_features(drama3-codec-bridge PRIVATE cxx_std_17)
target_compile_options(drama3-codec-bridge PRIVATE /W4 /WX /EHsc)
target_include_directories(drama3-codec-bridge PRIVATE
    "${DRAMA3_ROOT}/upstream/s2.cpp/include")
target_link_libraries(drama3-codec-bridge PRIVATE ggml-base ggml-cpu ggml-core)
foreach(runtime ggml-base ggml-cpu ggml-core)
    add_custom_command(TARGET drama3-codec-bridge POST_BUILD COMMAND ${CMAKE_COMMAND} -E
        copy_if_different "$<TARGET_FILE:${runtime}>" "$<TARGET_FILE_DIR:drama3-codec-bridge>")
endforeach()
add_library(drama3-experts-bridge SHARED experts_bridge.cpp)
target_compile_features(drama3-experts-bridge PRIVATE cxx_std_17)
target_compile_options(drama3-experts-bridge PRIVATE /W4 /WX /EHsc)
target_link_libraries(drama3-experts-bridge PRIVATE ggml-base ggml-cpu)
foreach(runtime ggml-base ggml-cpu)
    add_custom_command(TARGET drama3-experts-bridge POST_BUILD COMMAND ${CMAKE_COMMAND} -E
        copy_if_different "$<TARGET_FILE:${runtime}>" "$<TARGET_FILE_DIR:drama3-experts-bridge>")
endforeach()
add_executable(drama3-attention-fixture attention_fixture.cpp)
target_compile_features(drama3-attention-fixture PRIVATE cxx_std_17)
target_compile_options(drama3-attention-fixture PRIVATE /W4 /WX /EHsc)
target_link_libraries(drama3-attention-fixture PRIVATE ggml-base ggml-cpu)
foreach(runtime ggml-base ggml-cpu)
    add_custom_command(TARGET drama3-attention-fixture POST_BUILD COMMAND ${CMAKE_COMMAND} -E
        copy_if_different "$<TARGET_FILE:${runtime}>" "$<TARGET_FILE_DIR:drama3-attention-fixture>")
endforeach()
add_executable(drama3-gdn-fixture gdn_fixture.cpp)
target_compile_features(drama3-gdn-fixture PRIVATE cxx_std_17)
target_compile_options(drama3-gdn-fixture PRIVATE /W4 /WX /EHsc)
target_link_libraries(drama3-gdn-fixture PRIVATE ggml-base ggml-cpu)
foreach(runtime ggml-base ggml-cpu)
    add_custom_command(TARGET drama3-gdn-fixture POST_BUILD COMMAND ${CMAKE_COMMAND} -E
        copy_if_different "$<TARGET_FILE:${runtime}>" "$<TARGET_FILE_DIR:drama3-gdn-fixture>")
endforeach()

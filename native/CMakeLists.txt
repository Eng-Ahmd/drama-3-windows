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

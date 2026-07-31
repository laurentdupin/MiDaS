cmake_minimum_required(VERSION 3.20)
project(midas_native_c VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MIDAS_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)

if(MIDAS_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(MIDAS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${MIDAS_GENERATED_DIR}")

    function(midas_compile_shader NAME)
        set(SOURCE_NAME "${NAME}")
        set(EXTRA_ARGUMENTS)
        if(ARGC GREATER 1)
            set(SOURCE_NAME "${ARGV1}")
        endif()
        if(ARGC GREATER 2)
            list(APPEND EXTRA_ARGUMENTS "${ARGV2}")
        endif()
        set(SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SOURCE_NAME}.glsl")
        set(SPIRV "${MIDAS_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${MIDAS_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -O -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                ${EXTRA_ARGUMENTS}
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS "${SOURCE}"
            VERBATIM)
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}"
                "-DVARIABLE=midas_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS
                "${SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(MIDAS_SHADER_HEADERS
            ${MIDAS_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()

    midas_compile_shader(
        conv2d_grouped conv2d_grouped -DNO_BATCH_NORM=1)
    midas_compile_shader(
        conv2d_pointwise4 conv2d_pointwise4 -DNO_BATCH_NORM=1)
    midas_compile_shader(
        conv2d_pointwise_gemm conv2d_pointwise_gemm -DNO_BATCH_NORM=1)
    midas_compile_shader(
        conv2d_pointwise_gemm_residual conv2d_pointwise_gemm
        -DADD_RESIDUAL=1)
    midas_compile_shader(
        conv2d_depthwise3 conv2d_depthwise3 -DNO_BATCH_NORM=1)
    midas_compile_shader(
        conv2d_spatial4 conv2d_spatial4 -DNO_BATCH_NORM=1)
    midas_compile_shader(
        conv2d_spatial4_tiled conv2d_spatial4_tiled
        -DNO_BATCH_NORM=1)
    midas_compile_shader(
        conv2d_spatial4_tiled_relu conv2d_spatial4_tiled
        -DRELU_INPUT=1)
    midas_compile_shader(batch_norm_activation)
    midas_compile_shader(activation)
    midas_compile_shader(add)
    midas_compile_shader(bilinear)
endif()

add_library(midas_native SHARED
    src/midas_native.cpp
    src/inferbridge_harness.cpp
    src/cpu_executor.cpp
    src/image.cpp
    src/model.cpp)
target_include_directories(midas_native
    PUBLIC include
    PRIVATE src)
target_compile_definitions(
    midas_native PRIVATE
        MIDAS_BUILD_DLL
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
target_compile_definitions(midas_native PRIVATE
    $<$<BOOL:${MIDAS_WITH_VULKAN}>:MIDAS_WITH_VULKAN=1>)
set_target_properties(midas_native PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
if(MSVC)
    target_compile_options(midas_native PRIVATE /W4 /permissive-)
    set_property(TARGET midas_native PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(MIDAS_WITH_VULKAN)
    target_sources(midas_native PRIVATE
        src/vulkan.cpp
        src/vulkan_operators.cpp
        src/vulkan_executor.cpp
        ${MIDAS_SHADER_HEADERS})
    target_include_directories(midas_native PRIVATE
        "${MIDAS_GENERATED_DIR}")
    target_link_libraries(midas_native PRIVATE Vulkan::Vulkan)
endif()

include(CTest)
add_executable(midas_model_probe
    tools/model_probe.cpp
    src/model.cpp)
target_include_directories(midas_model_probe PRIVATE include src)
if(MSVC)
    target_compile_options(midas_model_probe PRIVATE /W4 /permissive-)
    set_property(TARGET midas_model_probe PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(MIDAS_WITH_VULKAN)
    add_executable(midas_gpu_probe tools/gpu_probe.c)
    target_link_libraries(midas_gpu_probe PRIVATE midas_native)
    target_include_directories(midas_gpu_probe PRIVATE include)
endif()

if(BUILD_TESTING)
    add_executable(midas_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(midas_c_abi_smoke PRIVATE midas_native)
    target_include_directories(midas_c_abi_smoke PRIVATE include)
    add_test(NAME midas_c_abi_smoke COMMAND midas_c_abi_smoke)
    add_executable(
        midas_harness_abi_smoke tests/harness_abi_smoke.c)
    target_link_libraries(
        midas_harness_abi_smoke PRIVATE midas_native)
    target_include_directories(
        midas_harness_abi_smoke PRIVATE include)
    add_test(
        NAME midas_harness_abi_smoke
        COMMAND midas_harness_abi_smoke)
    add_executable(
        midas_harness_full_graph tests/harness_full_graph.cpp)
    target_link_libraries(
        midas_harness_full_graph PRIVATE midas_native)
    target_include_directories(
        midas_harness_full_graph PRIVATE include)
    add_test(
        NAME midas_harness_full_graph
        COMMAND midas_harness_full_graph)
    set_tests_properties(
        midas_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(midas_image_tests
        tests/image_tests.cpp
        src/image.cpp)
    target_include_directories(midas_image_tests PRIVATE src)
    if(MSVC)
        target_compile_options(midas_image_tests PRIVATE /W4 /permissive-)
        set_property(TARGET midas_image_tests PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    add_test(NAME midas_image_tests COMMAND midas_image_tests)

    add_executable(midas_model_metadata_tests
        tests/model_metadata.cpp
        src/model.cpp)
    target_include_directories(midas_model_metadata_tests PRIVATE include src)
    if(MSVC)
        target_compile_options(midas_model_metadata_tests PRIVATE
            /W4 /permissive-)
        set_property(TARGET midas_model_metadata_tests PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    add_test(NAME midas_model_metadata_tests
        COMMAND midas_model_metadata_tests)
endif()

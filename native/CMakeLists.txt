cmake_minimum_required(VERSION 3.20)
project(midas_native_c VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(midas_native SHARED
    src/midas_native.cpp
    src/cpu_executor.cpp
    src/model.cpp)
target_include_directories(midas_native
    PUBLIC include
    PRIVATE src)
target_compile_definitions(midas_native PRIVATE MIDAS_BUILD_DLL)
set_target_properties(midas_native PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES)
if(MSVC)
    target_compile_options(midas_native PRIVATE /W4 /permissive-)
    set_property(TARGET midas_native PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
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

if(BUILD_TESTING)
    add_executable(midas_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(midas_c_abi_smoke PRIVATE midas_native)
    target_include_directories(midas_c_abi_smoke PRIVATE include)
    add_test(NAME midas_c_abi_smoke COMMAND midas_c_abi_smoke)

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

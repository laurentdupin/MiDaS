#ifndef MIDAS_NATIVE_H
#define MIDAS_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(MIDAS_BUILD_DLL)
#    define MIDAS_API __declspec(dllexport)
#  else
#    define MIDAS_API __declspec(dllimport)
#  endif
#  define MIDAS_CALL __cdecl
#else
#  define MIDAS_API __attribute__((visibility("default")))
#  define MIDAS_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MIDAS_ABI_VERSION 2u

typedef struct midas_context midas_context;

typedef enum midas_status {
    MIDAS_STATUS_OK = 0,
    MIDAS_STATUS_INVALID_ARGUMENT = 1,
    MIDAS_STATUS_MODEL_IO = 2,
    MIDAS_STATUS_MODEL_FORMAT = 3,
    MIDAS_STATUS_VULKAN_UNAVAILABLE = 4,
    MIDAS_STATUS_OUT_OF_MEMORY = 5,
    MIDAS_STATUS_INFERENCE_FAILED = 6,
    MIDAS_STATUS_BUFFER_TOO_SMALL = 7,
    MIDAS_STATUS_UNSUPPORTED = 8,
    MIDAS_STATUS_INTERNAL_ERROR = 9
} midas_status;

typedef enum midas_model_kind {
    MIDAS_MODEL_V21_SMALL_256 = 0
} midas_model_kind;

typedef struct midas_image_shape {
    int32_t width;
    int32_t height;
} midas_image_shape;

enum {
    MIDAS_GPU_CAP_VULKAN_GRAPH = 1ull << 0u,
    MIDAS_GPU_CAP_HOST_TENSOR_UPLOAD = 1ull << 1u,
    MIDAS_GPU_CAP_HOST_DEPTH_READBACK = 1ull << 2u
};

typedef struct midas_gpu_capabilities {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t flags;
    uint32_t maximum_in_flight_jobs;
    char device_name[256];
} midas_gpu_capabilities;

MIDAS_API uint32_t MIDAS_CALL midas_abi_version(void);
MIDAS_API const char* MIDAS_CALL midas_version_string(void);
MIDAS_API const char* MIDAS_CALL midas_status_string(midas_status status);
MIDAS_API const char* MIDAS_CALL midas_last_error(void);
MIDAS_API midas_status MIDAS_CALL midas_create(
    const char* model_path_utf8,
    midas_model_kind model,
    midas_context** context);
MIDAS_API midas_status MIDAS_CALL midas_create_vulkan(
    const char* model_path_utf8,
    midas_model_kind model,
    int32_t vulkan_device_index,
    midas_context** context);
MIDAS_API midas_status MIDAS_CALL midas_probe_gpu_capabilities(
    int32_t vulkan_device_index,
    midas_gpu_capabilities* capabilities);
MIDAS_API void MIDAS_CALL midas_destroy(midas_context* context);
MIDAS_API midas_status MIDAS_CALL midas_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    midas_image_shape* network_shape);
MIDAS_API midas_status MIDAS_CALL midas_infer_bgr8(
    midas_context* context,
    const uint8_t* bgr,
    int32_t width,
    int32_t height,
    int64_t stride_bytes,
    int32_t input_size,
    float* depth,
    uint64_t depth_elements);
/*
 * Exact InferBridge MiDaS Small worker boundary. Input is a BGRA8 capture;
 * the first three bytes retain their BGR ordering as model channels. Output
 * is min/max-normalized uint8 depth at midas_get_network_shape dimensions,
 * matching the Python worker (no resize back to the capture dimensions).
 */
MIDAS_API midas_status MIDAS_CALL midas_inferbridge_bgra8_u8(
    midas_context* context,
    const uint8_t* bgra,
    int32_t width,
    int32_t height,
    int64_t stride_bytes,
    int32_t input_size,
    uint8_t* depth,
    uint64_t depth_elements);
MIDAS_API midas_status MIDAS_CALL midas_infer_tensor_f32(
    midas_context* context,
    const float* normalized_rgb_chw,
    int32_t width,
    int32_t height,
    float* depth,
    uint64_t depth_elements);

#ifdef __cplusplus
}
#endif

#endif

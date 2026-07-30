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

#define MIDAS_ABI_VERSION 1u

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

/*
 * The complete inference lifecycle is added only after the graph passes the
 * PyTorch CPU accuracy gate. This first ABI slice intentionally exposes no
 * false inference or GPU capability.
 */
MIDAS_API uint32_t MIDAS_CALL midas_abi_version(void);
MIDAS_API const char* MIDAS_CALL midas_version_string(void);
MIDAS_API const char* MIDAS_CALL midas_status_string(midas_status status);
MIDAS_API const char* MIDAS_CALL midas_last_error(void);
MIDAS_API midas_status MIDAS_CALL midas_create(
    const char* model_path_utf8,
    midas_model_kind model,
    midas_context** context);
MIDAS_API void MIDAS_CALL midas_destroy(midas_context* context);
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

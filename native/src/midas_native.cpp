#include "midas_native.h"

#include <string>

namespace {
thread_local std::string last_error;
}

extern "C" {

uint32_t MIDAS_CALL midas_abi_version(void) {
    return MIDAS_ABI_VERSION;
}

const char* MIDAS_CALL midas_version_string(void) {
    return "0.1.0-foundation";
}

const char* MIDAS_CALL midas_status_string(midas_status status) {
    switch (status) {
        case MIDAS_STATUS_OK: return "ok";
        case MIDAS_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case MIDAS_STATUS_MODEL_IO: return "model I/O error";
        case MIDAS_STATUS_MODEL_FORMAT: return "invalid model format";
        case MIDAS_STATUS_VULKAN_UNAVAILABLE: return "Vulkan unavailable";
        case MIDAS_STATUS_OUT_OF_MEMORY: return "out of memory";
        case MIDAS_STATUS_INFERENCE_FAILED: return "inference failed";
        case MIDAS_STATUS_BUFFER_TOO_SMALL: return "buffer too small";
        case MIDAS_STATUS_UNSUPPORTED: return "unsupported";
        case MIDAS_STATUS_INTERNAL_ERROR: return "internal error";
        default: return "unknown status";
    }
}

const char* MIDAS_CALL midas_last_error(void) {
    return last_error.c_str();
}

}

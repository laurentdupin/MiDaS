#include "midas_native.h"

#include "cpu_executor.h"
#include "image.h"

#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

struct midas_context {
    std::unique_ptr<midas_native::CpuExecutor> executor;
    midas_native::ImageScratch image_scratch;
    std::vector<float> network_input;
    std::vector<float> network_depth;
};

namespace {
thread_local std::string last_error;

midas_status fail(midas_status status, const char* message) {
    last_error = message ? message : "";
    return status;
}

template <typename Function>
midas_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return MIDAS_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(MIDAS_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& error) {
        return fail(MIDAS_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(MIDAS_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(MIDAS_STATUS_INTERNAL_ERROR, "unknown internal error");
    }
}
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

midas_status MIDAS_CALL midas_create(
    const char* model_path_utf8,
    midas_model_kind model,
    midas_context** context) {
    if (context == nullptr) {
        return fail(MIDAS_STATUS_INVALID_ARGUMENT, "context is null");
    }
    *context = nullptr;
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0' ||
        model != MIDAS_MODEL_V21_SMALL_256) {
        return fail(MIDAS_STATUS_INVALID_ARGUMENT, "invalid create options");
    }
    return protect([&] {
        auto result = std::make_unique<midas_context>();
        result->executor =
            std::make_unique<midas_native::CpuExecutor>(model_path_utf8);
        *context = result.release();
    });
}

void MIDAS_CALL midas_destroy(midas_context* context) {
    delete context;
}

midas_status MIDAS_CALL midas_get_network_shape(
    int32_t image_width,
    int32_t image_height,
    int32_t input_size,
    midas_image_shape* network_shape) {
    if (network_shape == nullptr) {
        return fail(
            MIDAS_STATUS_INVALID_ARGUMENT, "network shape is null");
    }
    return protect([&] {
        const midas_native::ImageShape shape =
            midas_native::network_shape(
                image_width, image_height, input_size);
        network_shape->width = shape.width;
        network_shape->height = shape.height;
    });
}

midas_status MIDAS_CALL midas_infer_bgr8(
    midas_context* context,
    const uint8_t* bgr,
    int32_t width,
    int32_t height,
    int64_t stride_bytes,
    int32_t input_size,
    float* depth,
    uint64_t depth_elements) {
    if (context == nullptr || context->executor == nullptr ||
        bgr == nullptr || depth == nullptr ||
        width <= 0 || height <= 0 ||
        stride_bytes < int64_t(width) * 3 ||
        input_size <= 0 ||
        depth_elements < uint64_t(width) * height) {
        return fail(
            MIDAS_STATUS_INVALID_ARGUMENT, "invalid BGR8 inference input");
    }
    return protect([&] {
        const midas_native::ImageShape network =
            midas_native::network_shape(width, height, input_size);
        midas_native::preprocess_bgr8(
            bgr,
            width,
            height,
            static_cast<std::ptrdiff_t>(stride_bytes),
            network,
            context->image_scratch,
            context->network_input);
        context->network_depth.resize(
            static_cast<std::size_t>(
                uint64_t(network.width) * network.height));
        context->executor->infer(
            context->network_input.data(),
            static_cast<std::uint32_t>(network.width),
            static_cast<std::uint32_t>(network.height),
            context->network_depth.data(),
            context->network_depth.size());
        midas_native::resize_depth_bicubic(
            context->network_depth.data(),
            network,
            depth,
            {width, height});
    });
}

midas_status MIDAS_CALL midas_infer_tensor_f32(
    midas_context* context,
    const float* normalized_rgb_chw,
    int32_t width,
    int32_t height,
    float* depth,
    uint64_t depth_elements) {
    if (context == nullptr || context->executor == nullptr ||
        width <= 0 || height <= 0) {
        return fail(MIDAS_STATUS_INVALID_ARGUMENT, "invalid inference input");
    }
    return protect([&] {
        context->executor->infer(
            normalized_rgb_chw,
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            depth,
            depth_elements);
    });
}

}

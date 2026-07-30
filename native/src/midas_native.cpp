#include "midas_native.h"

#include "cpu_executor.h"
#include "image.h"
#if defined(MIDAS_WITH_VULKAN)
#  include "vulkan_executor.h"
#endif

#include <algorithm>
#include <memory>
#include <new>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct midas_context {
    std::unique_ptr<midas_native::CpuExecutor> executor;
#if defined(MIDAS_WITH_VULKAN)
    std::unique_ptr<midas_native::VulkanExecutor> vulkan_executor;
#endif
    midas_native::ImageScratch image_scratch;
    std::vector<std::uint8_t> packed_capture;
    std::vector<float> network_input;
    std::vector<float> network_depth;

    void infer(
        const float* input,
        std::uint32_t width,
        std::uint32_t height,
        float* depth,
        std::uint64_t depth_elements) {
#if defined(MIDAS_WITH_VULKAN)
        if (vulkan_executor) {
            vulkan_executor->infer(
                input, width, height, depth, depth_elements);
            return;
        }
#endif
        if (!executor) {
            throw std::runtime_error("MiDaS context has no executor");
        }
        executor->infer(input, width, height, depth, depth_elements);
    }
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

midas_status MIDAS_CALL midas_create_vulkan(
    const char* model_path_utf8,
    midas_model_kind model,
    int32_t vulkan_device_index,
    midas_context** context) {
    if (context == nullptr) {
        return fail(MIDAS_STATUS_INVALID_ARGUMENT, "context is null");
    }
    *context = nullptr;
    if (model_path_utf8 == nullptr || model_path_utf8[0] == '\0' ||
        model != MIDAS_MODEL_V21_SMALL_256 ||
        vulkan_device_index < 0) {
        return fail(MIDAS_STATUS_INVALID_ARGUMENT, "invalid create options");
    }
#if defined(MIDAS_WITH_VULKAN)
    return protect([&] {
        auto result = std::make_unique<midas_context>();
        result->vulkan_executor =
            std::make_unique<midas_native::VulkanExecutor>(
                model_path_utf8,
                static_cast<std::uint32_t>(vulkan_device_index));
        *context = result.release();
    });
#else
    return fail(
        MIDAS_STATUS_VULKAN_UNAVAILABLE,
        "this DLL was built without Vulkan");
#endif
}

midas_status MIDAS_CALL midas_probe_gpu_capabilities(
    int32_t vulkan_device_index,
    midas_gpu_capabilities* capabilities) {
    if (capabilities == nullptr ||
        capabilities->struct_size < sizeof(*capabilities) ||
        capabilities->abi_version != MIDAS_ABI_VERSION ||
        vulkan_device_index < 0) {
        return fail(
            MIDAS_STATUS_INVALID_ARGUMENT,
            "invalid GPU capability probe");
    }
#if defined(MIDAS_WITH_VULKAN)
    return protect([&] {
        midas_native::VulkanContext context(
            static_cast<std::uint32_t>(vulkan_device_index));
        *capabilities = {};
        capabilities->struct_size = sizeof(*capabilities);
        capabilities->abi_version = MIDAS_ABI_VERSION;
        capabilities->flags =
            MIDAS_GPU_CAP_VULKAN_GRAPH |
            MIDAS_GPU_CAP_HOST_TENSOR_UPLOAD |
            MIDAS_GPU_CAP_HOST_DEPTH_READBACK;
        capabilities->maximum_in_flight_jobs = 1;
        const std::string& name = context.device_name();
        const std::size_t bytes = std::min(
            name.size(), sizeof(capabilities->device_name) - 1);
        std::memcpy(capabilities->device_name, name.data(), bytes);
        capabilities->device_name[bytes] = '\0';
    });
#else
    return fail(
        MIDAS_STATUS_VULKAN_UNAVAILABLE,
        "this DLL was built without Vulkan");
#endif
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
    if (context == nullptr ||
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
        context->infer(
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

midas_status MIDAS_CALL midas_inferbridge_bgra8_u8(
    midas_context* context,
    const uint8_t* bgra,
    int32_t width,
    int32_t height,
    int64_t stride_bytes,
    int32_t input_size,
    uint8_t* depth,
    uint64_t depth_elements) {
    if (context == nullptr || bgra == nullptr || depth == nullptr ||
        width <= 0 || height <= 0 ||
        stride_bytes < int64_t(width) * 4 || input_size <= 0) {
        return fail(
            MIDAS_STATUS_INVALID_ARGUMENT,
            "invalid InferBridge BGRA8 inference input");
    }
    return protect([&] {
        const midas_native::ImageShape network =
            midas_native::network_shape(width, height, input_size);
        const std::uint64_t network_elements =
            std::uint64_t(network.width) * network.height;
        if (depth_elements < network_elements) {
            throw std::invalid_argument(
                "InferBridge depth output is too small");
        }
        context->packed_capture.resize(
            static_cast<std::size_t>(
                std::uint64_t(width) * height * 3u));
        for (int y = 0; y < height; ++y) {
            const std::uint8_t* source =
                bgra + static_cast<std::int64_t>(y) * stride_bytes;
            std::uint8_t* destination =
                context->packed_capture.data() +
                static_cast<std::size_t>(y) * width * 3u;
            for (int x = 0; x < width; ++x) {
                // preprocess_bgr8 swaps BGR to RGB. Reversing here preserves
                // the worker's first-three-byte BGR-as-model-channel quirk.
                destination[static_cast<std::size_t>(x) * 3u] =
                    source[static_cast<std::size_t>(x) * 4u + 2u];
                destination[static_cast<std::size_t>(x) * 3u + 1u] =
                    source[static_cast<std::size_t>(x) * 4u + 1u];
                destination[static_cast<std::size_t>(x) * 3u + 2u] =
                    source[static_cast<std::size_t>(x) * 4u];
            }
        }
        midas_native::preprocess_bgr8(
            context->packed_capture.data(), width, height,
            static_cast<std::ptrdiff_t>(width) * 3,
            network, context->image_scratch, context->network_input);
        context->network_depth.resize(
            static_cast<std::size_t>(network_elements));
        context->infer(
            context->network_input.data(),
            static_cast<std::uint32_t>(network.width),
            static_cast<std::uint32_t>(network.height),
            context->network_depth.data(),
            context->network_depth.size());
        const auto bounds = std::minmax_element(
            context->network_depth.begin(), context->network_depth.end());
        const float minimum = *bounds.first;
        const float range = *bounds.second - minimum;
        if (!(range > 0.0f)) {
            std::fill_n(
                depth, static_cast<std::size_t>(network_elements),
                std::uint8_t{0});
            return;
        }
        for (std::uint64_t index = 0u;
             index < network_elements; ++index) {
            const float value =
                (context->network_depth[static_cast<std::size_t>(index)] -
                    minimum) /
                range * 255.0f;
            depth[index] = static_cast<std::uint8_t>(
                std::clamp(value, 0.0f, 255.0f));
        }
    });
}

midas_status MIDAS_CALL midas_infer_tensor_f32(
    midas_context* context,
    const float* normalized_rgb_chw,
    int32_t width,
    int32_t height,
    float* depth,
    uint64_t depth_elements) {
    if (context == nullptr ||
        width <= 0 || height <= 0) {
        return fail(MIDAS_STATUS_INVALID_ARGUMENT, "invalid inference input");
    }
    return protect([&] {
        context->infer(
            normalized_rgb_chw,
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height),
            depth,
            depth_elements);
    });
}

}

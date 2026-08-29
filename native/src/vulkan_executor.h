#pragma once

#include "model.h"
#include "vulkan.h"
#include "vulkan_operators.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace midas_native {

class VulkanExecutor {
public:
    VulkanExecutor(const std::string& model_path, std::uint32_t device_index);

    void infer(
        const float* normalized_rgb_chw,
        std::uint32_t width,
        std::uint32_t height,
        float* depth,
        std::uint64_t depth_elements);
    VulkanBuffer infer_device(
        VulkanBuffer normalized_rgb_chw,
        std::uint32_t width,
        std::uint32_t height);
    VulkanContext& context() { return context_; }
    const VulkanContext& context() const { return context_; }
    const std::string& device_name() const {
        return context_.device_name();
    }

private:
    struct QuantizedWeight {
        VulkanBuffer packed;
        VulkanBuffer scales;
    };
    struct Tensor {
        std::uint32_t channels = 0;
        std::uint32_t height = 0;
        std::uint32_t width = 0;
        VulkanBuffer buffer;
    };

    const VulkanBuffer& weight(const std::string& name) const;
    Tensor conv(
        const Tensor& input,
        const std::string& weight_name,
        const char* bias_name,
        std::uint32_t stride,
        bool same_stride2,
        std::uint32_t groups = 1,
        const char* batch_norm_prefix = nullptr,
        std::uint32_t activation = 0,
        bool relu_input = false,
        const VulkanBuffer* residual = nullptr);
    Tensor batch_norm_activation(
        const Tensor& input,
        const std::string& prefix,
        std::uint32_t activation);
    Tensor activation(const Tensor& input, std::uint32_t kind);
    Tensor add(const Tensor& left, const Tensor& right);
    Tensor resize(
        const Tensor& input,
        std::uint32_t width,
        std::uint32_t height,
        bool align_corners);
    Tensor inverted(
        const Tensor& input,
        const std::string& prefix,
        std::uint32_t stride,
        bool residual);
    Tensor depthwise_separable(
        const Tensor& input,
        const std::string& prefix);
    Tensor residual_unit(
        const Tensor& input,
        const std::string& prefix);
    Tensor fusion(
        Tensor path,
        const Tensor* skip,
        const std::string& prefix);

    ModelFile model_;
    VulkanContext context_;
    VulkanOperators operators_;
    std::unordered_map<std::string, VulkanBuffer> weights_;
    std::unordered_map<std::string, VulkanBuffer> fp16_weights_;
    std::unordered_map<std::string, QuantizedWeight> int8_weights_;
    bool fp16_enabled_ = false;
    bool int8_enabled_ = false;
    VulkanBuffer zero_bias_;
};

}  // namespace midas_native

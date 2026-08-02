#pragma once

#include "vulkan.h"

#include <cstdint>

namespace midas_native {

class GpuIo {
public:
    explicit GpuIo(VulkanContext& context);
    void preprocess(
        VulkanBuffer& destination,
        const VulkanImage& source,
        std::uint32_t destination_width,
        std::uint32_t destination_height);
    void resize_depth(
        VulkanImage& destination,
        const VulkanBuffer& source,
        std::uint32_t source_width,
        std::uint32_t source_height);
    void normalize_relative(VulkanBuffer& depth, std::uint32_t count);

private:
    VulkanContext& context_;
    VulkanPipeline preprocess_;
    VulkanPipeline resize_depth_;
    VulkanPipeline resize_depth_buffer_;
    VulkanPipeline reduce_minmax_;
    VulkanPipeline normalize_relative_;
};

}  // namespace midas_native

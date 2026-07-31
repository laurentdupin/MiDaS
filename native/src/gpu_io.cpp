#include "gpu_io.h"

#include "preprocess_texture_spv.h"
#include "resize_depth_image_spv.h"

#include <stdexcept>

namespace midas_native {

GpuIo::GpuIo(VulkanContext& context)
    : context_(context),
      preprocess_(context.create_pipeline(
          midas_preprocess_texture_spv,
          midas_preprocess_texture_spv_size,
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT},
          4u * sizeof(std::uint32_t))),
      resize_depth_(context.create_pipeline(
          midas_resize_depth_image_spv,
          midas_resize_depth_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT},
          4u * sizeof(std::uint32_t))) {
    preprocess_.set_debug_name("midas_preprocess_texture");
    resize_depth_.set_debug_name("midas_resize_depth_image");
}

void GpuIo::preprocess(
    VulkanBuffer& destination,
    const VulkanImage& source,
    std::uint32_t destination_width,
    std::uint32_t destination_height) {
    const std::uint64_t required =
        static_cast<std::uint64_t>(destination_width) *
        destination_height * 3u * sizeof(float);
    if (source.width() == 0u || source.height() == 0u ||
        destination_width == 0u || destination_height == 0u ||
        destination.size() < required) {
        throw std::invalid_argument("invalid MiDaS GPU preprocessing shape");
    }
    struct Parameters {
        std::uint32_t source_width;
        std::uint32_t source_height;
        std::uint32_t destination_width;
        std::uint32_t destination_height;
    } parameters{
        source.width(), source.height(),
        destination_width, destination_height};
    context_.dispatch_image_to_buffer(
        preprocess_, source, destination,
        &parameters, sizeof(parameters),
        (destination_width + 7u) / 8u,
        (destination_height + 7u) / 8u);
}

void GpuIo::resize_depth(
    VulkanImage& destination,
    const VulkanBuffer& source,
    std::uint32_t source_width,
    std::uint32_t source_height) {
    if (source_width == 0u || source_height == 0u ||
        destination.width() == 0u || destination.height() == 0u ||
        destination.format() != VK_FORMAT_R32_SFLOAT ||
        source.size() < static_cast<std::uint64_t>(source_width) *
            source_height * sizeof(float)) {
        throw std::invalid_argument("invalid MiDaS GPU depth output shape");
    }
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t output_width;
        std::uint32_t output_height;
    } parameters{
        source_width, source_height,
        destination.width(), destination.height()};
    context_.dispatch_buffer_to_image(
        resize_depth_, source, destination,
        &parameters, sizeof(parameters),
        (destination.width() + 7u) / 8u,
        (destination.height() + 7u) / 8u);
}

}  // namespace midas_native

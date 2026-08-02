#include "gpu_io.h"

#include "preprocess_texture_spv.h"
#include "resize_depth_image_spv.h"
#include "resize_depth_buffer_spv.h"
#include "reduce_minmax_spv.h"
#include "normalize_relative_spv.h"

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
          4u * sizeof(std::uint32_t))),
      resize_depth_buffer_(context.create_pipeline(
          midas_resize_depth_buffer_spv,
          midas_resize_depth_buffer_spv_size, 2,
          4u * sizeof(std::uint32_t))),
      reduce_minmax_(context.create_pipeline(
          midas_reduce_minmax_spv, midas_reduce_minmax_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT}, 4)),
      normalize_relative_(context.create_pipeline(
          midas_normalize_relative_spv,
          midas_normalize_relative_spv_size, 2, 4)) {
    preprocess_.set_debug_name("midas_preprocess_texture");
    resize_depth_.set_debug_name("midas_resize_depth_image");
    resize_depth_buffer_.set_debug_name("midas_resize_depth_buffer");
    reduce_minmax_.set_debug_name("midas_reduce_minmax");
    normalize_relative_.set_debug_name("midas_normalize_relative");
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
    VulkanBuffer resized = context_.create_device_buffer(
        static_cast<std::uint64_t>(destination.width()) *
        destination.height() * sizeof(float));
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t output_width;
        std::uint32_t output_height;
    } parameters{
        source_width, source_height,
        destination.width(), destination.height()};
    context_.dispatch(
        resize_depth_buffer_, {&source, &resized},
        &parameters, sizeof(parameters),
        (destination.width() + 7u) / 8u,
        (destination.height() + 7u) / 8u);
    const std::uint32_t count = destination.width() * destination.height();
    normalize_relative(resized, count);
    Parameters copy_parameters{
        destination.width(), destination.height(),
        destination.width(), destination.height()};
    context_.dispatch_buffer_to_image(
        resize_depth_, resized, destination,
        &copy_parameters, sizeof(copy_parameters),
        (destination.width() + 7u) / 8u,
        (destination.height() + 7u) / 8u);
}

void GpuIo::normalize_relative(VulkanBuffer& depth, std::uint32_t count) {
    if (count == 0u || depth.size() <
        static_cast<std::uint64_t>(count) * sizeof(float))
        throw std::invalid_argument("invalid MiDaS relative depth buffer");
    VulkanBuffer range = context_.create_device_buffer(2u * sizeof(float));
    context_.dispatch(
        reduce_minmax_, {&depth, &range}, &count, sizeof(count), 1u);
    context_.dispatch(
        normalize_relative_, {&depth, &range}, &count, sizeof(count),
        (count + 255u) / 256u);
}

}  // namespace midas_native

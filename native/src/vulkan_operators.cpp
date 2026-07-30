#include "vulkan_operators.h"

#include "activation_spv.h"
#include "add_spv.h"
#include "batch_norm_activation_spv.h"
#include "bilinear_spv.h"
#include "conv2d_grouped_spv.h"

#include <vector>

namespace midas_native {
namespace {

std::uint32_t divide_up(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}

}  // namespace

VulkanOperators::VulkanOperators(VulkanContext& context)
    : context_(context),
      conv_(context.create_pipeline(
          midas_conv2d_grouped_spv,
          midas_conv2d_grouped_spv_size,
          4,
          52)),
      batch_norm_activation_(context.create_pipeline(
          midas_batch_norm_activation_spv,
          midas_batch_norm_activation_spv_size,
          6,
          16)),
      activation_(context.create_pipeline(
          midas_activation_spv,
          midas_activation_spv_size,
          2,
          8)),
      add_(context.create_pipeline(
          midas_add_spv,
          midas_add_spv_size,
          3,
          4)),
      bilinear_(context.create_pipeline(
          midas_bilinear_spv,
          midas_bilinear_spv_size,
          2,
          24)) {
    conv_.set_debug_name("midas_conv2d_grouped");
    batch_norm_activation_.set_debug_name(
        "midas_batch_norm_activation");
    activation_.set_debug_name("midas_activation");
    add_.set_debug_name("midas_add");
    bilinear_.set_debug_name("midas_bilinear");
}

void VulkanOperators::conv(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& weight,
    const VulkanBuffer& bias,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t input_channels,
    std::uint32_t output_width,
    std::uint32_t output_height,
    std::uint32_t output_channels,
    std::uint32_t kernel_height,
    std::uint32_t kernel_width,
    std::uint32_t stride,
    std::int32_t padding_top,
    std::int32_t padding_left,
    std::uint32_t groups,
    bool has_bias) {
    struct Parameters {
        std::uint32_t input_width;
        std::uint32_t input_height;
        std::uint32_t input_channels;
        std::uint32_t output_width;
        std::uint32_t output_height;
        std::uint32_t output_channels;
        std::uint32_t kernel_height;
        std::uint32_t kernel_width;
        std::uint32_t stride;
        std::int32_t padding_top;
        std::int32_t padding_left;
        std::uint32_t groups;
        std::uint32_t has_bias;
    } parameters{
        input_width, input_height, input_channels,
        output_width, output_height, output_channels,
        kernel_height, kernel_width, stride,
        padding_top, padding_left, groups, has_bias ? 1u : 0u};
    context_.dispatch(
        conv_,
        {&output, &input, &weight, &bias},
        &parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        output_channels);
}

void VulkanOperators::batch_norm_activation(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    const VulkanBuffer& gamma,
    const VulkanBuffer& beta,
    const VulkanBuffer& mean,
    const VulkanBuffer& variance,
    std::uint32_t count,
    std::uint32_t plane,
    std::uint32_t activation) {
    struct Parameters {
        std::uint32_t count;
        std::uint32_t plane;
        std::uint32_t activation;
        float epsilon;
    } parameters{count, plane, activation, 0.001f};
    context_.dispatch(
        batch_norm_activation_,
        {&output, &input, &gamma, &beta, &mean, &variance},
        &parameters,
        sizeof(parameters),
        divide_up(count, 256));
}

void VulkanOperators::activation(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t count,
    std::uint32_t kind) {
    const std::uint32_t parameters[2] = {count, kind};
    context_.dispatch(
        activation_,
        {&output, &input},
        parameters,
        sizeof(parameters),
        divide_up(count, 256));
}

void VulkanOperators::add(
    VulkanBuffer& output,
    const VulkanBuffer& left,
    const VulkanBuffer& right,
    std::uint32_t count) {
    context_.dispatch(
        add_,
        {&output, &left, &right},
        &count,
        sizeof(count),
        divide_up(count, 256));
}

void VulkanOperators::resize(
    VulkanBuffer& output,
    const VulkanBuffer& input,
    std::uint32_t input_width,
    std::uint32_t input_height,
    std::uint32_t output_width,
    std::uint32_t output_height,
    std::uint32_t channels,
    bool align_corners) {
    const std::uint32_t parameters[6] = {
        input_width,
        input_height,
        output_width,
        output_height,
        channels,
        align_corners ? 1u : 0u};
    context_.dispatch(
        bilinear_,
        {&output, &input},
        parameters,
        sizeof(parameters),
        divide_up(output_width, 8),
        divide_up(output_height, 8),
        channels);
}

}  // namespace midas_native

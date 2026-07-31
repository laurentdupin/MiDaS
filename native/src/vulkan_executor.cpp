#include "vulkan_executor.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace midas_native {
namespace {

std::uint64_t elements(
    std::uint32_t channels,
    std::uint32_t height,
    std::uint32_t width) {
    return std::uint64_t(channels) * height * width;
}

std::string tensor_weight(const std::string& prefix) {
    return prefix + ".weight";
}

std::string tensor_bias(const std::string& prefix) {
    return prefix + ".bias";
}

std::uint32_t checked_count(std::uint64_t count) {
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("MiDaS tensor is too large");
    }
    return static_cast<std::uint32_t>(count);
}

}  // namespace

VulkanExecutor::VulkanExecutor(
    const std::string& model_path,
    std::uint32_t device_index)
    : model_(model_path, MIDAS_MODEL_V21_SMALL_256),
      context_(device_index),
      operators_(context_),
      zero_bias_(context_.create_device_buffer(sizeof(float))) {
    const float zero = 0.0f;
    context_.upload(zero_bias_, &zero, sizeof(zero));
    weights_.reserve(model_.tensor_count() + 64);
    for (std::string_view name_view : model_.tensor_names()) {
        const std::string name(name_view);
        const TensorView& tensor = model_.tensor(name);
        VulkanBuffer buffer = context_.create_device_buffer(
            tensor.elements * sizeof(float));
        context_.upload(
            buffer,
            tensor.data,
            static_cast<std::size_t>(
                tensor.elements * sizeof(float)));
        weights_.emplace(name, std::move(buffer));
    }
    const auto fold_batch_norm =
        [&](const std::string& weight_name, const std::string& prefix) {
        const TensorView& source = model_.tensor(weight_name);
        const TensorView& gamma = model_.tensor(tensor_weight(prefix));
        const TensorView& beta = model_.tensor(tensor_bias(prefix));
        const TensorView& mean = model_.tensor(prefix + ".running_mean");
        const TensorView& variance =
            model_.tensor(prefix + ".running_var");
        if (source.rank != 4 || source.dimensions[0] != gamma.elements ||
            gamma.elements != beta.elements ||
            gamma.elements != mean.elements ||
            gamma.elements != variance.elements) {
            throw std::runtime_error(
                "GPU batch norm tensor shape mismatch");
        }
        const std::size_t output_channels =
            static_cast<std::size_t>(gamma.elements);
        const std::size_t elements_per_channel =
            static_cast<std::size_t>(source.elements) / output_channels;
        std::vector<float> folded(
            static_cast<std::size_t>(source.elements));
        std::vector<float> bias(output_channels);
        for (std::size_t channel = 0;
             channel < output_channels;
             ++channel) {
            const float scale = gamma.data[channel] /
                std::sqrt(variance.data[channel] + 0.001f);
            bias[channel] =
                beta.data[channel] - mean.data[channel] * scale;
            const std::size_t begin = channel * elements_per_channel;
            for (std::size_t index = 0;
                 index < elements_per_channel;
                 ++index) {
                folded[begin + index] =
                    source.data[begin + index] * scale;
            }
        }
        VulkanBuffer folded_buffer = context_.create_device_buffer(
            folded.size() * sizeof(float));
        VulkanBuffer bias_buffer = context_.create_device_buffer(
            bias.size() * sizeof(float));
        context_.upload(
            folded_buffer,
            folded.data(),
            folded.size() * sizeof(float));
        context_.upload(
            bias_buffer,
            bias.data(),
            bias.size() * sizeof(float));
        weights_.emplace(
            weight_name + ".folded", std::move(folded_buffer));
        weights_.emplace(
            weight_name + ".folded_bias", std::move(bias_buffer));
    };
    fold_batch_norm(
        "pretrained.layer1.0.weight", "pretrained.layer1.1");
    for (std::string_view name_view : model_.tensor_names()) {
        const std::string name(name_view);
        constexpr std::string_view pointwise = ".conv_pw.weight";
        constexpr std::string_view depthwise = ".conv_dw.weight";
        constexpr std::string_view projection = ".conv_pwl.weight";
        std::string prefix;
        std::string batch_norm;
        if (name_view.size() > pointwise.size() &&
            name_view.substr(name_view.size() - pointwise.size()) ==
                pointwise) {
            prefix = name.substr(0, name.size() - pointwise.size());
            batch_norm = model_.contains(prefix + ".conv_pwl.weight")
                ? prefix + ".bn1"
                : prefix + ".bn2";
        } else if (
            name_view.size() > depthwise.size() &&
            name_view.substr(name_view.size() - depthwise.size()) ==
                depthwise) {
            prefix = name.substr(0, name.size() - depthwise.size());
            batch_norm = model_.contains(prefix + ".conv_pwl.weight")
                ? prefix + ".bn2"
                : prefix + ".bn1";
        } else if (
            name_view.size() > projection.size() &&
            name_view.substr(name_view.size() - projection.size()) ==
                projection) {
            prefix = name.substr(0, name.size() - projection.size());
            batch_norm = prefix + ".bn3";
        }
        if (!batch_norm.empty()) {
            fold_batch_norm(name, batch_norm);
        }
    }
}

const VulkanBuffer& VulkanExecutor::weight(
    const std::string& name) const {
    const auto found = weights_.find(name);
    if (found == weights_.end()) {
        throw std::runtime_error(
            "GPU model is missing tensor: " + name);
    }
    return found->second;
}

VulkanExecutor::Tensor VulkanExecutor::conv(
    const Tensor& input,
    const std::string& weight_name,
    const char* bias_name,
    std::uint32_t stride,
    bool same_stride2,
    std::uint32_t groups,
    const char* batch_norm_prefix,
    std::uint32_t activation,
    bool relu_input,
    const VulkanBuffer* residual) {
    const TensorView& shape = model_.tensor(weight_name);
    if (shape.rank != 4 || groups == 0 ||
        input.channels % groups != 0 ||
        shape.dimensions[1] != input.channels / groups) {
        throw std::runtime_error("GPU convolution shape mismatch");
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(shape.dimensions[0]);
    const std::uint32_t kernel_h =
        static_cast<std::uint32_t>(shape.dimensions[2]);
    const std::uint32_t kernel_w =
        static_cast<std::uint32_t>(shape.dimensions[3]);
    const std::uint32_t output_h = same_stride2
        ? (input.height + stride - 1) / stride
        : (input.height + 2 * (kernel_h / 2) - kernel_h) / stride + 1;
    const std::uint32_t output_w = same_stride2
        ? (input.width + stride - 1) / stride
        : (input.width + 2 * (kernel_w / 2) - kernel_w) / stride + 1;
    const std::int32_t padding_top = same_stride2
        ? static_cast<std::int32_t>(
              ((output_h - 1) * stride + kernel_h - input.height) / 2)
        : static_cast<std::int32_t>(kernel_h / 2);
    const std::int32_t padding_left = same_stride2
        ? static_cast<std::int32_t>(
              ((output_w - 1) * stride + kernel_w - input.width) / 2)
        : static_cast<std::int32_t>(kernel_w / 2);
    Tensor output{
        output_channels,
        output_h,
        output_w,
        context_.create_device_buffer(
            elements(output_channels, output_h, output_w) *
            sizeof(float))};
    const VulkanBuffer* convolution_weight = &weight(weight_name);
    const VulkanBuffer* convolution_bias =
        bias_name ? &weight(bias_name) : &zero_bias_;
    bool convolution_has_bias = bias_name != nullptr;
    if (batch_norm_prefix != nullptr) {
        convolution_weight = &weight(weight_name + ".folded");
        convolution_bias = &weight(weight_name + ".folded_bias");
        convolution_has_bias = true;
    }
    operators_.conv(
        output.buffer,
        input.buffer,
        *convolution_weight,
        *convolution_bias,
        input.width,
        input.height,
        input.channels,
        output.width,
        output.height,
        output.channels,
        kernel_h,
        kernel_w,
        stride,
        padding_top,
        padding_left,
        groups,
        convolution_has_bias,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        activation,
        relu_input,
        residual);
    return output;
}

VulkanExecutor::Tensor VulkanExecutor::batch_norm_activation(
    const Tensor& input,
    const std::string& prefix,
    std::uint32_t activation_kind) {
    Tensor output{
        input.channels,
        input.height,
        input.width,
        context_.create_device_buffer(
            elements(input.channels, input.height, input.width) *
            sizeof(float))};
    operators_.batch_norm_activation(
        output.buffer,
        input.buffer,
        weight(tensor_weight(prefix)),
        weight(tensor_bias(prefix)),
        weight(prefix + ".running_mean"),
        weight(prefix + ".running_var"),
        checked_count(elements(
            input.channels, input.height, input.width)),
        input.width * input.height,
        activation_kind);
    return output;
}

VulkanExecutor::Tensor VulkanExecutor::activation(
    const Tensor& input,
    std::uint32_t kind) {
    Tensor output{
        input.channels,
        input.height,
        input.width,
        context_.create_device_buffer(
            elements(input.channels, input.height, input.width) *
            sizeof(float))};
    operators_.activation(
        output.buffer,
        input.buffer,
        checked_count(elements(
            input.channels, input.height, input.width)),
        kind);
    return output;
}

VulkanExecutor::Tensor VulkanExecutor::add(
    const Tensor& left,
    const Tensor& right) {
    if (left.channels != right.channels ||
        left.height != right.height ||
        left.width != right.width) {
        throw std::runtime_error("GPU residual shape mismatch");
    }
    Tensor output{
        left.channels,
        left.height,
        left.width,
        context_.create_device_buffer(
            elements(left.channels, left.height, left.width) *
            sizeof(float))};
    operators_.add(
        output.buffer,
        left.buffer,
        right.buffer,
        checked_count(elements(
            left.channels, left.height, left.width)));
    return output;
}

VulkanExecutor::Tensor VulkanExecutor::resize(
    const Tensor& input,
    std::uint32_t width,
    std::uint32_t height,
    bool align_corners) {
    Tensor output{
        input.channels,
        height,
        width,
        context_.create_device_buffer(
            elements(input.channels, height, width) * sizeof(float))};
    operators_.resize(
        output.buffer,
        input.buffer,
        input.width,
        input.height,
        output.width,
        output.height,
        output.channels,
        align_corners);
    return output;
}

VulkanExecutor::Tensor VulkanExecutor::depthwise_separable(
    const Tensor& input,
    const std::string& prefix) {
    Tensor value = conv(
        input, tensor_weight(prefix + ".conv_dw"),
        nullptr, 1, false, input.channels,
        (prefix + ".bn1").c_str(), 2);
    value = conv(
        value, tensor_weight(prefix + ".conv_pw"),
        nullptr, 1, false, 1,
        (prefix + ".bn2").c_str(), 0);
    return value;
}

VulkanExecutor::Tensor VulkanExecutor::inverted(
    const Tensor& input,
    const std::string& prefix,
    std::uint32_t stride,
    bool residual) {
    Tensor value = conv(
        input, tensor_weight(prefix + ".conv_pw"),
        nullptr, 1, false, 1,
        (prefix + ".bn1").c_str(), 2);
    value = conv(
        value, tensor_weight(prefix + ".conv_dw"),
        nullptr, stride, stride == 2, value.channels,
        (prefix + ".bn2").c_str(), 2);
    value = conv(
        value, tensor_weight(prefix + ".conv_pwl"),
        nullptr, 1, false, 1,
        (prefix + ".bn3").c_str(), 0, false,
        residual ? &input.buffer : nullptr);
    return value;
}

VulkanExecutor::Tensor VulkanExecutor::residual_unit(
    const Tensor& input,
    const std::string& prefix) {
    const std::string conv1 = prefix + ".conv1";
    Tensor value = conv(
        input, tensor_weight(conv1),
        tensor_bias(conv1).c_str(), 1, false, 1, nullptr, 1, true);
    const std::string conv2 = prefix + ".conv2";
    value = conv(
        value, tensor_weight(conv2),
        tensor_bias(conv2).c_str(), 1, false);
    return add(value, input);
}

VulkanExecutor::Tensor VulkanExecutor::fusion(
    Tensor path,
    const Tensor* skip,
    const std::string& prefix) {
    if (skip != nullptr) {
        Tensor residual = residual_unit(
            *skip, prefix + ".resConfUnit1");
        path = add(path, residual);
    }
    path = residual_unit(path, prefix + ".resConfUnit2");
    path = resize(path, path.width * 2, path.height * 2, true);
    const std::string output = prefix + ".out_conv";
    return conv(
        path,
        tensor_weight(output),
        tensor_bias(output).c_str(),
        1,
        false);
}

void VulkanExecutor::infer(
    const float* normalized_rgb_chw,
    std::uint32_t width,
    std::uint32_t height,
    float* depth,
    std::uint64_t depth_elements) {
    if (normalized_rgb_chw == nullptr || depth == nullptr ||
        width == 0 || height == 0 ||
        width % 32 != 0 || height % 32 != 0 ||
        depth_elements < std::uint64_t(width) * height) {
        throw std::invalid_argument("invalid Vulkan MiDaS tensor shape");
    }
    Tensor input{
        3,
        height,
        width,
        context_.create_device_buffer(
            elements(3, height, width) * sizeof(float))};
    context_.upload(
        input.buffer,
        normalized_rgb_chw,
        static_cast<std::size_t>(
            elements(3, height, width) * sizeof(float)));
    Tensor output;
    context_.batch([&] {
        Tensor value = conv(
            input, "pretrained.layer1.0.weight",
            nullptr, 2, true, 1, "pretrained.layer1.1", 2);
        value = depthwise_separable(
            value, "pretrained.layer1.3.0");
        for (std::uint32_t block = 0; block < 3; ++block) {
            value = inverted(
                value,
                "pretrained.layer1.4." + std::to_string(block),
                block == 0 ? 2u : 1u,
                block != 0);
        }
        Tensor layer1 = std::move(value);

        value = inverted(
            layer1, "pretrained.layer2.0.0", 2, false);
        for (std::uint32_t block = 1; block < 3; ++block) {
            value = inverted(
                value,
                "pretrained.layer2.0." + std::to_string(block),
                1,
                true);
        }
        Tensor layer2 = std::move(value);

        value = inverted(
            layer2, "pretrained.layer3.0.0", 2, false);
        for (std::uint32_t block = 1; block < 5; ++block) {
            value = inverted(
                value,
                "pretrained.layer3.0." + std::to_string(block),
                1,
                true);
        }
        for (std::uint32_t block = 0; block < 5; ++block) {
            value = inverted(
                value,
                "pretrained.layer3.1." + std::to_string(block),
                1,
                block != 0);
        }
        Tensor layer3 = std::move(value);

        value = inverted(
            layer3, "pretrained.layer4.0.0", 2, false);
        for (std::uint32_t block = 1; block < 6; ++block) {
            value = inverted(
                value,
                "pretrained.layer4.0." + std::to_string(block),
                1,
                true);
        }
        value = inverted(
            value, "pretrained.layer4.1.0", 1, false);
        Tensor layer4 = std::move(value);

        Tensor layer1_rn = conv(
            layer1, "scratch.layer1_rn.weight",
            nullptr, 1, false);
        Tensor layer2_rn = conv(
            layer2, "scratch.layer2_rn.weight",
            nullptr, 1, false);
        Tensor layer3_rn = conv(
            layer3, "scratch.layer3_rn.weight",
            nullptr, 1, false);
        Tensor layer4_rn = conv(
            layer4, "scratch.layer4_rn.weight",
            nullptr, 1, false);

        Tensor path = fusion(
            std::move(layer4_rn), nullptr, "scratch.refinenet4");
        path = fusion(
            std::move(path), &layer3_rn, "scratch.refinenet3");
        path = fusion(
            std::move(path), &layer2_rn, "scratch.refinenet2");
        path = fusion(
            std::move(path), &layer1_rn, "scratch.refinenet1");

        path = conv(
            path, "scratch.output_conv.0.weight",
            "scratch.output_conv.0.bias", 1, false);
        path = resize(path, width, height, false);
        path = conv(
            path, "scratch.output_conv.2.weight",
            "scratch.output_conv.2.bias", 1, false);
        path = activation(path, 1);
        path = conv(
            path, "scratch.output_conv.4.weight",
            "scratch.output_conv.4.bias", 1, false);
        output = activation(path, 1);
    });
    context_.download(
        output.buffer,
        depth,
        static_cast<std::size_t>(
            std::uint64_t(width) * height * sizeof(float)));
}

}  // namespace midas_native

#include "cpu_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace midas_native {
namespace {

std::uint64_t elements(
    std::uint32_t channels,
    std::uint32_t height,
    std::uint32_t width) {
    return std::uint64_t(channels) * height * width;
}

std::string weight(const std::string& prefix) {
    return prefix + ".weight";
}

std::string bias(const std::string& prefix) {
    return prefix + ".bias";
}

}  // namespace

CpuExecutor::CpuExecutor(const std::string& model_path)
    : model_(model_path, MIDAS_MODEL_V21_SMALL_256) {}

CpuExecutor::Tensor CpuExecutor::conv(
    const Tensor& input,
    const char* weight_name,
    const char* bias_name,
    std::uint32_t stride,
    bool same_stride2,
    std::uint32_t groups) const {
    const TensorView& weights = model_.tensor(weight_name);
    if (weights.rank != 4 || stride == 0 || groups == 0) {
        throw std::runtime_error("invalid convolution");
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weights.dimensions[0]);
    const std::uint32_t input_per_group =
        static_cast<std::uint32_t>(weights.dimensions[1]);
    const std::uint32_t kernel_h =
        static_cast<std::uint32_t>(weights.dimensions[2]);
    const std::uint32_t kernel_w =
        static_cast<std::uint32_t>(weights.dimensions[3]);
    if (input.channels % groups != 0 ||
        output_channels % groups != 0 ||
        input.channels / groups != input_per_group) {
        throw std::runtime_error("convolution shape mismatch");
    }

    const std::uint32_t output_h =
        same_stride2 ? (input.height + stride - 1) / stride :
        (input.height + 2 * (kernel_h / 2) - kernel_h) / stride + 1;
    const std::uint32_t output_w =
        same_stride2 ? (input.width + stride - 1) / stride :
        (input.width + 2 * (kernel_w / 2) - kernel_w) / stride + 1;
    const std::int32_t pad_top = same_stride2
        ? static_cast<std::int32_t>(
              ((output_h - 1) * stride + kernel_h - input.height) / 2)
        : static_cast<std::int32_t>(kernel_h / 2);
    const std::int32_t pad_left = same_stride2
        ? static_cast<std::int32_t>(
              ((output_w - 1) * stride + kernel_w - input.width) / 2)
        : static_cast<std::int32_t>(kernel_w / 2);

    Tensor output{
        output_channels,
        output_h,
        output_w,
        std::vector<float>(
            static_cast<std::size_t>(
                elements(output_channels, output_h, output_w)))};
    const float* biases = nullptr;
    if (bias_name != nullptr) {
        const TensorView& values = model_.tensor(bias_name);
        if (values.rank != 1 ||
            values.elements != output_channels) {
            throw std::runtime_error("convolution bias shape mismatch");
        }
        biases = values.data;
    }

    const std::uint32_t output_per_group = output_channels / groups;
    const std::uint64_t input_plane =
        std::uint64_t(input.height) * input.width;
    const std::uint64_t output_plane =
        std::uint64_t(output_h) * output_w;
    for (std::uint32_t oc = 0; oc < output_channels; ++oc) {
        const std::uint32_t group = oc / output_per_group;
        const std::uint32_t input_begin = group * input_per_group;
        for (std::uint32_t oy = 0; oy < output_h; ++oy) {
            for (std::uint32_t ox = 0; ox < output_w; ++ox) {
                float sum = biases ? biases[oc] : 0.0f;
                for (std::uint32_t icg = 0;
                     icg < input_per_group;
                     ++icg) {
                    const std::uint32_t ic = input_begin + icg;
                    for (std::uint32_t ky = 0; ky < kernel_h; ++ky) {
                        const std::int32_t iy =
                            static_cast<std::int32_t>(oy * stride + ky) -
                            pad_top;
                        if (iy < 0 || iy >=
                            static_cast<std::int32_t>(input.height)) {
                            continue;
                        }
                        for (std::uint32_t kx = 0; kx < kernel_w; ++kx) {
                            const std::int32_t ix =
                                static_cast<std::int32_t>(
                                    ox * stride + kx) - pad_left;
                            if (ix < 0 || ix >=
                                static_cast<std::int32_t>(input.width)) {
                                continue;
                            }
                            const std::uint64_t input_index =
                                std::uint64_t(ic) * input_plane +
                                std::uint64_t(iy) * input.width +
                                static_cast<std::uint32_t>(ix);
                            const std::uint64_t weight_index =
                                ((std::uint64_t(oc) * input_per_group +
                                  icg) * kernel_h + ky) * kernel_w + kx;
                            sum += input.values[
                                static_cast<std::size_t>(input_index)] *
                                weights.data[weight_index];
                        }
                    }
                }
                output.values[
                    static_cast<std::size_t>(
                        std::uint64_t(oc) * output_plane +
                        std::uint64_t(oy) * output_w + ox)] = sum;
            }
        }
    }
    return output;
}

void CpuExecutor::batch_norm(
    Tensor& tensor,
    const std::string& prefix) const {
    const TensorView& gamma = model_.tensor(weight(prefix));
    const TensorView& beta = model_.tensor(bias(prefix));
    const TensorView& mean = model_.tensor(prefix + ".running_mean");
    const TensorView& variance = model_.tensor(prefix + ".running_var");
    if (gamma.elements != tensor.channels ||
        beta.elements != tensor.channels ||
        mean.elements != tensor.channels ||
        variance.elements != tensor.channels) {
        throw std::runtime_error("batch normalization shape mismatch");
    }
    const std::uint64_t plane =
        std::uint64_t(tensor.height) * tensor.width;
    for (std::uint32_t channel = 0; channel < tensor.channels; ++channel) {
        const float scale =
            gamma.data[channel] /
            std::sqrt(variance.data[channel] + 0.001f);
        const float offset = beta.data[channel] - mean.data[channel] * scale;
        float* values = tensor.values.data() +
            static_cast<std::size_t>(std::uint64_t(channel) * plane);
        for (std::uint64_t index = 0; index < plane; ++index) {
            values[index] = values[index] * scale + offset;
        }
    }
}

void CpuExecutor::relu(Tensor& tensor) {
    for (float& value : tensor.values) value = std::max(value, 0.0f);
}

void CpuExecutor::relu6(Tensor& tensor) {
    for (float& value : tensor.values) {
        value = std::min(std::max(value, 0.0f), 6.0f);
    }
}

void CpuExecutor::add_in_place(
    Tensor& destination,
    const Tensor& source) {
    if (destination.channels != source.channels ||
        destination.height != source.height ||
        destination.width != source.width) {
        throw std::runtime_error("residual shape mismatch");
    }
    for (std::size_t index = 0;
         index < destination.values.size();
         ++index) {
        destination.values[index] += source.values[index];
    }
}

CpuExecutor::Tensor CpuExecutor::resize(
    const Tensor& input,
    std::uint32_t width,
    std::uint32_t height,
    bool align_corners) {
    Tensor output{
        input.channels,
        height,
        width,
        std::vector<float>(
            static_cast<std::size_t>(
                elements(input.channels, height, width)))};
    const std::uint64_t input_plane =
        std::uint64_t(input.height) * input.width;
    const std::uint64_t output_plane = std::uint64_t(height) * width;
    for (std::uint32_t y = 0; y < height; ++y) {
        const float source_y = align_corners && height > 1
            ? float(y) * float(input.height - 1) / float(height - 1)
            : (float(y) + 0.5f) * float(input.height) / float(height) -
                0.5f;
        const float clamped_y =
            std::min(std::max(source_y, 0.0f), float(input.height - 1));
        const std::uint32_t y0 =
            static_cast<std::uint32_t>(std::floor(clamped_y));
        const std::uint32_t y1 = std::min(y0 + 1, input.height - 1);
        const float fy = clamped_y - float(y0);
        for (std::uint32_t x = 0; x < width; ++x) {
            const float source_x = align_corners && width > 1
                ? float(x) * float(input.width - 1) / float(width - 1)
                : (float(x) + 0.5f) * float(input.width) / float(width) -
                    0.5f;
            const float clamped_x =
                std::min(std::max(source_x, 0.0f), float(input.width - 1));
            const std::uint32_t x0 =
                static_cast<std::uint32_t>(std::floor(clamped_x));
            const std::uint32_t x1 = std::min(x0 + 1, input.width - 1);
            const float fx = clamped_x - float(x0);
            for (std::uint32_t c = 0; c < input.channels; ++c) {
                const float* plane = input.values.data() +
                    static_cast<std::size_t>(
                        std::uint64_t(c) * input_plane);
                const float top =
                    plane[std::uint64_t(y0) * input.width + x0] *
                        (1.0f - fx) +
                    plane[std::uint64_t(y0) * input.width + x1] * fx;
                const float bottom =
                    plane[std::uint64_t(y1) * input.width + x0] *
                        (1.0f - fx) +
                    plane[std::uint64_t(y1) * input.width + x1] * fx;
                output.values[
                    static_cast<std::size_t>(
                        std::uint64_t(c) * output_plane +
                        std::uint64_t(y) * width + x)] =
                    top * (1.0f - fy) + bottom * fy;
            }
        }
    }
    return output;
}

CpuExecutor::Tensor CpuExecutor::depthwise_separable(
    const Tensor& input,
    const std::string& prefix) const {
    Tensor value = conv(
        input,
        weight(prefix + ".conv_dw").c_str(),
        nullptr,
        1,
        false,
        input.channels);
    batch_norm(value, prefix + ".bn1");
    relu6(value);
    value = conv(
        value,
        weight(prefix + ".conv_pw").c_str(),
        nullptr,
        1,
        false);
    batch_norm(value, prefix + ".bn2");
    return value;
}

CpuExecutor::Tensor CpuExecutor::inverted(
    const Tensor& input,
    const std::string& prefix,
    std::uint32_t stride,
    bool residual) const {
    Tensor value = conv(
        input,
        weight(prefix + ".conv_pw").c_str(),
        nullptr,
        1,
        false);
    batch_norm(value, prefix + ".bn1");
    relu6(value);
    value = conv(
        value,
        weight(prefix + ".conv_dw").c_str(),
        nullptr,
        stride,
        stride == 2,
        value.channels);
    batch_norm(value, prefix + ".bn2");
    relu6(value);
    value = conv(
        value,
        weight(prefix + ".conv_pwl").c_str(),
        nullptr,
        1,
        false);
    batch_norm(value, prefix + ".bn3");
    if (residual) add_in_place(value, input);
    return value;
}

CpuExecutor::Tensor CpuExecutor::residual_unit(
    const Tensor& input,
    const std::string& prefix) const {
    Tensor value = input;
    relu(value);
    value = conv(
        value,
        weight(prefix + ".conv1").c_str(),
        bias(prefix + ".conv1").c_str(),
        1,
        false);
    relu(value);
    value = conv(
        value,
        weight(prefix + ".conv2").c_str(),
        bias(prefix + ".conv2").c_str(),
        1,
        false);
    add_in_place(value, input);
    return value;
}

CpuExecutor::Tensor CpuExecutor::fusion(
    const Tensor& path,
    const Tensor* skip,
    const std::string& prefix,
    bool expand) const {
    Tensor value = path;
    if (skip != nullptr) {
        Tensor residual = residual_unit(
            *skip, prefix + ".resConfUnit1");
        add_in_place(value, residual);
    }
    value = residual_unit(value, prefix + ".resConfUnit2");
    value = resize(value, value.width * 2, value.height * 2, true);
    value = conv(
        value,
        weight(prefix + ".out_conv").c_str(),
        bias(prefix + ".out_conv").c_str(),
        1,
        false);
    if (expand && value.channels * 2 != path.channels) {
        throw std::runtime_error("fusion expansion mismatch");
    }
    return value;
}

void CpuExecutor::infer(
    const float* normalized_rgb_chw,
    std::uint32_t width,
    std::uint32_t height,
    float* depth,
    std::uint64_t depth_elements) {
    if (normalized_rgb_chw == nullptr || depth == nullptr ||
        width == 0 || height == 0 ||
        width % 32 != 0 || height % 32 != 0 ||
        depth_elements < std::uint64_t(width) * height) {
        throw std::invalid_argument("invalid MiDaS tensor inference shape");
    }
    Tensor value{
        3,
        height,
        width,
        std::vector<float>(
            normalized_rgb_chw,
            normalized_rgb_chw + std::uint64_t(3) * width * height)};

    value = conv(
        value, "pretrained.layer1.0.weight", nullptr, 2, true);
    batch_norm(value, "pretrained.layer1.1");
    relu6(value);
    value = depthwise_separable(value, "pretrained.layer1.3.0");
    for (std::uint32_t block = 0; block < 3; ++block) {
        value = inverted(
            value,
            "pretrained.layer1.4." + std::to_string(block),
            block == 0 ? 2u : 1u,
            block != 0);
    }
    const Tensor layer1 = value;

    for (std::uint32_t block = 0; block < 3; ++block) {
        value = inverted(
            value,
            "pretrained.layer2.0." + std::to_string(block),
            block == 0 ? 2u : 1u,
            block != 0);
    }
    const Tensor layer2 = value;

    for (std::uint32_t block = 0; block < 5; ++block) {
        value = inverted(
            value,
            "pretrained.layer3.0." + std::to_string(block),
            block == 0 ? 2u : 1u,
            block != 0);
    }
    for (std::uint32_t block = 0; block < 5; ++block) {
        value = inverted(
            value,
            "pretrained.layer3.1." + std::to_string(block),
            1,
            block != 0);
    }
    const Tensor layer3 = value;

    for (std::uint32_t block = 0; block < 6; ++block) {
        value = inverted(
            value,
            "pretrained.layer4.0." + std::to_string(block),
            block == 0 ? 2u : 1u,
            block != 0);
    }
    value = inverted(value, "pretrained.layer4.1.0", 1, false);
    const Tensor layer4 = value;

    Tensor layer1_rn = conv(
        layer1, "scratch.layer1_rn.weight", nullptr, 1, false);
    Tensor layer2_rn = conv(
        layer2, "scratch.layer2_rn.weight", nullptr, 1, false);
    Tensor layer3_rn = conv(
        layer3, "scratch.layer3_rn.weight", nullptr, 1, false);
    Tensor layer4_rn = conv(
        layer4, "scratch.layer4_rn.weight", nullptr, 1, false);

    Tensor path = fusion(
        layer4_rn, nullptr, "scratch.refinenet4", true);
    path = fusion(
        path, &layer3_rn, "scratch.refinenet3", true);
    path = fusion(
        path, &layer2_rn, "scratch.refinenet2", true);
    path = fusion(
        path, &layer1_rn, "scratch.refinenet1", false);

    path = conv(
        path, "scratch.output_conv.0.weight",
        "scratch.output_conv.0.bias", 1, false);
    path = resize(path, width, height, false);
    path = conv(
        path, "scratch.output_conv.2.weight",
        "scratch.output_conv.2.bias", 1, false);
    relu(path);
    path = conv(
        path, "scratch.output_conv.4.weight",
        "scratch.output_conv.4.bias", 1, false);
    relu(path);
    if (path.channels != 1 ||
        path.width != width || path.height != height) {
        throw std::runtime_error("invalid MiDaS output shape");
    }
    std::memcpy(
        depth,
        path.values.data(),
        static_cast<std::size_t>(
            std::uint64_t(width) * height * sizeof(float)));
}

}  // namespace midas_native

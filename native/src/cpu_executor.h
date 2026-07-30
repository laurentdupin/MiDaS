#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

namespace midas_native {

class CpuExecutor {
public:
    explicit CpuExecutor(const std::string& model_path);

    void infer(
        const float* normalized_rgb_chw,
        std::uint32_t width,
        std::uint32_t height,
        float* depth,
        std::uint64_t depth_elements);

private:
    struct Tensor {
        std::uint32_t channels = 0;
        std::uint32_t height = 0;
        std::uint32_t width = 0;
        std::vector<float> values;
    };

    Tensor conv(
        const Tensor& input,
        const char* weight_name,
        const char* bias_name,
        std::uint32_t stride,
        bool same_stride2,
        std::uint32_t groups = 1) const;
    void batch_norm(Tensor& tensor, const std::string& prefix) const;
    static void relu(Tensor& tensor);
    static void relu6(Tensor& tensor);
    static void add_in_place(Tensor& destination, const Tensor& source);
    static Tensor resize(
        const Tensor& input,
        std::uint32_t width,
        std::uint32_t height,
        bool align_corners);
    Tensor inverted(
        const Tensor& input,
        const std::string& prefix,
        std::uint32_t stride,
        bool residual) const;
    Tensor depthwise_separable(
        const Tensor& input,
        const std::string& prefix) const;
    Tensor residual_unit(
        const Tensor& input,
        const std::string& prefix) const;
    Tensor fusion(
        const Tensor& path,
        const Tensor* skip,
        const std::string& prefix,
        bool expand) const;

    ModelFile model_;
};

}  // namespace midas_native

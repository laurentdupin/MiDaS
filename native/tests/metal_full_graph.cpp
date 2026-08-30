#include "midas_native.h"

#include <inferbridge/native_harness_precision.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

bool compare_precision(
    const char* model_path, inferbridge::native::Precision precision,
    const std::vector<float>& input, const std::vector<float>& reference,
    float reference_range) {
    midas_context* metal = nullptr;
    {
        const inferbridge::native::ScopedPrecisionRequest scope(precision);
        if (midas_create_metal(
                model_path, MIDAS_MODEL_V21_SMALL_256, &metal) !=
            MIDAS_STATUS_OK) {
            std::fprintf(stderr, "Metal load failed: %s\n", midas_last_error());
            return false;
        }
    }
    std::vector<float> output(reference.size());
    const midas_status status = midas_infer_tensor_f32(
        metal, input.data(), 64, 64, output.data(), output.size());
    midas_destroy(metal);
    if (status != MIDAS_STATUS_OK) {
        std::fprintf(stderr, "Metal inference failed: %s\n", midas_last_error());
        return false;
    }
    double mean_absolute_error = 0.0;
    float maximum_absolute_error = 0.0f;
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (!std::isfinite(output[index])) return false;
        const float error = std::abs(output[index] - reference[index]);
        mean_absolute_error += error;
        maximum_absolute_error = std::max(maximum_absolute_error, error);
    }
    mean_absolute_error /= static_cast<double>(output.size());
    const double normalized_mean_error = mean_absolute_error / reference_range;
    std::printf(
        "precision=%s mean_absolute_error=%.9g maximum_absolute_error=%.9g "
        "normalized_mean_error=%.9g\n",
        precision == inferbridge::native::Precision::fp16 ? "fp16" : "fp32",
        mean_absolute_error, maximum_absolute_error, normalized_mean_error);
    const double tolerance =
        precision == inferbridge::native::Precision::fp16 ? 0.02 : 0.001;
    return normalized_mean_error <= tolerance;
}

}  // namespace

int main() {
    const char* model_path = std::getenv("MIDAS_MODEL");
    if (model_path == nullptr || *model_path == '\0') return 77;
    constexpr std::size_t pixels = 64u * 64u;
    std::vector<float> input(pixels * 3u);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            (static_cast<float>((index * 1103515245u + 12345u) & 1023u) /
                 1023.0f -
             0.5f) *
            2.0f;
    }
    midas_context* cpu = nullptr;
    if (midas_create(model_path, MIDAS_MODEL_V21_SMALL_256, &cpu) !=
        MIDAS_STATUS_OK) {
        std::fprintf(stderr, "CPU load failed: %s\n", midas_last_error());
        return 1;
    }
    std::vector<float> reference(pixels);
    const midas_status cpu_status = midas_infer_tensor_f32(
        cpu, input.data(), 64, 64, reference.data(), reference.size());
    midas_destroy(cpu);
    if (cpu_status != MIDAS_STATUS_OK) {
        std::fprintf(stderr, "CPU inference failed: %s\n", midas_last_error());
        return 1;
    }
    const auto bounds = std::minmax_element(reference.begin(), reference.end());
    const float range = *bounds.second - *bounds.first;
    if (!(range > 0.0f)) return 1;
    if (!compare_precision(
            model_path, inferbridge::native::Precision::fp32,
            input, reference, range))
        return 1;
    if (!compare_precision(
            model_path, inferbridge::native::Precision::fp16,
            input, reference, range))
        return 1;
    return 0;
}

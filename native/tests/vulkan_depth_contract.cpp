#include "gpu_io.h"
#include "vulkan.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    midas_native::VulkanContext context(0u);
    midas_native::GpuIo io(context);
    const std::vector<float> raw{2.0f, 3.0f, 5.0f, 10.0f};
    auto depth = context.create_device_buffer(raw.size() * sizeof(float));
    context.upload(depth, raw.data(), raw.size() * sizeof(float));
    io.normalize_relative(depth, static_cast<std::uint32_t>(raw.size()));
    std::vector<float> normalized(raw.size());
    context.download(depth, normalized.data(), normalized.size() * sizeof(float));
    const std::vector<float> expected{0.0f, 0.125f, 0.375f, 1.0f};
    for (std::size_t index = 0; index < expected.size(); ++index)
        assert(std::abs(normalized[index] - expected[index]) <= 1.0e-6f);
    std::cout << context.device_name() << " normalized relative depth\n";
}

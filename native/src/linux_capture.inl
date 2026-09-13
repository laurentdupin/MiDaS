#if defined(__linux__) && !defined(__ANDROID__) && defined(MIDAS_WITH_VULKAN)
#include "linux_capture.h"
#include <inferbridge/linux_capture_harness.h>
#include <inferbridge/linux_capture_preprocess.h>

ibr_linux_capture_capabilities
midas_linux_capture_capabilities(midas_context *context) {
  if (!context || !context->vulkan_executor)
    return {};
  return context->vulkan_executor->context().linux_capture_capabilities();
}
void midas_infer_linux_capture(
    midas_context *context,
    const inferbridge::linux_capture::LinuxDmaBufImage &source, uint32_t size,
    float *output) {
  if (!context || !context->vulkan_executor)
    throw std::runtime_error("MiDaS Vulkan context is unavailable");
  auto &executor = *context->vulkan_executor;
  auto &vk = executor.context();
  const auto shape =
      midas_native::network_shape(source.width, source.height, size);
  auto input = inferbridge::linux_capture::capture_tensor(
      vk, source, shape.width, shape.height,
      {2, true, {.485f, .456f, .406f, 0}, {.229f, .224f, .225f, 1}});
  auto depth =
      executor.infer_device(std::move(input), shape.width, shape.height);
  std::vector<float> host(uint64_t(shape.width) * shape.height);
  vk.download(depth, host.data(), host.size() * sizeof(float));
  const auto bounds = std::minmax_element(host.begin(), host.end());
  const float minimum = *bounds.first, span = *bounds.second - minimum;
  for (auto &value : host)
    value = span > 0 ? float(static_cast<uint8_t>(std::clamp(
                           (value - minimum) / span * 255.f, 0.f, 255.f))) /
                           255.f
                     : 0.f;
  inferbridge::linux_capture::resize_nearest(host.data(), shape.width,
                                             shape.height, output, shape.width,
                                             shape.height);
}
#endif

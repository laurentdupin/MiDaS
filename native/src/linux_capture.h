#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include "midas_native.h"
#include <inferbridge/linux_capture_vulkan.h>
ibr_linux_capture_capabilities
midas_linux_capture_capabilities(midas_context *);
void midas_infer_linux_capture(
    midas_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint32_t input_size, float *output);
#endif

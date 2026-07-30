#include "midas_native.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    const int device_index = argc > 1 ? atoi(argv[1]) : 0;
    midas_gpu_capabilities capabilities = {0};
    capabilities.struct_size = sizeof(capabilities);
    capabilities.abi_version = MIDAS_ABI_VERSION;

    const midas_status status =
        midas_probe_gpu_capabilities(device_index, &capabilities);
    if (status != MIDAS_STATUS_OK) {
        fprintf(
            stderr,
            "GPU probe failed: %s: %s\n",
            midas_status_string(status),
            midas_last_error());
        return 1;
    }

    printf("device_index=%d\n", device_index);
    printf("device_name=%s\n", capabilities.device_name);
    printf("flags=0x%016" PRIx64 "\n", capabilities.flags);
    printf(
        "vulkan_graph=%u\n",
        (capabilities.flags & MIDAS_GPU_CAP_VULKAN_GRAPH) != 0);
    printf(
        "host_tensor_upload=%u\n",
        (capabilities.flags & MIDAS_GPU_CAP_HOST_TENSOR_UPLOAD) != 0);
    printf(
        "host_depth_readback=%u\n",
        (capabilities.flags & MIDAS_GPU_CAP_HOST_DEPTH_READBACK) != 0);
    printf(
        "maximum_in_flight_jobs=%u\n",
        capabilities.maximum_in_flight_jobs);
    return 0;
}

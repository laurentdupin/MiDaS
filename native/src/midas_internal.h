#pragma once

#include "external_gpu.h"
#include "midas_native.h"

#include <memory>

namespace midas_native {

std::shared_ptr<ExternalJob> submit_external_texture(
    midas_context* context, const ExternalTextureRequest& request);

}  // namespace midas_native

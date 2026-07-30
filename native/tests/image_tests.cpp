#include "image.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main() {
    using midas_native::ImageShape;
    const auto landscape =
        midas_native::network_shape(640, 480, 256);
    assert(landscape.width == 256 && landscape.height == 192);
    const auto portrait =
        midas_native::network_shape(480, 640, 256);
    assert(portrait.width == 192 && portrait.height == 256);
    const auto square =
        midas_native::network_shape(256, 256, 256);
    assert(square.width == 256 && square.height == 256);
    const auto hd =
        midas_native::network_shape(1920, 1080, 256);
    assert(hd.width == 256 && hd.height == 128);

    const std::uint8_t bgr[] = {0, 0, 255};
    std::vector<float> chw;
    midas_native::preprocess_bgr8(
        bgr, 1, 1, 3, {1, 1}, chw);
    assert(chw.size() == 3);
    assert(
        std::abs(
            chw[0] -
            static_cast<float>((1.0 - 0.485) / 0.229)) < 1e-6f);
    assert(
        std::abs(
            chw[1] -
            static_cast<float>((0.0 - 0.456) / 0.224)) < 1e-6f);
    assert(
        std::abs(
            chw[2] -
            static_cast<float>((0.0 - 0.406) / 0.225)) < 1e-6f);

    const float source[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float identity[4]{};
    midas_native::resize_depth_bicubic(
        source, {2, 2}, identity, {2, 2});
    for (int index = 0; index < 4; ++index) {
        assert(std::abs(identity[index] - source[index]) < 1e-6f);
    }
    return 0;
}

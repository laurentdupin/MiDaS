#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(binding = 0) uniform sampler2D source_texture;
layout(std430, binding = 1) writeonly buffer Destination { float values[]; } destination_data;
layout(push_constant) uniform Parameters {
    uint source_width;
    uint source_height;
    uint destination_width;
    uint destination_height;
} parameters;

float cubic1(float value) {
    const float a = -0.75;
    return ((a + 2.0) * value - (a + 3.0)) * value * value + 1.0;
}
float cubic2(float value) {
    const float a = -0.75;
    return ((a * value - 5.0 * a) * value + 8.0 * a) * value - 4.0 * a;
}
float coefficient(int tap, float fraction) {
    if (tap == 0) return cubic2(fraction + 1.0);
    if (tap == 1) return cubic1(fraction);
    if (tap == 2) return cubic1(1.0 - fraction);
    return cubic2(2.0 - fraction);
}

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    if (x >= parameters.destination_width || y >= parameters.destination_height) return;
    const float sx = (float(x) + 0.5) * float(parameters.source_width) /
        float(parameters.destination_width) - 0.5;
    const float sy = (float(y) + 0.5) * float(parameters.source_height) /
        float(parameters.destination_height) - 0.5;
    const int bx = int(floor(sx));
    const int by = int(floor(sy));
    const float fx = sx - float(bx);
    const float fy = sy - float(by);
    const uint plane = parameters.destination_width * parameters.destination_height;
    const uint index = y * parameters.destination_width + x;
    const float means[3] = float[3](0.485, 0.456, 0.406);
    const float deviations[3] = float[3](0.229, 0.224, 0.225);
    for (uint channel = 0; channel < 3; ++channel) {
        float resized = 0.0;
        for (int ty = 0; ty < 4; ++ty) {
            const int iy = clamp(by - 1 + ty, 0, int(parameters.source_height) - 1);
            float row = 0.0;
            for (int tx = 0; tx < 4; ++tx) {
                const int ix = clamp(bx - 1 + tx, 0, int(parameters.source_width) - 1);
                row += texelFetch(source_texture, ivec2(ix, iy), 0)[channel] *
                    coefficient(tx, fx);
            }
            resized += row * coefficient(ty, fy);
        }
        destination_data.values[channel * plane + index] =
            (resized - means[channel]) / deviations[channel];
    }
}

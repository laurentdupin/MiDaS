#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, r32f) uniform writeonly image2D output_image;
layout(set = 0, binding = 1, std430) readonly buffer Input { float data[]; } input_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
} parameters;

float read_value(uint x, uint y) {
    return input_buffer.data[y * parameters.input_width + x];
}
void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    if (x >= parameters.output_width || y >= parameters.output_height) return;
    const float sx = clamp((float(x) + 0.5) * float(parameters.input_width) /
        float(parameters.output_width) - 0.5, 0.0, float(parameters.input_width - 1));
    const float sy = clamp((float(y) + 0.5) * float(parameters.input_height) /
        float(parameters.output_height) - 0.5, 0.0, float(parameters.input_height - 1));
    const uint x0 = uint(floor(sx));
    const uint y0 = uint(floor(sy));
    const uint x1 = min(x0 + 1, parameters.input_width - 1);
    const uint y1 = min(y0 + 1, parameters.input_height - 1);
    const float fx = sx - float(x0);
    const float fy = sy - float(y0);
    const float top = mix(read_value(x0, y0), read_value(x1, y0), fx);
    const float bottom = mix(read_value(x0, y1), read_value(x1, y1), fx);
    imageStore(output_image, ivec2(x, y), vec4(mix(top, bottom, fy)));
}

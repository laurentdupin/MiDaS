#version 450

layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    float data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint input_width; uint input_height; uint input_channels;
    uint output_width; uint output_height; uint output_channels;
    uint kernel_height; uint kernel_width; uint stride;
    int padding_top; int padding_left; uint groups; uint has_bias;
} parameters;
shared float spatial_tile[720];
shared float kernel_tile[288];

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel_base = gl_GlobalInvocationID.z * 8;
    const bool valid =
        x < parameters.output_width &&
        y < parameters.output_height &&
        channel_base < parameters.output_channels;
    float sums[8] = float[8](0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0);
    const uint lane = gl_LocalInvocationID.y * 16 +
        gl_LocalInvocationID.x;
    const int origin_x = int(gl_WorkGroupID.x * 16) - 1;
    const int origin_y = int(gl_WorkGroupID.y * 8) - 1;
    for (uint input_channel_base = 0;
         input_channel_base < parameters.input_channels;
         input_channel_base += 4) {
        for (uint index = lane; index < 720; index += 128) {
            const uint input_offset = index / 180;
            const uint tile_index = index % 180;
            const uint input_channel =
                input_channel_base + input_offset;
            const int input_x = origin_x + int(tile_index % 18);
            const int input_y = origin_y + int(tile_index / 18);
            spatial_tile[index] =
                input_channel < parameters.input_channels &&
                input_x >= 0 && input_x < int(parameters.input_width) &&
                input_y >= 0 && input_y < int(parameters.input_height)
                ? input_buffer.data[
                      (input_channel * parameters.input_height +
                       uint(input_y)) * parameters.input_width +
                      uint(input_x)]
                : 0.0;
        }
        for (uint index = lane; index < 288; index += 128) {
            const uint input_offset = index / 72;
            const uint kernel_index = index % 72;
            const uint input_channel =
                input_channel_base + input_offset;
            const uint offset = kernel_index / 9;
            const uint channel = channel_base + offset;
            kernel_tile[index] =
                input_channel < parameters.input_channels &&
                channel < parameters.output_channels
                ? weight_buffer.data[
                      (channel * parameters.input_channels + input_channel) *
                      9 + kernel_index % 9]
                : 0.0;
        }
        barrier();
        if (valid) {
            for (uint input_offset = 0;
                 input_offset < 4 &&
                 input_channel_base + input_offset <
                     parameters.input_channels;
                 ++input_offset)
                for (uint ky = 0; ky < 3; ++ky)
                    for (uint kx = 0; kx < 3; ++kx) {
                        const float value = spatial_tile[
                            input_offset * 180 +
                            (gl_LocalInvocationID.y + ky) * 18 +
                            gl_LocalInvocationID.x + kx];
                        const uint kernel_index = ky * 3 + kx;
                        for (uint offset = 0; offset < 8; ++offset)
                            sums[offset] += value * kernel_tile[
                                input_offset * 72 + offset * 9 +
                                kernel_index];
                    }
        }
        barrier();
    }
    if (!valid) return;
    for (uint offset = 0; offset < 8; ++offset) {
        const uint channel = channel_base + offset;
        if (channel < parameters.output_channels)
            output_buffer.data[
                (channel * parameters.output_height + y) *
                parameters.output_width + x] =
                sums[offset] + (parameters.has_bias != 0
                    ? bias_buffer.data[channel] : 0.0);
    }
}

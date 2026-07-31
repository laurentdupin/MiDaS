#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
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

#define K_TILE 32
shared float input_tile[32 * K_TILE];
shared float weight_tile[32 * K_TILE];

void main() {
    const uint spatial_count =
        parameters.input_width * parameters.input_height;
    const uint spatial_base =
        gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x * 4;
    const uint channel_base =
        gl_WorkGroupID.y * 32 + gl_LocalInvocationID.y * 4;
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    float sums[4][4];
    for (uint channel = 0; channel < 4; ++channel)
        for (uint spatial = 0; spatial < 4; ++spatial)
            sums[channel][spatial] = 0.0;

    for (uint input_base = 0;
         input_base < parameters.input_channels;
         input_base += K_TILE) {
        for (uint index = lane; index < 32 * K_TILE; index += 64) {
            const uint spatial_offset = index / K_TILE;
            const uint input_channel = input_base + index % K_TILE;
            const uint spatial = gl_WorkGroupID.x * 32 + spatial_offset;
            input_tile[index] =
                spatial < spatial_count &&
                input_channel < parameters.input_channels
                ? input_buffer.data[input_channel * spatial_count + spatial]
                : 0.0;
        }
        for (uint index = lane; index < 32 * K_TILE; index += 64) {
            const uint channel_offset = index / K_TILE;
            const uint input_channel = input_base + index % K_TILE;
            const uint channel =
                gl_WorkGroupID.y * 32 + channel_offset;
            weight_tile[index] =
                channel < parameters.output_channels &&
                input_channel < parameters.input_channels
                ? weight_buffer.data[
                      channel * parameters.input_channels + input_channel]
                : 0.0;
        }
        barrier();
        const uint count =
            min(K_TILE, parameters.input_channels - input_base);
        for (uint inner = 0; inner < count; ++inner) {
            float inputs[4];
            float weights[4];
            for (uint spatial = 0; spatial < 4; ++spatial)
                inputs[spatial] = input_tile[
                    (gl_LocalInvocationID.x * 4 + spatial) * K_TILE +
                    inner];
            for (uint channel = 0; channel < 4; ++channel)
                weights[channel] = weight_tile[
                    (gl_LocalInvocationID.y * 4 + channel) * K_TILE +
                    inner];
            for (uint channel = 0; channel < 4; ++channel)
                for (uint spatial = 0; spatial < 4; ++spatial)
                    sums[channel][spatial] +=
                        weights[channel] * inputs[spatial];
        }
        barrier();
    }

    for (uint channel_offset = 0; channel_offset < 4; ++channel_offset) {
        const uint channel = channel_base + channel_offset;
        if (channel >= parameters.output_channels) continue;
        const float bias = parameters.has_bias != 0
            ? bias_buffer.data[channel] : 0.0;
        for (uint spatial_offset = 0; spatial_offset < 4; ++spatial_offset) {
            const uint spatial = spatial_base + spatial_offset;
            if (spatial < spatial_count)
                output_buffer.data[channel * spatial_count + spatial] =
                    sums[channel_offset][spatial_offset] + bias;
        }
    }
}

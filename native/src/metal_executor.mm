#include "metal_executor.h"

#include "image.h"
#include "inferbridge/native_harness_metal_texture.h"
#include "model.h"
#include <inferbridge/native_harness_precision.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace midas_native {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}

MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t index = 0; index < tensor.rank; ++index)
        [result addObject:@(tensor.dimensions[index])];
    return result;
}

struct Tensor {
    MPSGraphTensor* value = nil;
    int channels = 0;
    int height = 0;
    int width = 0;
};

class GraphBuilder {
public:
    GraphBuilder(const ModelFile& model, int width, int height, bool fp16)
        : model_(model), width_(width), height_(height), fp16_(fp16),
          graph_([MPSGraph new]) {}

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* input() const { return input_; }
    MPSGraphTensor* output() const { return output_; }

    void build() {
        input_ = [graph_ placeholderWithShape:shape({1, 3, height_, width_})
                                     dataType:MPSDataTypeFloat32 name:@"input"];
        Tensor value{fp16_ ? [graph_ castTensor:input_
                                           toType:MPSDataTypeFloat16 name:nil]
                           : input_, 3, height_, width_};
        value = conv(value, "pretrained.layer1.0.weight", nullptr, 2, true);
        value = batch_norm(value, "pretrained.layer1.1");
        value = relu6(value);
        value = depthwise_separable(value, "pretrained.layer1.3.0");
        for (int block = 0; block < 3; ++block)
            value = inverted(value, "pretrained.layer1.4." +
                std::to_string(block), block == 0 ? 2 : 1, block != 0);
        Tensor layer1 = value;
        for (int block = 0; block < 3; ++block)
            value = inverted(value, "pretrained.layer2.0." +
                std::to_string(block), block == 0 ? 2 : 1, block != 0);
        Tensor layer2 = value;
        for (int block = 0; block < 5; ++block)
            value = inverted(value, "pretrained.layer3.0." +
                std::to_string(block), block == 0 ? 2 : 1, block != 0);
        for (int block = 0; block < 5; ++block)
            value = inverted(value, "pretrained.layer3.1." +
                std::to_string(block), 1, block != 0);
        Tensor layer3 = value;
        for (int block = 0; block < 6; ++block)
            value = inverted(value, "pretrained.layer4.0." +
                std::to_string(block), block == 0 ? 2 : 1, block != 0);
        value = inverted(value, "pretrained.layer4.1.0", 1, false);
        Tensor layer4 = value;

        Tensor layer1_rn = conv(layer1, "scratch.layer1_rn.weight", nullptr, 1, false);
        Tensor layer2_rn = conv(layer2, "scratch.layer2_rn.weight", nullptr, 1, false);
        Tensor layer3_rn = conv(layer3, "scratch.layer3_rn.weight", nullptr, 1, false);
        Tensor layer4_rn = conv(layer4, "scratch.layer4_rn.weight", nullptr, 1, false);
        Tensor path = fusion(layer4_rn, nullptr, "scratch.refinenet4");
        path = fusion(path, &layer3_rn, "scratch.refinenet3");
        path = fusion(path, &layer2_rn, "scratch.refinenet2");
        path = fusion(path, &layer1_rn, "scratch.refinenet1");
        path = conv(path, "scratch.output_conv.0.weight",
                    "scratch.output_conv.0.bias", 1, false);
        path = resize(path, width_, height_, false);
        path = conv(path, "scratch.output_conv.2.weight",
                    "scratch.output_conv.2.bias", 1, false);
        path = relu(path);
        path = conv(path, "scratch.output_conv.4.weight",
                    "scratch.output_conv.4.bias", 1, false);
        path = relu(path);
        output_ = fp16_ ? [graph_ castTensor:path.value
                                      toType:MPSDataTypeFloat32 name:@"depth"]
                        : path.value;
    }

    void build_presentation() {
        build();
        NSArray<NSNumber*>* axes = @[@0, @1, @2, @3];
        MPSGraphTensor* minimum = [graph_
            reductionMinimumWithTensor:output_ axes:axes name:nil];
        MPSGraphTensor* maximum = [graph_
            reductionMaximumWithTensor:output_ axes:axes name:nil];
        MPSGraphTensor* span = [graph_
            subtractionWithPrimaryTensor:maximum
                          secondaryTensor:minimum name:nil];
        MPSGraphTensor* epsilon = [graph_ constantWithScalar:1.0e-12
            dataType:MPSDataTypeFloat32];
        span = [graph_ maximumWithPrimaryTensor:span
                                secondaryTensor:epsilon name:nil];
        output_ = [graph_ divisionWithPrimaryTensor:[graph_
            subtractionWithPrimaryTensor:output_
                          secondaryTensor:minimum name:nil]
                                      secondaryTensor:span name:@"depth_normalized"];
    }

private:
    MPSGraphTensor* constant(const std::string& name) {
        const TensorView& tensor = model_.tensor(name);
        NSData* data = [NSData dataWithBytesNoCopy:
            const_cast<float*>(tensor.data)
            length:tensor.elements * sizeof(float) freeWhenDone:NO];
        MPSGraphTensor* value = [graph_ constantWithData:data
            shape:shape(tensor) dataType:MPSDataTypeFloat32];
        return fp16_ ? [graph_ castTensor:value
                                  toType:MPSDataTypeFloat16 name:nil] : value;
    }

    MPSGraphTensor* channel_constant(const std::string& name) {
        const TensorView& tensor = model_.tensor(name);
        return [graph_ reshapeTensor:constant(name)
            withShape:shape({1, static_cast<NSInteger>(tensor.elements), 1, 1})
            name:nil];
    }

    Tensor conv(const Tensor& input, const std::string& weight_name,
                const char* bias_name, int stride, bool same_stride2,
                int groups = 1) {
        const TensorView& weight = model_.tensor(weight_name);
        const int output_channels = static_cast<int>(weight.dimensions[0]);
        const int kernel_h = static_cast<int>(weight.dimensions[2]);
        const int kernel_w = static_cast<int>(weight.dimensions[3]);
        const int output_h = same_stride2 ? (input.height + stride - 1) / stride
            : (input.height + 2 * (kernel_h / 2) - kernel_h) / stride + 1;
        const int output_w = same_stride2 ? (input.width + stride - 1) / stride
            : (input.width + 2 * (kernel_w / 2) - kernel_w) / stride + 1;
        const int total_h = std::max(0,
            (output_h - 1) * stride + kernel_h - input.height);
        const int total_w = std::max(0,
            (output_w - 1) * stride + kernel_w - input.width);
        const int top = same_stride2 ? total_h / 2 : kernel_h / 2;
        const int left = same_stride2 ? total_w / 2 : kernel_w / 2;
        const int bottom = same_stride2 ? total_h - top : kernel_h / 2;
        const int right = same_stride2 ? total_w - left : kernel_w / 2;
        auto* descriptor = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride
            dilationRateInX:1 dilationRateInY:1 groups:groups
            paddingLeft:left paddingRight:right paddingTop:top paddingBottom:bottom
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:input.value
            weightsTensor:constant(weight_name) descriptor:descriptor name:nil];
        if (bias_name != nullptr) {
            result = [graph_ additionWithPrimaryTensor:result
                secondaryTensor:channel_constant(bias_name) name:nil];
        }
        return {result, output_channels, output_h, output_w};
    }

    Tensor batch_norm(const Tensor& input, const std::string& prefix) {
        MPSGraphTensor* variance = channel_constant(prefix + ".running_var");
        MPSGraphTensor* epsilon = [graph_ constantWithScalar:0.001
            dataType:fp16_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32];
        MPSGraphTensor* scale = [graph_ divisionWithPrimaryTensor:
            channel_constant(prefix + ".weight")
            secondaryTensor:[graph_ squareRootWithTensor:
                [graph_ additionWithPrimaryTensor:variance
                    secondaryTensor:epsilon name:nil] name:nil] name:nil];
        MPSGraphTensor* offset = [graph_ subtractionWithPrimaryTensor:
            channel_constant(prefix + ".bias")
            secondaryTensor:[graph_ multiplicationWithPrimaryTensor:
                channel_constant(prefix + ".running_mean")
                secondaryTensor:scale name:nil] name:nil];
        MPSGraphTensor* value = [graph_ additionWithPrimaryTensor:
            [graph_ multiplicationWithPrimaryTensor:input.value
                secondaryTensor:scale name:nil]
            secondaryTensor:offset name:nil];
        return {value, input.channels, input.height, input.width};
    }

    Tensor relu(const Tensor& input) {
        return {[graph_ reLUWithTensor:input.value name:nil], input.channels,
                input.height, input.width};
    }
    Tensor relu6(const Tensor& input) {
        MPSGraphTensor* low = [graph_ constantWithScalar:0.0
            dataType:fp16_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32];
        MPSGraphTensor* high = [graph_ constantWithScalar:6.0
            dataType:fp16_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32];
        return {[graph_ clampWithTensor:input.value minValueTensor:low
            maxValueTensor:high name:nil], input.channels, input.height, input.width};
    }
    Tensor add(const Tensor& left, const Tensor& right) {
        return {[graph_ additionWithPrimaryTensor:left.value
            secondaryTensor:right.value name:nil], left.channels, left.height, left.width};
    }
    Tensor resize(const Tensor& input, int width, int height, bool align) {
        return {[graph_ resizeTensor:input.value size:shape({height, width})
            mode:MPSGraphResizeBilinear centerResult:align ? NO : YES
            alignCorners:align ? YES : NO
            layout:MPSGraphTensorNamedDataLayoutNCHW name:nil],
            input.channels, height, width};
    }
    Tensor depthwise_separable(const Tensor& input, const std::string& prefix) {
        Tensor value = conv(input, prefix + ".conv_dw.weight", nullptr,
                            1, false, input.channels);
        value = relu6(batch_norm(value, prefix + ".bn1"));
        return batch_norm(conv(value, prefix + ".conv_pw.weight", nullptr,
                               1, false), prefix + ".bn2");
    }
    Tensor inverted(const Tensor& input, const std::string& prefix,
                    int stride, bool residual) {
        Tensor value = relu6(batch_norm(conv(input, prefix + ".conv_pw.weight",
            nullptr, 1, false), prefix + ".bn1"));
        value = relu6(batch_norm(conv(value, prefix + ".conv_dw.weight", nullptr,
            stride, stride == 2, value.channels), prefix + ".bn2"));
        value = batch_norm(conv(value, prefix + ".conv_pwl.weight", nullptr,
            1, false), prefix + ".bn3");
        return residual ? add(value, input) : value;
    }
    Tensor residual_unit(const Tensor& input, const std::string& prefix) {
        Tensor value = conv(relu(input), prefix + ".conv1.weight",
                            (prefix + ".conv1.bias").c_str(), 1, false);
        value = conv(relu(value), prefix + ".conv2.weight",
                     (prefix + ".conv2.bias").c_str(), 1, false);
        return add(value, input);
    }
    Tensor fusion(const Tensor& path, const Tensor* skip,
                  const std::string& prefix) {
        Tensor value = path;
        if (skip) value = add(value, residual_unit(*skip,
            prefix + ".resConfUnit1"));
        value = residual_unit(value, prefix + ".resConfUnit2");
        value = resize(value, value.width * 2, value.height * 2, true);
        return conv(value, prefix + ".out_conv.weight",
                    (prefix + ".out_conv.bias").c_str(), 1, false);
    }

    const ModelFile& model_;
    int width_;
    int height_;
    bool fp16_;
    MPSGraph* graph_;
    MPSGraphTensor* input_ = nil;
    MPSGraphTensor* output_ = nil;
};

struct Plan {
    MPSGraph* graph = nil;
    MPSGraphTensor* input = nil;
    MPSGraphExecutable* executable = nil;
};

class MetalExternalJob final : public ExternalJob {
public:
    explicit MetalExternalJob(
        std::shared_ptr<inferbridge::native_harness::metal::Submission> value)
        : submission_(std::move(value)) {}
    ExternalJobState state() const override {
        if (submission_->cancelled()) return ExternalJobState::cancelled;
        return submission_->complete() ? ExternalJobState::complete :
            ExternalJobState::running;
    }
    void cancel() override { submission_->cancel(); }
private:
    std::shared_ptr<inferbridge::native_harness::metal::Submission> submission_;
};

}  // namespace

class MetalExecutor::Impl {
public:
    explicit Impl(const std::string& path)
        : model_(path, MIDAS_MODEL_V21_SMALL_256) {
        device_ = MTLCreateSystemDefaultDevice();
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (device_ == nil || queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("Metal is unavailable for MiDaS");
        const auto precision = inferbridge::native::requested_precision();
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("MiDaS Metal does not support INT8");
        texture_pipeline_ = std::make_unique<
            inferbridge::native_harness::metal::TexturePipeline>(device_);
    }

    void infer(const float* input, std::uint32_t width, std::uint32_t height,
               float* depth, std::uint64_t depth_elements) {
        if (!input || !depth || width == 0 || height == 0 ||
            width % 32 || height % 32 || depth_elements < width * height)
            throw std::invalid_argument("invalid Metal MiDaS tensor shape");
        std::lock_guard<std::mutex> lock(mutex_);
        @autoreleasepool {
            Plan& plan = get_plan(static_cast<int>(width), static_cast<int>(height));
            const NSUInteger bytes = width * height * 3u * sizeof(float);
            id<MTLBuffer> buffer = [device_ newBufferWithBytes:input
                length:bytes options:MTLResourceStorageModeShared];
            MPSGraphTensorData* data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:buffer shape:shape({1, 3,
                    static_cast<NSInteger>(height), static_cast<NSInteger>(width)})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* descriptor =
                [MPSGraphExecutableExecutionDescriptor new];
            descriptor.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_ inputsArray:@[data]
                resultsArray:nil executionDescriptor:descriptor];
            if (results.count != 1u)
                throw std::runtime_error("MiDaS Metal graph returned no output");
            [results[0].mpsndarray readBytes:depth strideBytes:nil];
        }
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) {
        const ImageShape network = network_shape(
            static_cast<int>(request.width),
            static_cast<int>(request.height),
            static_cast<int>(request.input_size));
        constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
        constexpr float deviation[3] = {0.229f, 0.224f, 0.225f};
        inferbridge::native_harness::metal::Request texture_request;
        texture_request.input_texture = request.shared_texture_handle;
        texture_request.input_width = request.width;
        texture_request.input_height = request.height;
        texture_request.input_format = request.rgba
            ? inferbridge::native_harness::metal::PixelFormat::rgba8
            : inferbridge::native_harness::metal::PixelFormat::bgra8;
        texture_request.reverse_channels = true;
        texture_request.wait_event = request.wait_fence_handle;
        texture_request.wait_value = request.wait_fence_value;
        texture_request.output_texture = request.output_texture_handle;
        texture_request.output_width = request.output_width;
        texture_request.output_height = request.output_height;
        texture_request.signal_event = request.signal_fence_handle;
        texture_request.signal_value = request.signal_fence_value;
        std::lock_guard<std::mutex> lock(mutex_);
        @autoreleasepool {
            auto prepared = texture_pipeline_->prepare(texture_request,
                network.width, network.height, mean, deviation);
            Plan& plan = get_presentation_plan(network.width, network.height);
            prepared.input_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:prepared.input_buffer
                shape:shape({1, 3, network.height, network.width})
                dataType:MPSDataTypeFloat32];
            prepared.output_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:prepared.output_buffer
                shape:shape({1, 1, network.height, network.width})
                dataType:MPSDataTypeFloat32];
            MPSGraphExecutableExecutionDescriptor* descriptor =
                [MPSGraphExecutableExecutionDescriptor new];
            descriptor.waitUntilCompleted = NO;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runAsyncWithMTLCommandQueue:texture_pipeline_->queue()
                inputsArray:@[prepared.input_data]
                resultsArray:@[prepared.output_data]
                executionDescriptor:descriptor];
            if (results.count != 1u)
                throw std::runtime_error("MiDaS Metal output binding failed");
            return std::make_shared<MetalExternalJob>(
                texture_pipeline_->finish(prepared,
                    network.width, network.height, true));
        }
    }

private:
    Plan& get_plan(int width, int height) {
        const std::uint64_t key = (std::uint64_t(width) << 32u) |
            static_cast<std::uint32_t>(height);
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        GraphBuilder builder(model_, width, height, fp16_);
        builder.build();
        MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, height, width})
            dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* descriptor =
            [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
        descriptor.waitForCompilationCompletion = YES;
        MPSGraphExecutable* executable = [builder.graph()
            compileWithDevice:graph_device_ feeds:@{builder.input(): input_type}
            targetTensors:@[builder.output()] targetOperations:nil
            compilationDescriptor:descriptor];
        if (executable == nil)
            throw std::runtime_error("failed to compile MiDaS Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(key, Plan{builder.graph(), builder.input(), executable})
            .first->second;
    }

    Plan& get_presentation_plan(int width, int height) {
        const std::uint64_t key = (1ull << 63u) |
            (std::uint64_t(width) << 32u) |
            static_cast<std::uint32_t>(height);
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        GraphBuilder builder(model_, width, height, fp16_);
        builder.build_presentation();
        MPSGraphShapedType* input_type = [[MPSGraphShapedType alloc]
            initWithShape:shape({1, 3, height, width})
            dataType:MPSDataTypeFloat32];
        MPSGraphCompilationDescriptor* descriptor =
            [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
        descriptor.waitForCompilationCompletion = YES;
        MPSGraphExecutable* executable = [builder.graph()
            compileWithDevice:graph_device_ feeds:@{builder.input(): input_type}
            targetTensors:@[builder.output()] targetOperations:nil
            compilationDescriptor:descriptor];
        if (executable == nil)
            throw std::runtime_error(
                "failed to compile MiDaS Metal presentation graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(
            key, Plan{builder.graph(), builder.input(), executable})
            .first->second;
    }

    ModelFile model_;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<std::uint64_t, Plan> plans_;
    std::mutex mutex_;
    std::unique_ptr<inferbridge::native_harness::metal::TexturePipeline>
        texture_pipeline_;
};

MetalExecutor::MetalExecutor(const std::string& path)
    : implementation_(std::make_unique<Impl>(path)) {}
MetalExecutor::~MetalExecutor() = default;
void MetalExecutor::infer(const float* input, std::uint32_t width,
                          std::uint32_t height, float* depth,
                          std::uint64_t depth_elements) {
    implementation_->infer(input, width, height, depth, depth_elements);
}

std::shared_ptr<ExternalJob> MetalExecutor::submit_texture(
    const ExternalTextureRequest& request) {
    return implementation_->submit_texture(request);
}

}  // namespace midas_native

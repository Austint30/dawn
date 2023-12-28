// Copyright 2017 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <vector>

#include "dawn/samples/SampleUtils.h"

#include "dawn/utils/ComboRenderPipelineDescriptor.h"
#include "dawn/utils/SystemUtils.h"
#include "dawn/utils/WGPUHelpers.h"

static int circleSegments = 2 << 6; // Number of triangles in the circle

static int indexBufferSize = circleSegments * 3;
static int vertexBufferSize = (circleSegments + 1) * 4;

wgpu::Device device;

wgpu::Buffer indexBuffer;
wgpu::Buffer vertexBuffer;

wgpu::Texture texture;
wgpu::Sampler sampler;

wgpu::Queue queue;
wgpu::SwapChain swapchain;
wgpu::TextureView depthStencilView;
wgpu::RenderPipeline pipeline;
wgpu::BindGroup bindGroup;

void initBuffers() {

    // Create a buffer for the position data
    std::vector<float> positionData(vertexBufferSize);

    // Intialize the position of the center of the circle
    positionData[0] = 0.0f;
    positionData[1] = 0.0f;
    positionData[2] = 0.0f;
    positionData[3] = 1.0f;

    // Calculate the positions of the circle vertices
    float angle_step = 2.0f * M_PI / circleSegments;
    for (int i = 1; i < circleSegments + 1; i++)
    {
        float angle = (i-1) * angle_step;
        positionData[4 * i]     = std::cos(angle);
        positionData[4 * i + 1] = std::sin(angle);
        positionData[4 * i + 2] = 0.0f;  // Z coordinate is 0 as we're drawing in 2D
        positionData[4 * i + 3] = 1.0f;  // W coordinate is 1 for normalised space
    }

    vertexBuffer = dawn::utils::CreateBufferFromData(device, positionData.data(),
                                                     sizeof(float)* positionData.size(),
                                                     wgpu::BufferUsage::Vertex);

    std::vector<int> indexData(indexBufferSize);

    for (int i = 0; i < circleSegments; ++i) {
        indexData[i*3]      = 0; // Always reuse center vertex
        indexData[i*3 + 1]  = i + 1;
        indexData[i*3 + 2]  = i == circleSegments - 1 ? 1 : i + 2;
    }

    indexBuffer = dawn::utils::CreateBufferFromData(device, indexData.data(), sizeof(int)*indexData.size(),
                                                    wgpu::BufferUsage::Index);
}

void initTextures() {
    wgpu::TextureDescriptor descriptor;
    descriptor.dimension = wgpu::TextureDimension::e2D;
    descriptor.size.width = 1024;
    descriptor.size.height = 1024;
    descriptor.size.depthOrArrayLayers = 1;
    descriptor.sampleCount = 1;
    descriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    descriptor.mipLevelCount = 1;
    descriptor.usage = wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::TextureBinding;
    texture = device.CreateTexture(&descriptor);

    sampler = device.CreateSampler();

    // Initialize the texture with arbitrary data until we can load images
    std::vector<uint8_t> data(4 * 1024 * 1024, 0);
    // 4* because of RGBA
    for (size_t i = 0; i < data.size(); ++i) {

        uint8_t value = 0;

        switch (i % 4) {
            case 0:
                // Red
                value = i % 253;
                break;
            case 2:
                // Blue
                value = 253 - (i % 253);
                break;
        }

        data[i] = static_cast<uint8_t>(value);
    }

    // AUS: Formats the data into a buffer.
    wgpu::Buffer stagingBuffer = dawn::utils::CreateBufferFromData(
        device, data.data(), static_cast<uint32_t>(data.size()), wgpu::BufferUsage::CopySrc);

    // AUS: Copies the stagingBuffer (passes by value) to the ImageCopyBuffer struct.
    //      Struct also accepts a layout. Which can be used to set the bytes per row.
    wgpu::ImageCopyBuffer imageCopyBuffer =
        dawn::utils::CreateImageCopyBuffer(stagingBuffer, 0, 4 * 1024);

    // https://docs.rs/wgpu/latest/wgpu/type.ImageCopyTexture.html
    // View of a texture which can be used to copy to/from a buffer/texture.
    //  texture: &'a Texture    The texture to be copied to/from.
    //  level: u32              The target mip level of the texture.
    //  origin: Origin3d        The base texel of the texture in the selected mip_level.
    //                          Together with the copy_size argument to copy functions, defines the sub-region of the texture to copy.
    wgpu::ImageCopyTexture imageCopyTexture =
        dawn::utils::CreateImageCopyTexture(texture, 0, {0, 0, 0});

    wgpu::Extent3D copySize = {1024, 1024, 1};

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    // Copy the data from the buffer to the texture
    encoder.CopyBufferToTexture(&imageCopyBuffer, &imageCopyTexture, &copySize);

    wgpu::CommandBuffer copy = encoder.Finish();
    queue.Submit(1, &copy);
}

void init() {
    CreateCppDawnDeviceOptions options;
    options.window.width = 640;
    options.window.height = 640;

    device = CreateCppDawnDevice(&options);

    queue = device.GetQueue();
    swapchain = GetSwapChain();

    initBuffers();
    initTextures();

    wgpu::ShaderModule vsModule = dawn::utils::CreateShaderModule(device, R"(
        @vertex fn main(@location(0) pos : vec4f)
                            -> @builtin(position) vec4f {
            return pos;
        })");

    wgpu::ShaderModule fsModule = dawn::utils::CreateShaderModule(device, R"(
        @group(0) @binding(0) var mySampler: sampler;
        @group(0) @binding(1) var myTexture : texture_2d<f32>;

        @fragment fn main(@builtin(position) FragCoord : vec4f)
                              -> @location(0) vec4f {
            return textureSample(myTexture, mySampler, FragCoord.xy / vec2f(640.0, 480.0));
        })");

    auto bgl = dawn::utils::MakeBindGroupLayout(
        device, {
                    {0, wgpu::ShaderStage::Fragment, wgpu::SamplerBindingType::Filtering},
                    {1, wgpu::ShaderStage::Fragment, wgpu::TextureSampleType::Float},
                });

    wgpu::PipelineLayout pl = dawn::utils::MakeBasicPipelineLayout(device, &bgl);

    depthStencilView = CreateDefaultDepthStencilView(device);

    dawn::utils::ComboRenderPipelineDescriptor descriptor;
    descriptor.layout = dawn::utils::MakeBasicPipelineLayout(device, &bgl);
    descriptor.vertex.module = vsModule;
    descriptor.vertex.bufferCount = 1;
    descriptor.cBuffers[0].arrayStride = 4 * sizeof(float);
    descriptor.cBuffers[0].attributeCount = 1;
    descriptor.cAttributes[0].format = wgpu::VertexFormat::Float32x4;
    descriptor.cFragment.module = fsModule;
    descriptor.cTargets[0].format = GetPreferredSwapChainTextureFormat();
    descriptor.EnableDepthStencil(wgpu::TextureFormat::Depth24PlusStencil8);

    pipeline = device.CreateRenderPipeline(&descriptor);

    wgpu::TextureView view = texture.CreateView();

    bindGroup = dawn::utils::MakeBindGroup(device, bgl, {{0, sampler}, {1, view}});
}

struct {
    uint32_t a;
    float b;
} s;
void frame() {
    s.a = (s.a + 1) % 256;
    s.b += 0.02f;
    if (s.b >= 1.0f) {
        s.b = 0.0f;
    }

    wgpu::TextureView backbufferView = swapchain.GetCurrentTextureView();
    dawn::utils::ComboRenderPassDescriptor renderPass({backbufferView}, depthStencilView);

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    {
        wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass);
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bindGroup);
        pass.SetVertexBuffer(0, vertexBuffer);
        pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32);
        pass.DrawIndexed(indexBufferSize);
        pass.End();
    }

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);
    swapchain.Present();
    DoFlush();
}

int main(int argc, const char* argv[]) {
    if (!InitSample(argc, argv)) {
        return 1;
    }
    init();

    while (!ShouldQuit()) {
        ProcessEvents();
        frame();
        dawn::utils::USleep(16000);
    }
}

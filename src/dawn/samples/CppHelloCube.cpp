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

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "dawn/samples/SampleUtils.h"
#include "dawn/utils/ComboRenderPipelineDescriptor.h"
#include "dawn/utils/SystemUtils.h"
#include "dawn/utils/WGPUHelpers.h"
#include "dawn/utils/Timer.h"


#define CUBE_INDICES 6*2*3
#define CUBE_VERTICES 8

wgpu::Device device;

wgpu::Buffer indexBuffer;
wgpu::Buffer vertexBuffer;
wgpu::Buffer vertexColorBuffer;

wgpu::Queue queue;
wgpu::SwapChain swapchain;
wgpu::TextureView depthStencilView;
wgpu::RenderPipeline pipeline;

struct CubeData {
    glm::vec3 pos;
    wgpu::Buffer mvpBuffer;
    wgpu::BindGroup bindGroup;
};

struct SceneState {
    float viewRotation = 0;
};

static SceneState sceneState;

static double deltaTime;

static std::array<CubeData, 1> cubeData = {{
    {glm::vec3(0.0f, 0.0f, 0.0f) },
}};

void initBuffers() {
    static const std::array<uint32_t, CUBE_INDICES> indexData = {
        3, 1, 0, // Front Triangle 1
        3, 2, 1, // Front Triangle 2,

        7, 4, 5, // Back Triangle 1
        6, 7, 5, // Back Triangle 2,

        0, 3, 7, // Top Triangle 1
        0, 7, 4, // Top Triangle 2

        1, 6, 2, // Bottom Triangle 1
        1, 5, 6, // Bottom Triangle 2

        2, 6, 7, // Right Triangle 1
        2, 7, 3, // Right Triangle 2

        0, 5, 1, // Left Triangle 1
        0, 4, 5, // Left Triangle 2,
    };

    indexBuffer = dawn::utils::CreateBufferFromData(device, indexData.data(), sizeof(indexData),
                                                    wgpu::BufferUsage::Index);

    const std::array<float, CUBE_VERTICES*4> vertexData = {
        // Front vertices
        -0.5f, 0.5f, 0.5f, 1.0f,
        -0.5f, -0.5f, 0.5f, 1.0f,
        0.5f, -0.5f, 0.5f, 1.0f,
        0.5f, 0.5f, 0.5f, 1.0f,

        // Back vertices
        -0.5f, 0.5f, -0.5f, 1.0f,
        -0.5f, -0.5f, -0.5f, 1.0f,
        0.5f, -0.5f, -0.5f, 1.0f,
        0.5f, 0.5f, -0.5f, 1.0f,
    };

    vertexBuffer = dawn::utils::CreateBufferFromData(device, vertexData.data(), sizeof(vertexData),
                                                     wgpu::BufferUsage::Vertex);

    // Vertex colors
    const float vertexColorData[CUBE_VERTICES*4] = {
        0.583f,  0.771f,  0.014f, 1.0f,
        0.609f,  0.115f,  0.436f, 1.0f,
        0.327f,  0.483f,  0.844f, 1.0f,
        0.822f,  0.569f,  0.201f, 1.0f,
        0.435f,  0.602f,  0.223f, 1.0f,
        0.310f,  0.747f,  0.185f, 1.0f,
        0.597f,  0.770f,  0.761f, 1.0f,
        0.559f,  0.436f,  0.730f, 1.0f,
    };

    vertexColorBuffer = dawn::utils::CreateBufferFromData(device, vertexColorData, sizeof(float)*CUBE_VERTICES*4, wgpu::BufferUsage::Storage);

    glm::fmat4 identity = glm::fmat4();

    for (CubeData& data : cubeData){
        data.mvpBuffer = dawn::utils::CreateBufferFromData(device, &identity, sizeof(identity), wgpu::BufferUsage::Uniform);
    }
}

void initObjects(wgpu::BindGroupLayout bgl){
    for (int i = 0; i < cubeData.size(); ++i) {
        cubeData[i].bindGroup = dawn::utils::MakeBindGroup(
            device, bgl,
            {
                {0, vertexColorBuffer},
                {1, cubeData[i].mvpBuffer}
            });
    }
}

void init() {
    CreateCppDawnDeviceOptions options;
    options.window.width = 640;
    options.window.height = 640;

    device = CreateCppDawnDevice(&options);

    queue = device.GetQueue();
    swapchain = GetSwapChain();

    initBuffers();

    wgpu::ShaderModule vsModule = dawn::utils::CreateShaderModule(device, R"(

        @group(0) @binding(0) var<storage> vColors: array<vec4f>;
        @group(0) @binding(1) var<uniform> modelViewProjection: mat4x4f;

        struct VertexOut {
            @location(0) vColor : vec4f,
            @builtin(position) pos : vec4f,
        };

        @vertex fn main(
            @builtin(vertex_index) vertexIndex : u32,
            @location(0) pos : vec4f
            ) -> VertexOut {

            var color = vColors[vertexIndex];

            var output : VertexOut;
            output.vColor = color;
            output.pos = modelViewProjection * pos;
            return output;
        })");

    wgpu::ShaderModule fsModule = dawn::utils::CreateShaderModule(device, R"(
        @fragment fn main(@location(0) vColor : vec4f) -> @location(0) vec4f {
            return vColor;
        })");

    auto bgl = dawn::utils::MakeBindGroupLayout(
        device, {
                    {0, wgpu::ShaderStage::Vertex, wgpu::BufferBindingType::ReadOnlyStorage},
                    {1, wgpu::ShaderStage::Vertex, wgpu::BufferBindingType::Uniform }
                });

    wgpu::PipelineLayout pl = dawn::utils::MakeBasicPipelineLayout(device, &bgl);

    depthStencilView = CreateDefaultDepthStencilView(device);

    const wgpu::DepthStencilState depthStencil {
        .format = wgpu::TextureFormat::Depth24PlusStencil8,
        .depthWriteEnabled = true,
        .depthCompare = wgpu::CompareFunction::Less,
        .stencilFront =
            wgpu::StencilFaceState{
                .compare = wgpu::CompareFunction::Always,
            },
        .stencilBack =
            wgpu::StencilFaceState{
                .compare = wgpu::CompareFunction::Always,
            },
    };

    dawn::utils::ComboRenderPipelineDescriptor descriptor;
    descriptor.layout = dawn::utils::MakeBasicPipelineLayout(device, &bgl);
    descriptor.vertex.module = vsModule;
    descriptor.vertex.bufferCount = 1;
    descriptor.cBuffers[0].arrayStride = 4 * sizeof(float);
    descriptor.cBuffers[0].attributeCount = 1;
    descriptor.cAttributes[0].format = wgpu::VertexFormat::Float32x4;
    descriptor.cFragment.module = fsModule;
    descriptor.cTargets[0].format = GetPreferredSwapChainTextureFormat();
    descriptor.depthStencil = &depthStencil;

    pipeline = device.CreateRenderPipeline(&descriptor);

    initObjects(bgl);
}

glm::mat4 getViewProjection(){
    glm::mat4 view, projection, rotation;

    rotation = glm::rotate(sceneState.viewRotation, glm::vec3(0.0f, 1.0f, 0.0f));
    view = glm::mat4(1.0f);

    // Apply translation first, then rotation. This will make the camera orbit the center of the world.
    view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f)) * rotation;

    projection = glm::perspective(45.0f, 1.0f, 0.1f, 100.0f);

    return projection * view;
}

void drawCube(wgpu::RenderPassEncoder pass, CubeData & _cubeData){

    pass.SetBindGroup(0, _cubeData.bindGroup);

    glm::mat4 model, mvp, viewProjection;

    model = glm::mat4(1.0f);
    model = glm::translate(model, _cubeData.pos);

    viewProjection = getViewProjection();
    mvp = viewProjection * model;

    device.GetQueue().WriteBuffer(_cubeData.mvpBuffer, 0, &mvp, sizeof(mvp));
    pass.DrawIndexed(CUBE_INDICES);
}

void drawScene(wgpu::RenderPassEncoder pass){
    for (int i = 0; i < cubeData.size(); ++i) {
        drawCube(pass, cubeData[i]);
    }
}

void updateScene(){
    // One full rotation in 8 seconds
    sceneState.viewRotation += (float)(M_PIf*deltaTime)/8;
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

    wgpu::RenderPassColorAttachment colorAttachment;
    colorAttachment.view = backbufferView;
    colorAttachment.loadOp = wgpu::LoadOp::Clear;
    colorAttachment.storeOp = wgpu::StoreOp::Store;
    colorAttachment.clearValue = {0.0f, 0.0f, 0.0f, 0.0f}; // black

    wgpu::RenderPassDepthStencilAttachment depthStencilAttachment;
    depthStencilAttachment.view = depthStencilView;
    depthStencilAttachment.depthClearValue = 1.0f;
    depthStencilAttachment.stencilClearValue = 0;
    depthStencilAttachment.depthLoadOp = wgpu::LoadOp::Clear;
    depthStencilAttachment.depthStoreOp = wgpu::StoreOp::Store;
    depthStencilAttachment.stencilLoadOp = wgpu::LoadOp::Clear;
    depthStencilAttachment.stencilStoreOp = wgpu::StoreOp::Store;

    wgpu::RenderPassDescriptor renderPass;
    renderPass.colorAttachmentCount = 1;
    renderPass.colorAttachments = { &colorAttachment };
    renderPass.depthStencilAttachment = &depthStencilAttachment;

    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();

    wgpu::RenderPassEncoder pass = encoder.BeginRenderPass(&renderPass);
    pass.SetPipeline(pipeline);
    pass.SetVertexBuffer(0, vertexBuffer);
    pass.SetIndexBuffer(indexBuffer, wgpu::IndexFormat::Uint32);
    drawScene(pass);
    pass.End();

    wgpu::CommandBuffer commands = encoder.Finish();
    queue.Submit(1, &commands);

    swapchain.Present();
    DoFlush();

    updateScene();
}

int main(int argc, const char* argv[]) {
    if (!InitSample(argc, argv)) {
        return 1;
    }
    init();

    dawn::utils::Timer* timer = dawn::utils::CreateTimer();

    while (!ShouldQuit()) {
        timer->Start();
        ProcessEvents();
        frame();

        // 120 fps
        dawn::utils::USleep(8.33*1000);

        deltaTime = timer->GetElapsedTime();
    }
}

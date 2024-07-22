---
layout: post
title:  "Intro to DirectX 12 Pipeline"
author: Gurwinder
categories: [ Game Development ]
image: assets/images/dx12-pipeline.png
featured: true
hidden: false
---

DirectX 12 organizes graphics rendering into pipelines.

### Components of DirectX 12 Pipeline:

1. **Command Lists and Command Allocators:**
   Command lists contain the rendering commands (draw calls, resource bindings, compute dispatches) that GPU will execute. Command allocators manage the memory for these command lists.

2. **Pipeline State Objects (PSOs):**
   PSOs define the state and configuration of the GPU pipeline. This includes input assembler configuration (vertex buffers, index buffers), shader configuration (vertex, hull, domain, geometry, pixel shaders), rasterizer state, depth-stencil state (depth and stencil testing), blend state (output merging), and other fixed function states.

3. **Root Signatures:**
   Root signatures describe which resources (constant buffers, textures, samplers) are accessible to shaders. They define the interface between shaders and the application.

4. **Descriptor Heaps and Tables:**
   Descriptor heaps and tables manage resources like textures and buffers. Descriptor heaps store resource descriptors, and descriptor tables contain pointers to these descriptors, which shaders use to access resources.

5. **Resource Barriers:**
   DirectX 12 introduces explicit resource barriers to specify transitions between resource states (render target to shader resource) and synchronization between GPU and CPU operations.

### Execution Flow:
Let's break down the steps:
1. **Initialization:**
    Create the device, command queue, swap chain, and other necessary objects.
   - **Create the Device and Command Queue:**
     ```cpp
     ID3D12Device* device;
     D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));

     ID3D12CommandQueue* commandQueue;
     D3D12_COMMAND_QUEUE_DESC queueDesc = {};
     queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
     queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
     device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue));
     ```

   - **Create the Swap Chain:**
     ```cpp
     IDXGISwapChain3* swapChain;
     DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
     swapChainDesc.BufferCount = 2;
     swapChainDesc.BufferDesc.Width = width;
     swapChainDesc.BufferDesc.Height = height;
     swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
     swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
     swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
     swapChainDesc.OutputWindow = hwnd;
     swapChainDesc.SampleDesc.Count = 1;
     swapChainDesc.Windowed = TRUE;
     IDXGIFactory4* factory;
     CreateDXGIFactory1(IID_PPV_ARGS(&factory));
     factory->CreateSwapChain(commandQueue, &swapChainDesc, (IDXGISwapChain**)&swapChain);
     ```

2. **Resource Setup:**
    Create buffers, textures, and views in GPU memory.
   - **Create Vertex Buffer:**
     ```cpp
     struct Vertex {
         float position[3];
         float color[3];
     };
     Vertex triangleVertices[] = {
         { { 0.0f,  0.5f, 0.0f }, { 1.0f, 0.0f, 0.0f } },
         { { 0.5f, -0.5f, 0.0f }, { 0.0f, 1.0f, 0.0f } },
         { {-0.5f, -0.5f, 0.0f }, { 0.0f, 0.0f, 1.0f } }
     };
     ID3D12Resource* vertexBuffer;
     CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
     CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(triangleVertices));
     device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vertexBuffer));
     void* pVertexDataBegin;
     CD3DX12_RANGE readRange(0, 0);
     vertexBuffer->Map(0, &readRange, &pVertexDataBegin);
     memcpy(pVertexDataBegin, triangleVertices, sizeof(triangleVertices));
     vertexBuffer->Unmap(0, nullptr);
     ```

   - **Create Vertex Buffer View:**
     ```cpp
     D3D12_VERTEX_BUFFER_VIEW vertexBufferView = {};
     vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
     vertexBufferView.StrideInBytes = sizeof(Vertex);
     vertexBufferView.SizeInBytes = sizeof(triangleVertices);
     ```

3. **Pipeline Setup:**
    Define and compile shaders (HLSL), create PSOs, define root signatures, and set up descriptor heaps.
   - **Define and Compile Shaders:**
     ```hlsl
     // Vertex Shader (HLSL)
     struct VSInput {
         float3 position : POSITION;
         float3 color : COLOR;
     };
     struct PSInput {
         float4 position : SV_POSITION;
         float3 color : COLOR;
     };
     PSInput VSMain(VSInput input) {
         PSInput output;
         output.position = float4(input.position, 1.0f);
         output.color = input.color;
         return output;
     }

     // Pixel Shader (HLSL)
     struct PSInput {
         float4 position : SV_POSITION;
         float3 color : COLOR;
     };
     float4 PSMain(PSInput input) : SV_TARGET {
         return float4(input.color, 1.0f);
     }
     ```

   - **Create Root Signature:**
     ```cpp
     D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc;
     rootSignatureDesc.NumParameters = 0;
     rootSignatureDesc.pParameters = nullptr;
     rootSignatureDesc.NumStaticSamplers = 0;
     rootSignatureDesc.pStaticSamplers = nullptr;
     rootSignatureDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;

     ID3DBlob* signature;
     ID3DBlob* error;
     D3D12SerializeRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error);
     ID3D12RootSignature* rootSignature;
     device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&rootSignature));
     ```

   - **Create Pipeline State Object (PSO):**
     ```cpp
     D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
     psoDesc.InputLayout = { inputLayout, _countof(inputLayout) };
     psoDesc.pRootSignature = rootSignature;
     psoDesc.VS = { vertexShader->GetBufferPointer(), vertexShader->GetBufferSize() };
     psoDesc.PS = { pixelShader->GetBufferPointer(), pixelShader->GetBufferSize() };
     psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
     psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
     psoDesc.DepthStencilState.DepthEnable = FALSE;
     psoDesc.DepthStencilState.StencilEnable = FALSE;
     psoDesc.SampleMask = UINT_MAX;
     psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
     psoDesc.NumRenderTargets = 1;
     psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
     psoDesc.SampleDesc.Count = 1;
     ID3D12PipelineState* pipelineState;
     device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState));
     ```

4. **Rendering Loop:**
    Begin a command list, set the pipeline state, bind resources (through descriptors), issue draw calls or dispatch compute shaders, and then execute the command list on the GPU.
   - **Begin Command List:**
     ```cpp
     ID3D12CommandAllocator* commandAllocator;
     device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator));
     ID3D12GraphicsCommandList* commandList;
     device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, commandAllocator, pipelineState, IID_PPV_ARGS(&commandList));
     commandList->Close();
     ```

   - **Record Commands:**
     ```cpp
     commandAllocator->Reset();
     commandList->Reset(commandAllocator, pipelineState);
     commandList->SetGraphicsRootSignature(rootSignature);
     commandList->RSSetViewports(1, &viewport);
     commandList->RSSetScissorRects(1, &scissorRect);
     commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
     commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
     const float clearColor[] = { 0.0f, 0.2f, 0.4f, 1.0f };
     commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
     commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
     commandList->IASetVertexBuffers(0, 1, &vertexBufferView);
     commandList->DrawInstanced(3, 1, 0, 0);
     commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(renderTarget, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));
     commandList->Close();
     ```

   - **Execute Command List:**
     ```cpp
     ID3D12CommandList* ppCommandLists[] = { commandList };
     commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
     swapChain->Present(1, 0);
     ```

5. **Resource Management and Synchronization:**
    Use resource barriers to synchronize access to resources between different stages of the pipeline and between GPU and CPU.
   - **Use Fences for Synchronization:**
     ```cpp
     ID3D12Fence* fence;
     device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
     HANDLE fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
     UINT64 fenceValue = 1;

     // Signal and wait for the fence
     commandQueue->Signal(fence, fenceValue);
     if (fence->GetCompletedValue() < fenceValue) {
         fence->SetEventOnCompletion(fenceValue, fenceEvent);
         WaitForSingleObject(fenceEvent, INFINITE);
     }
     fenceValue++;
     ```

6. **Cleanup:**
    Release resources and clean up allocated memory when rendering is complete.
   - **Release Resources:**
     ```cpp
     vertexBuffer->Release();
     pipelineState->Release();
     rootSignature->Release();


     commandList->Release();
     commandAllocator->Release();
     commandQueue->Release();
     swapChain->Release();
     device->Release();
     ```

### Summary:

1. **Initialization:** Set up the device, command queue, and swap chain.
2. **Resource Setup:** Create and upload vertex buffer data.
3. **Pipeline Setup:** Define shaders, create root signature and PSO.
4. **Rendering Loop:** Record and execute command lists for rendering the triangle.
5. **Resource Management:** Use fences for synchronization.
6. **Cleanup:** Release resources after rendering.

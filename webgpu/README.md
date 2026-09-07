# WebGPU Compute Shader Suite for Edge LLM Inference

This module provides native **WGSL (WebGPU Shading Language)** compute shaders enabling high-efficiency Transformer inference directly on client GPUs across Chrome, Safari, Edge, Firefox, and Node.js WebGPU environments.

## Architecture

1. **`attention_compute.wgsl`**:
   - Implements block-tiled attention using WebGPU Workgroup Shared Memory (`var<workgroup>`).
   - Dynamically tracks running row-maximum and exponential sums via **Online Softmax**, bounding workgroup memory consumption to $O(B_r \cdot d + B_c \cdot d)$ instead of $O(N^2)$.
   - Eliminates intermediate attention score buffer allocations in VRAM.

2. **`fused_rmsnorm.wgsl`**:
   - Single-pass Root Mean Square Normalization.
   - Computes thread-local sum of squares, intra-workgroup logarithmic reduction tree via `workgroupBarrier()`, fast `inverseSqrt()` computation, and in-place scaled write-back.

## Integration Example (JavaScript / TypeScript)

```typescript
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// Load WGSL Shader Module
const shaderModule = device.createShaderModule({
  code: await (await fetch('attention_compute.wgsl')).text(),
});

const pipeline = device.createComputePipeline({
  layout: 'auto',
  compute: {
    module: shaderModule,
    entryPoint: 'main',
  },
});
```

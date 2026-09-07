#include <torch/extension.h>
#include <vector>
#include "flash_engine.h"

// =============================================================================
// PyTorch C++ / CUDA Extension Bindings (PyBind11)
// =============================================================================

// Forward Attention Binding
torch::Tensor flash_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal
) {
    TORCH_CHECK(q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "V must be contiguous");

    int B = q.size(0);
    int H = q.size(1);
    int N = q.size(2);
    int d = q.size(3);
    int M = k.size(2);

    auto out = torch::empty_like(q);

    flash_engine::attention::AttentionConfig config(B, H, N, M, d, is_causal);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (q.scalar_type() == at::ScalarType::Float) {
        flash_engine::attention::launch_flash_attention_forward_fp32(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            out.data_ptr<float>(),
            config,
            stream
        );
    } else if (q.scalar_type() == at::ScalarType::Half) {
        flash_engine::attention::launch_flash_attention_forward_fp16(
            reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            config,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported scalar type for FlashAttention: only Float and Half are supported");
    }

    return out;
}

// Fused RMSNorm Binding
torch::Tensor fused_rmsnorm_forward(
    torch::Tensor x,
    torch::Tensor gamma,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "gamma must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");

    int hidden_dim = x.size(-1);
    int total_tokens = x.numel() / hidden_dim;

    auto out = torch::empty_like(x);
    flash_engine::norm::RMSNormConfig config(total_tokens, hidden_dim, eps);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (x.scalar_type() == at::ScalarType::Float) {
        flash_engine::norm::launch_fused_rmsnorm_fp32(
            x.data_ptr<float>(),
            gamma.data_ptr<float>(),
            out.data_ptr<float>(),
            config,
            stream
        );
    } else if (x.scalar_type() == at::ScalarType::Half) {
        flash_engine::norm::launch_fused_rmsnorm_fp16(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(gamma.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            config,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported tensor type for RMSNorm");
    }

    return out;
}

// Fused SwiGLU Binding
torch::Tensor fused_swiglu_forward(
    torch::Tensor gate,
    torch::Tensor up
) {
    TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(gate.sizes() == up.sizes(), "Gate and Up tensors must match shapes");

    auto out = torch::empty_like(gate);
    int num_elements = gate.numel();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (gate.scalar_type() == at::ScalarType::Float) {
        flash_engine::activation::launch_fused_swiglu_fp32(
            gate.data_ptr<float>(),
            up.data_ptr<float>(),
            out.data_ptr<float>(),
            num_elements,
            stream
        );
    } else if (gate.scalar_type() == at::ScalarType::Half) {
        flash_engine::activation::launch_fused_swiglu_fp16(
            reinterpret_cast<const half*>(gate.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(up.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            num_elements,
            stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported tensor type for SwiGLU");
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashKernel-Engine: High Performance CUDA & Tensor Core Optimizations";
    m.def("flash_attention_forward", &flash_attention_forward, "FlashAttention-2 Forward (CUDA)");
    m.def("fused_rmsnorm_forward", &fused_rmsnorm_forward, "Fused RMSNorm (CUDA)");
    m.def("fused_swiglu_forward", &fused_swiglu_forward, "Fused SwiGLU (CUDA)");
}

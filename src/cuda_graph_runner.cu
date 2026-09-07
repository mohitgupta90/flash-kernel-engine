#include "cuda_graph_runner.h"
#include <stdexcept>

namespace flash_engine {
namespace graph {

CUDAGraphRunner::CUDAGraphRunner()
    : graph_(nullptr), instance_(nullptr), instantiated_(false) {}

CUDAGraphRunner::~CUDAGraphRunner() {
    reset();
}

void CUDAGraphRunner::capture(std::function<void(cudaStream_t)> record_func, cudaStream_t stream) {
    reset();

    // Begin stream capture mode
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Execute recorded operations
    try {
        record_func(stream);
    } catch (...) {
        // If an exception occurs, ensure capture is terminated
        cudaStreamEndCapture(stream, &graph_);
        reset();
        throw;
    }

    // End stream capture mode and retrieve graph
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));

    // Instantiate executable graph
    CUDA_CHECK(cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0));
    instantiated_ = true;
}

void CUDAGraphRunner::launch(cudaStream_t stream) {
    if (!instantiated_) {
        throw std::runtime_error("CUDAGraphRunner: Cannot launch uncaptured graph.");
    }
    CUDA_CHECK(cudaGraphLaunch(instance_, stream));
}

void CUDAGraphRunner::reset() {
    if (instance_) {
        cudaGraphExecDestroy(instance_);
        instance_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    instantiated_ = false;
}

} // namespace graph
} // namespace flash_engine

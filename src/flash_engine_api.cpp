#include "flash_engine.h"
#include <iostream>
#include <iomanip>

namespace flash_engine {

void print_device_info(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "========================================================\n";
    std::cout << " FlashKernel-Engine Target GPU Architecture Diagnostics\n";
    std::cout << "========================================================\n";
    std::cout << " Device:                        " << prop.name << "\n";
    std::cout << " Compute Capability:            " << prop.major << "." << prop.minor << "\n";
    std::cout << " Streaming Multiprocessors:     " << prop.multiProcessorCount << "\n";
    std::cout << " Warp Size:                     " << prop.warpSize << "\n";
    std::cout << " Max Threads Per SM:            " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << " Max Threads Per Block:         " << prop.maxThreadsPerBlock << "\n";
    std::cout << " Shared Memory Per Block:       " << (prop.sharedMemPerBlock / 1024.0) << " KB\n";
    std::cout << " Shared Memory Per SM:          " << (prop.sharedMemPerMultiprocessor / 1024.0) << " KB\n";
    std::cout << " Total Global Memory:           " << std::fixed << std::setprecision(2)
              << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB\n";
    std::cout << " Memory Bus Width:              " << prop.memoryBusWidth << " bits\n";
    std::cout << " Peak Memory Clock:             " << (prop.memoryClockRate * 1e-3) << " MHz\n";
    std::cout << "========================================================\n";
}

} // namespace flash_engine

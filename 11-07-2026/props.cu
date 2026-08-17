#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

int getCoresPerSM(int major, int minor) {
    // Defines cores per SM based on architecture generation
    switch (major) {
    case 2: // Fermi
        return (minor == 1) ? 48 : 32;
    case 3: // Kepler
        return 192;
    case 5: // Maxwell
        return 128;
    case 6: // Pascal
        if (minor == 1 || minor == 2)
            return 128;
        if (minor == 0)
            return 64;
        return 128; // Default fallback for Pascal
    case 7:         // Volta (7.0), Turing (7.5)
        return 64;
    case 8: // Ampere (8.0, 8.6, 8.7), Ada Lovelace (8.9)
        if (minor == 0)
            return 64;
        if (minor == 6 || minor == 9)
            return 128;
        return 64; // Default fallback for Ampere variants
    case 9:        // Hopper (9.0), Blackwell (9.5)
        return 128;
    default:
        return 128; // Standard fallback for future architectures
    }
}

int main() {
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No available CUDA-capable devices.\n");
        return 1;
    }

    printf("Detected %d CUDA Capable device(s)\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("---------------- Device %d: \"%s\" ----------------\n", i, prop.name);
        printf("CUDA Capability Major/Minor version:           %d.%d\n", prop.major, prop.minor);
        printf("Total amount of global memory:                 %.2f GB\n",
               (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("Number of Streaming Multiprocessors (SMs):     %d\n", prop.multiProcessorCount);
        printf("Total amount of constant memory:               %zu KB\n", prop.totalConstMem / 1024);
        printf("Total amount of shared memory per block:       %zu KB\n", prop.sharedMemPerBlock / 1024);
        printf("Total shared memory per multiprocessor:        %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
        printf("Total number of registers available per block: %d\n", prop.regsPerBlock);
        printf("Warp size:                                     %d threads\n", prop.warpSize);
        printf("Maximum number of threads per multiprocessor:  %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Maximum number of threads per block:           %d\n", prop.maxThreadsPerBlock);
        printf("Max dimension size of a thread block (x,y,z):  (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max dimension size of a grid size (x,y,z):     (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Memory Clock Rate:                             %.2f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("Memory Bus Width:                              %d-bit\n", prop.memoryBusWidth);
        printf("L2 Cache Size:                                 %d KB\n", prop.l2CacheSize / 1024);
        printf("Concurrent copy and execution:                 %s\n", prop.asyncEngineCount ? "Yes" : "No");
        printf("Integrated GPU sharing Host Memory:            %s\n", prop.integrated ? "Yes" : "No");
        printf("Support Host Page-locked Memory Mapping:       %s\n", prop.canMapHostMemory ? "Yes" : "No");
    }

    return 0;
}

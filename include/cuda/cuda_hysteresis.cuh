#ifndef CUDA_HYSTERESIS_CUH
#define CUDA_HYSTERESIS_CUH

#include <stdio.h>
#include <cuda_runtime.h>

#define NOEDGE 255
#define POSSIBLE_EDGE 128
#define EDGE 0

// CUDA error checking helper function
#define CHECK_CUDA_ERROR(call) { \
    cudaError_t cudaStatus = call; \
    if (cudaStatus != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus)); \
        return cudaStatus; \
    } \
}

__global__ void mark_initial_edges_kernel(short* mag, unsigned char* edge, int rows, int cols, int highthreshold);

__global__ void propagate_edges_kernel(unsigned char* edge, short* mag, int rows, int cols, int lowthreshold, int* changed);

__global__ void cleanup_edges_kernel(unsigned char* edge, int rows, int cols);

__global__ void initialize_edge_map_kernel(unsigned char* nms, unsigned char* edge, int rows, int cols);

__global__ void compute_histogram_kernel(short* mag, unsigned char* edge, int rows, int cols, int* hist);

cudaError_t cuda_apply_hysteresis(short* mag, unsigned char* nms, int rows, int cols, float tlow, float thigh, unsigned char* edge);

#endif
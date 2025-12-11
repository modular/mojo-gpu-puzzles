#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define SIZE 9
typedef float dtype;

__global__ void add_10_blocks(dtype* output, const dtype* a, unsigned int size) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    printf("blockDim.x: %u, blockIdx.x: %u, threadIdx.x: %u, i: %u\n", blockDim.x, blockIdx.x, threadIdx.x, i);
    if (i < size) {
        output[i] = a[i] + 10.0f;
    }
}

int main() {
    // Allocate device memory
    dtype* d_output;
    dtype* d_a;
    cudaMalloc((void**)&d_output, SIZE * sizeof(dtype));
    cudaMalloc((void**)&d_a, SIZE * sizeof(dtype));
    
    // Allocate host memory
    dtype* h_a = (dtype*)malloc(SIZE * sizeof(dtype));
    dtype* h_output = (dtype*)malloc(SIZE * sizeof(dtype));
    dtype* h_expected = (dtype*)malloc(SIZE * sizeof(dtype));
    
    // Initialize input array on host
    for (int i = 0; i < SIZE; i++) {
        h_a[i] = (dtype)i;
    }
    
    // Copy input to device
    cudaMemcpy(d_a, h_a, SIZE * sizeof(dtype), cudaMemcpyHostToDevice);
    
    // Initialize output on device to 0
    cudaMemset(d_output, 0, SIZE * sizeof(dtype));
    
    // Launch kernel with grid_dim=(3,1) and block_dim=(4,1)
    dim3 gridDim(3, 1);
    dim3 blockDim(4, 1);
    add_10_blocks<<<gridDim, blockDim>>>(d_output, d_a, SIZE);
    
    // Synchronize
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_output, d_output, SIZE * sizeof(dtype), cudaMemcpyDeviceToHost);
    
    // Prepare expected values
    for (int i = 0; i < SIZE; i++) {
        h_expected[i] = (dtype)i + 10.0f;
    }
    
    // Verify results
    printf("out: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%.1f ", h_output[i]);
    }
    printf("\n");
    
    printf("expected: ");
    for (int i = 0; i < SIZE; i++) {
        printf("%.1f ", h_expected[i]);
    }
    printf("\n");
    
    for (int i = 0; i < SIZE; i++) {
        assert(h_output[i] == h_expected[i]);
    }
    
    printf("Test passed!\n");
    
    // Cleanup
    cudaFree(d_output);
    cudaFree(d_a);
    free(h_a);
    free(h_output);
    free(h_expected);
    
    return 0;
}

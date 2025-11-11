#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 32

// CUDA error macro
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void matmul(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int row = blockIdx.y * blockDim.y * 4 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float result0 = 0.0;
    float result1 = 0.0;
    float result2 = 0.0;
    float result3 = 0.0;
    __shared__ float as0[BLOCKSIZE][BLOCKSIZE];
    __shared__ float as1[BLOCKSIZE][BLOCKSIZE];
    __shared__ float as2[BLOCKSIZE][BLOCKSIZE];
    __shared__ float as3[BLOCKSIZE][BLOCKSIZE];
    __shared__ float bs[BLOCKSIZE][BLOCKSIZE];
    
    for (int kt = 0; kt < K; kt += BLOCKSIZE) {
        as0[threadIdx.y][threadIdx.x] = (row < M && (kt + threadIdx.x) < K) ? A[row * K + kt + threadIdx.x] : 0.0f;
        as1[threadIdx.y][threadIdx.x] = (row + BLOCKSIZE < M && (kt + threadIdx.x) < K) ? A[(row + BLOCKSIZE) * K + kt + threadIdx.x] : 0.0f;
        as2[threadIdx.y][threadIdx.x] = (row + 2 * BLOCKSIZE < M && (kt + threadIdx.x) < K) ? A[(row + 2 * BLOCKSIZE) * K + kt + threadIdx.x] : 0.0f;
        as3[threadIdx.y][threadIdx.x] = (row + 3 * BLOCKSIZE < M && (kt + threadIdx.x) < K) ? A[(row + 3 * BLOCKSIZE) * K + kt + threadIdx.x] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = ((kt + threadIdx.y) < K && col < N) ? B[(kt + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; ++k) {
            result0 += as0[threadIdx.y][k] * bs[k][threadIdx.x];
            result1 += as1[threadIdx.y][k] * bs[k][threadIdx.x];
            result2 += as2[threadIdx.y][k] * bs[k][threadIdx.x];
            result3 += as3[threadIdx.y][k] * bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = result0;
    if (row + BLOCKSIZE < M && col < N) C[(row + BLOCKSIZE) * N + col] = result1;
    if (row + 2 * BLOCKSIZE < M && col < N) C[(row + 2 * BLOCKSIZE) * N + col] = result2;
    if (row + 3 * BLOCKSIZE < M && col < N) C[(row + 3 * BLOCKSIZE) * N + col] = result3;
}

void cpu_matmul(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("Usage: %s <M> <N> <K>\n", argv[0]);
        return 1;
    }

    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    const float tolerance = 0.001;

    printf("Matrix-multiplication (i2): A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // host memory 
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_C_ref = (float*)malloc(sizeC);

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        printf("Host memory allocation failed!\n");
        return 1;
    }

    srand(137);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;
    memset(h_C_ref, 0, sizeC);
    cpu_matmul(h_A, h_B, h_C_ref, M, N, K);
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE * 4 - 1) / (BLOCKSIZE * 4));
    matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > tolerance) {
            printf("Computation error! index = %d, h_C[%d]=%f, h_C_ref[%d]=%f, diff=%f\n", i, i, h_C[i], i, h_C_ref[i], diff);
            return 1;
        }
    }
    printf("Correctness check passed!\n");
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

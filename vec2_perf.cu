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
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // Each thread handles 2 columns

    float2 result = make_float2(0.0f, 0.0f);
    __shared__ float as[BLOCKSIZE][BLOCKSIZE];
    __shared__ float bs[BLOCKSIZE][BLOCKSIZE * 2];  // Wider to store 2 columns per thread

    for (int kt = 0; kt < K; kt += BLOCKSIZE) {
        // Load tile of A (same as before - one element per thread)
        as[threadIdx.y][threadIdx.x] = (row < M && (kt + threadIdx.x) < K) ? A[row * K + kt + threadIdx.x] : 0.0f;

        // Load tile of B - vectorized load of 2 elements
        int b_row = kt + threadIdx.y;
        int b_col = col;
        if (b_row < K && b_col < N) {
            // Check how many elements we can safely load
            if (b_col + 1 < N) {
                // Load all 2 elements as float2
                float2 b_vec = *reinterpret_cast<const float2*>(&B[b_row * N + b_col]);
                bs[threadIdx.y][threadIdx.x * 2 + 0] = b_vec.x;
                bs[threadIdx.y][threadIdx.x * 2 + 1] = b_vec.y;
            } else {
                // Boundary case - load elements individually
                bs[threadIdx.y][threadIdx.x * 2 + 0] = (b_col + 0 < N) ? B[b_row * N + b_col + 0] : 0.0f;
                bs[threadIdx.y][threadIdx.x * 2 + 1] = (b_col + 1 < N) ? B[b_row * N + b_col + 1] : 0.0f;
            }
        } else {
            bs[threadIdx.y][threadIdx.x * 2 + 0] = 0.0f;
            bs[threadIdx.y][threadIdx.x * 2 + 1] = 0.0f;
        }
        __syncthreads();

        // Compute - each thread computes 2 output elements
        for (int k = 0; k < BLOCKSIZE; ++k) {
            float a_val = as[threadIdx.y][k];
            result.x += a_val * bs[k][threadIdx.x * 2 + 0];
            result.y += a_val * bs[k][threadIdx.x * 2 + 1];
        }
        __syncthreads();
    }

    // Write results - vectorized write when possible
    if (row < M && col < N) {
        if (col + 1 < N) {
            // Write all 2 elements as float2
            *reinterpret_cast<float2*>(&C[row * N + col]) = result;
        } else {
            // Boundary case - write elements individually
            if (col + 0 < N) C[row * N + col + 0] = result.x;
            if (col + 1 < N) C[row * N + col + 1] = result.y;
        }
    }
}

// CPU reference implementation for a single element at index (row, col)
float cpu_matmul_element(const float *A, const float *B, int row, int col, int M, int N, int K)
{
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    return sum;
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
    printf("Matrix-multiplication (Vectorized SMEM Float2): A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);

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

    // Initialize matrices
    srand(137);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;
    memset(h_C_ref, 0, sizeC);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    // Grid dimensions adjusted for vectorization - each thread handles 2 columns
    dim3 grid((N + BLOCKSIZE * 2 - 1) / (BLOCKSIZE * 2), (M + BLOCKSIZE - 1) / BLOCKSIZE);

    // Warmup
    int warmup = 100;
    for (int i = 0; i < warmup; ++i)
        matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed measurement
    int iterations = 1000;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i)
        matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Compute GFLOPS: 2 * M * N * K per iteration
    double total_flops = 2.0 * (double)M * N * K * iterations;
    double gflops = total_flops / (milliseconds * 1e-3) / 1e9;

    printf("Average kernel time over %d iterations: %.3f ms\n", iterations, milliseconds / iterations);
    printf("Performance: %d GFLOPS\n", (int)gflops);

    // Copy result back to host for correctness check
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Correctness check: randomly select 20 elements
    printf("\nPerforming correctness check on 20 random elements...\n");
    const float diff_tolerance = 0.001f;
    int num_checks = 20;
    int total_elements = M * N;

    // Use a different seed for random sampling to avoid conflicts
    srand(42);
    int errors = 0;

    for (int i = 0; i < num_checks; ++i) {
        int idx = rand() % total_elements;
        int row = idx / N;
        int col = idx % N;

        // Compute CPU reference for this specific element
        float cpu_result = cpu_matmul_element(h_A, h_B, row, col, M, N, K);
        float gpu_result = h_C[idx];
        float diff = fabs(gpu_result - cpu_result);

        if (diff > diff_tolerance) {
            printf("  Mismatch at index %d (row=%d, col=%d): GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                   idx, row, col, gpu_result, cpu_result, diff);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Correctness check PASSED! All %d sampled elements match within tolerance(%f).\n", num_checks, diff_tolerance);
    } else {
        printf("Correctness check FAILED! %d out of %d sampled elements had errors.\n", errors, num_checks);
    }
    printf("============================================================================================\n\n");
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

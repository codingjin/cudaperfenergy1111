#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvml.h>

#define BLOCKSIZE 32

// CUDA error macro
#define CUDA_CHECK(x) do { \
    cudaError_t err = x; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// NVML error macro
#define NVML_CHECK(x) do { \
    nvmlReturn_t result = x; \
    if (result != NVML_SUCCESS) { \
        printf("NVML Error: %s at %s:%d\n", nvmlErrorString(result), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void matmul(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int row = blockIdx.y * blockDim.y * 2 + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float result0 = 0.0;
    float result1 = 0.0;
    __shared__ float as0[BLOCKSIZE][BLOCKSIZE];
    __shared__ float as1[BLOCKSIZE][BLOCKSIZE];
    __shared__ float bs[BLOCKSIZE][BLOCKSIZE];

    for (int kt = 0; kt < K; kt += BLOCKSIZE) {
        as0[threadIdx.y][threadIdx.x] = (row < M && (kt + threadIdx.x) < K) ? A[row * K + kt + threadIdx.x] : 0.0f;
        as1[threadIdx.y][threadIdx.x] = (row + BLOCKSIZE < M && (kt + threadIdx.x) < K) ? A[(row + BLOCKSIZE) * K + kt + threadIdx.x] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = ((kt + threadIdx.y) < K && col < N) ? B[(kt + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; ++k) {
            result0 += as0[threadIdx.y][k] * bs[k][threadIdx.x];
            result1 += as1[threadIdx.y][k] * bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) C[row * N + col] = result0;
    if (row + BLOCKSIZE < M && col < N) C[(row + BLOCKSIZE) * N + col] = result1;
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
    const float tolerance = 0.001;

    printf("Matrix-multiplication (i2 Energy Measurement): A(%d x %d) * B(%d x %d) = C(%d x %d)\n", M, K, K, N, M, N);

    // Initialize NVML
    NVML_CHECK(nvmlInit());

    // Get device handle (assuming GPU 0)
    nvmlDevice_t device;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &device));

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
        nvmlShutdown();
        return 1;
    }

    srand(137);
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE * 2 - 1) / (BLOCKSIZE * 2));

    // Warmup
    int warmup = 100;
    for (int i = 0; i < warmup; ++i)
        matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Energy measurement: 100 rounds of 100 iterations each
    int rounds = 100;
    int iterations_per_round = 100;
    double *energy_per_round = (double*)malloc(rounds * sizeof(double));

    if (!energy_per_round) {
        printf("Failed to allocate memory for energy measurements!\n");
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_ref);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        nvmlShutdown();
        return 1;
    }

    printf("\n=== Running %d rounds of %d iterations each ===\n", rounds, iterations_per_round);

    for (int r = 0; r < rounds; ++r) {
        unsigned long long energyBefore, energyAfter;

        // Read initial energy counter
        NVML_CHECK(nvmlDeviceGetTotalEnergyConsumption(device, &energyBefore));

        // Run kernels for this round
        for (int i = 0; i < iterations_per_round; ++i)
            matmul<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Read final energy counter
        NVML_CHECK(nvmlDeviceGetTotalEnergyConsumption(device, &energyAfter));

        // Calculate energy for this round in joules
        unsigned long long energyConsumed = energyAfter - energyBefore;  // in millijoules
        energy_per_round[r] = energyConsumed / 1000.0;  // Convert to joules

        if ((r + 1) % 10 == 0) {
            printf("Completed %d/%d rounds...\n", r + 1, rounds);
        }
    }

    // Calculate statistics
    double sum = 0.0;
    for (int r = 0; r < rounds; ++r) {
        sum += energy_per_round[r];
    }
    double mean = sum / rounds;

    double variance_sum = 0.0;
    for (int r = 0; r < rounds; ++r) {
        double diff = energy_per_round[r] - mean;
        variance_sum += diff * diff;
    }
    double std_dev = sqrt(variance_sum / rounds);

    // Calculate average energy per kernel
    double avg_energy_per_kernel = mean / iterations_per_round;
    double std_dev_per_kernel = std_dev / iterations_per_round;

    printf("\n=== Energy Metrics ===\n");
    printf("Rounds: %d\n", rounds);
    printf("Iterations per round: %d\n", iterations_per_round);
    printf("Total iterations: %d\n", rounds * iterations_per_round);
    printf("\nPer round (100 iterations):\n");
    printf("  Mean energy: %.6f J (%.3f mJ)\n", mean, mean * 1000);
    printf("  Std deviation: %.6f J (%.3f mJ)\n", std_dev, std_dev * 1000);
    printf("\nPer kernel:\n");
    printf("  Mean energy: %.6f J (%.3f mJ)\n", avg_energy_per_kernel, avg_energy_per_kernel * 1000);
    printf("  Std deviation: %.6f J (%.3f mJ)\n", std_dev_per_kernel, std_dev_per_kernel * 1000);

    free(energy_per_round);

    // Copy result back to host for correctness check
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Correctness check: randomly select 20 elements
    printf("\n=== Correctness Check ===\n");
    printf("Performing correctness check on 20 random elements...\n");
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

        if (diff > tolerance) {
            printf("  Mismatch at index %d (row=%d, col=%d): GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                   idx, row, col, gpu_result, cpu_result, diff);
            errors++;
        }
    }

    if (errors == 0) {
        printf("Correctness check PASSED! All %d sampled elements match within tolerance(%.3f).\n", num_checks, tolerance);
    } else {
        printf("Correctness check FAILED! %d out of %d sampled elements had errors.\n", errors, num_checks);
    }
    printf("============================================================================================\n");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Shutdown NVML
    nvmlShutdown();

    return 0;
}

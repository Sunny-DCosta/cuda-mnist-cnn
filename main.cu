#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <mma.h>
#include "mnist_loader.h"

using namespace nvcuda;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Kernel error at %s:%d — %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define BATCH_SIZE   32
#define WMMA_M       16
#define WMMA_N       16
#define WMMA_K       16

#define NUM_FILTERS  16
#define CONV_OUT     26
#define POOL_OUT     13
#define DENSE_IN     (NUM_FILTERS * POOL_OUT * POOL_OUT)  // 2704
#define DENSE_OUT    10
#define DENSE_OUT_PAD 16
#define IN_DIM       28

// ======================= WARP REDUCTION HELPER =======================
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ======================= WEIGHT INIT =======================
__global__ void init_weights_fp32(float* __restrict__ w, int n, float scale, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState s;
        curand_init(seed, idx, 0, &s);
        w[idx] = (curand_uniform(&s) - 0.5f) * 2.0f * scale;
    }
}

__global__ void fp32_to_fp16(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(__ldg(&src[idx]));
}

// ======================= FUSED CONV+RELU+POOL FORWARD =======================
__global__ void batched_fused_conv_relu_pool_fp16(
    const float* __restrict__ images,
    const __half* __restrict__ g_conv_w,
    const float* __restrict__ g_conv_b,
    float* __restrict__ pool_out,
    int* __restrict__ max_indices)
{
    __shared__ __half s_w[NUM_FILTERS * 9 + 1]; // +1 Pad
    __shared__ float  s_b[NUM_FILTERS + 1];     // +1 Pad
    __shared__ __half s_img[IN_DIM * IN_DIM + 3]; // +3 Pad to offset 32-bank alignment

    int n     = blockIdx.x;
    int f     = blockIdx.z;
    int p_col = threadIdx.x;
    int p_row = threadIdx.y;
    int tid   = threadIdx.y * blockDim.x + threadIdx.x;

    if (tid < NUM_FILTERS * 9) s_w[tid] = g_conv_w[tid];
    if (tid < NUM_FILTERS) s_b[tid] = g_conv_b[tid];

    const float* img_base = images + (long long)n * IN_DIM * IN_DIM;
    for (int px = tid; px < IN_DIM * IN_DIM; px += 256)
        s_img[px] = __float2half(__ldg(&img_base[px]));

    __syncthreads();

    if (p_row >= POOL_OUT || p_col >= POOL_OUT || f >= NUM_FILTERS) return;

    float max_val = -1e38f;
    int   max_idx = -1;
    float bias    = s_b[f];
    const __half* w = s_w + f * 9;

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            int c_row = p_row * 2 + i;
            int c_col = p_col * 2 + j;

            __half h_conv_sum = __float2half(0.0f);

            int r0 = (c_row + 0) * IN_DIM;
            int r1 = (c_row + 1) * IN_DIM;
            int r2 = (c_row + 2) * IN_DIM;

            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r0 + c_col + 0], w[0]));
            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r0 + c_col + 1], w[1]));
            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r0 + c_col + 2], w[2]));

            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r1 + c_col + 0], w[3]));
            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r1 + c_col + 1], w[4]));
            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r1 + c_col + 2], w[5]));

            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r2 + c_col + 0], w[6]));
            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r2 + c_col + 1], w[7]));
            h_conv_sum = __hadd(h_conv_sum, __hmul(s_img[r2 + c_col + 2], w[8]));

            float conv_sum = __half2float(h_conv_sum) + bias;
            float relu_val = fmaxf(0.0f, conv_sum);

            if (relu_val > max_val) {
                max_val = relu_val;
                max_idx = f * (CONV_OUT * CONV_OUT) + c_row * CONV_OUT + c_col;
            }
        }
    }

    int out_idx = (long long)n * (NUM_FILTERS * POOL_OUT * POOL_OUT) + f * (POOL_OUT * POOL_OUT) + p_row * POOL_OUT + p_col;
    pool_out[out_idx]    = max_val;
    max_indices[out_idx] = max_idx;
}

// ======================= TENSOR CORE DENSE FORWARD =======================
__global__ void batched_dense_forward_tensor_cores(
    const __half* __restrict__ input_fp16,
    const __half* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ output,
    int dense_in)
{
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int batch_row = blockIdx.y * WMMA_M;
    int out_col   = blockIdx.x * WMMA_N;

    for (int k = 0; k < dense_in; k += WMMA_K) {
        wmma::load_matrix_sync(a_frag, input_fp16 + batch_row * dense_in + k, dense_in);
        wmma::load_matrix_sync(b_frag, weights + out_col * dense_in + k, dense_in);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(output + batch_row * DENSE_OUT_PAD + out_col, c_frag, DENSE_OUT_PAD, wmma::mem_row_major);

    if (threadIdx.x < WMMA_M) {
        int b_idx = batch_row + threadIdx.x;
        for (int c = 0; c < DENSE_OUT; c++) {
            output[b_idx * DENSE_OUT_PAD + (out_col + c)] += __ldg(&biases[out_col + c]);
        }
    }
}

__global__ void cast_pool_to_fp16(const float* __restrict__ src, __half* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2half(__ldg(&src[idx]));
}

// ======================= BATCHED LOG SOFTMAX + CROSS ENTROPY =======================
__global__ void batched_log_softmax_cross_entropy(
    float* __restrict__ logits, const int* __restrict__ labels,
    float* __restrict__ d_out, float* __restrict__ losses,
    int dense_out_pad, int dense_out_real)
{
    int n = blockIdx.x;
    float* logit = logits + n * dense_out_pad;
    float* dout  = d_out  + n * dense_out_pad;

    float max_val = -1e38f;
    for (int i = 0; i < dense_out_real; i++) if (logit[i] > max_val) max_val = logit[i];

    float sum = 0.0f;
    for (int i = 0; i < dense_out_real; i++) sum += expf(logit[i] - max_val);
    float log_sum = logf(sum);
    int target = __ldg(&labels[n]);

    losses[n] = -(logit[target] - max_val - log_sum);
    for (int i = 0; i < dense_out_real; i++) {
        float softmax_i = expf(logit[i] - max_val) / sum;
        dout[i] = softmax_i - (i == target ? 1.0f : 0.0f);
    }
}

// ======================= BATCHED DENSE BACKWARD: WEIGHTS =======================
__global__ void batched_dense_backward_weights(
    const float* __restrict__ d_out, const float* __restrict__ input,
    float* __restrict__ d_weights, float* __restrict__ d_biases,
    int dense_in, int dense_out_pad, int dense_out_real, int batch_size)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    int i = blockIdx.y;

    if (j < dense_in && i < dense_out_real) {
        float grad = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad += __ldg(&d_out[b * dense_out_pad + i]) * __ldg(&input[b * dense_in + j]);
        }
        d_weights[i * dense_in + j] = grad;
    }

    if (j < 32 && i < dense_out_real) {
        float b_val = (j < batch_size) ? __ldg(&d_out[j * dense_out_pad + i]) : 0.0f;
        float b_sum = warp_reduce_sum(b_val);
        if (j == 0) d_biases[i] = b_sum;
    }
}

// ======================= BATCHED DENSE BACKWARD: INPUT =======================
__global__ void batched_dense_backward_input(
    const float* __restrict__ d_out, const float* __restrict__ weights, float* __restrict__ d_in,
    int dense_in, int dense_out_pad, int dense_out_real, int batch_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    if (j >= dense_in || n >= batch_size) return;

    float sum = 0.0f;
    for (int o = 0; o < dense_out_real; o++)
        sum += __ldg(&weights[o * dense_in + j]) * __ldg(&d_out[n * dense_out_pad + o]);

    d_in[(long long)n * dense_in + j] = sum;
}

// ======================= BATCHED POOL+RELU BACKWARD =======================
__global__ void batched_pool_relu_backward(
    const float* __restrict__ d_pool_in, const float* __restrict__ pool_fwd_out,
    const int* __restrict__ max_indices, float* __restrict__ d_conv_out, int total_pool)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pool) return;
    if (__ldg(&pool_fwd_out[idx]) <= 0.0f) return;

    int n         = idx / (NUM_FILTERS * POOL_OUT * POOL_OUT);
    int conv_base = n * (NUM_FILTERS * CONV_OUT * CONV_OUT);
    d_conv_out[conv_base + __ldg(&max_indices[idx])] = __ldg(&d_pool_in[idx]);
}

// ======================= BATCHED CONV BACKWARD =======================
__global__ void batched_conv2d_backward(
    const float* __restrict__ d_conv_out, const float* __restrict__ images,
    float* __restrict__ d_weights, float* __restrict__ d_biases, int batch_size)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= NUM_FILTERS) return;

    float dw[9]    = {0};
    float bias_sum = 0.0f;

    for (int b = 0; b < batch_size; b++) {
        const float* d_out_b = d_conv_out + (long long)b * (NUM_FILTERS * CONV_OUT * CONV_OUT);
        const float* img_b   = images     + (long long)b * (IN_DIM * IN_DIM);

        for (int r = 0; r < CONV_OUT; r++) {
            for (int c = 0; c < CONV_OUT; c++) {
                float dz = __ldg(&d_out_b[f * CONV_OUT * CONV_OUT + r * CONV_OUT + c]);
                if (dz != 0.0f) {
                    bias_sum += dz;
                    #pragma unroll
                    for (int i = 0; i < 3; i++)
                        #pragma unroll
                        for (int j = 0; j < 3; j++)
                            dw[i*3+j] += dz * __ldg(&img_b[(r+i)*IN_DIM + (c+j)]);
                }
            }
        }
    }

    d_biases[f] = bias_sum;
    #pragma unroll
    for (int k = 0; k < 9; k++) d_weights[f * 9 + k] = dw[k];
}

// ======================= SGD =======================
__global__ void sgd_update_and_cast(
    float* __restrict__ master, __half* __restrict__ fp16_weights,
    const float* __restrict__ grads, float lr, int size, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    master[idx] -= lr * (__ldg(&grads[idx]) / (float)batch_size);
    fp16_weights[idx] = __float2half(master[idx]);
}

__global__ void sgd_update_bias(
    float* __restrict__ biases, const float* __restrict__ grads,
    float lr, int size, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    biases[idx] -= lr * (__ldg(&grads[idx]) / (float)batch_size);
}

__global__ void compute_predictions(
    const float* __restrict__ probs, const int* __restrict__ labels,
    int* __restrict__ correct_count, int batch_size, int dense_out_pad, int dense_out_real)
{
    int correct = 0;
    for (int n = 0; n < batch_size; n++) {
        const float* p = probs + n * dense_out_pad;
        int max_p = 0;
        for (int i = 1; i < dense_out_real; i++)
            if (__ldg(&p[i]) > __ldg(&p[max_p])) max_p = i;
        if (max_p == __ldg(&labels[n])) correct++;
    }
    *correct_count = correct;
}

// ======================= MAIN =======================
int main() {
    // ✅ Correct paths (IMPORTANT FIX)
    const char* train_images_path = "../data/train-images-idx3-ubyte";
    const char* train_labels_path = "../data/train-labels-idx1-ubyte";
    const char* test_images_path  = "../data/t10k-images-idx3-ubyte";
    const char* test_labels_path  = "../data/t10k-labels-idx1-ubyte";

    int num_images, num_test;

    // ✅ Load MNIST
    float* h_images = read_mnist_images(train_images_path, &num_images);
    uint8_t* h_labels = read_mnist_labels(train_labels_path, &num_images);

    float* test_images = read_mnist_images(test_images_path, &num_test);
    uint8_t* test_labels = read_mnist_labels(test_labels_path, &num_test);

    // ✅ Error check
    if (!h_images || !h_labels || !test_images || !test_labels) {
        fprintf(stderr, "❌ Failed to load MNIST\n");
        return -1;
    }

    // ✅ Debug print
    printf("✅ MNIST Loaded Successfully!\n");
    printf("Train Samples: %d\n", num_images);
    printf("Test Samples: %d\n", num_test);
    printf("First Train Label: %d\n\n", h_labels[0]);

    const int   train_samples = 5000;
    const int   batch_size    = BATCH_SIZE;
    const int   num_batches   = train_samples / batch_size;
    const int   epochs        = 5;
    const float lr            = 0.01f*BATCH_SIZE;

    printf("CUDA HACKER PATH Batched CNN (WMMA + Shuffles + Bank Pad)\n");
    printf("Batch: %d | Filters: %d | Samples: %d | Batches/epoch: %d\n\n",
           batch_size, NUM_FILTERS, train_samples, num_batches);

    float* d_images;
    CUDA_CHECK(cudaMalloc(&d_images, (long long)num_images * IN_DIM * IN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_images, h_images,
               (long long)num_images * IN_DIM * IN_DIM * sizeof(float), cudaMemcpyHostToDevice));

    int* h_labels_int = (int*)malloc(num_images * sizeof(int));
    for (int i = 0; i < num_images; i++) h_labels_int[i] = (int)h_labels[i];
    int* d_labels;
    CUDA_CHECK(cudaMalloc(&d_labels, num_images * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels_int, num_images * sizeof(int), cudaMemcpyHostToDevice));

    float *d_conv_w_fp32, *d_conv_b, *d_dense_w_fp32, *d_dense_b;
    CUDA_CHECK(cudaMalloc(&d_conv_w_fp32,  NUM_FILTERS * 9          * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_b,       NUM_FILTERS              * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense_w_fp32, DENSE_OUT_PAD * DENSE_IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dense_b,      DENSE_OUT_PAD            * sizeof(float)));

    __half *d_conv_w_fp16, *d_dense_w_fp16;
    CUDA_CHECK(cudaMalloc(&d_conv_w_fp16,  NUM_FILTERS * 9          * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dense_w_fp16, DENSE_OUT_PAD * DENSE_IN * sizeof(__half)));

    float *d_pool_out, *d_dense_out;
    __half *d_pool_out_fp16;
    int   *d_max_indices;
    int total_pool = batch_size * NUM_FILTERS * POOL_OUT * POOL_OUT;

    CUDA_CHECK(cudaMalloc(&d_pool_out,      total_pool * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pool_out_fp16, total_pool * sizeof(__half)));
    CUDA_CHECK(cudaMalloc((void**)&d_max_indices, total_pool * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dense_out, batch_size * DENSE_OUT_PAD * sizeof(float)));

    float *d_d_out, *d_d_dense_w, *d_d_dense_b;
    float *d_d_pool_in, *d_d_conv_out, *d_d_conv_w, *d_d_conv_b;
    CUDA_CHECK(cudaMalloc(&d_d_out,     batch_size * DENSE_OUT_PAD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_dense_w, DENSE_OUT_PAD * DENSE_IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_dense_b, DENSE_OUT_PAD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_pool_in, total_pool * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_conv_out,(long long)batch_size * NUM_FILTERS * CONV_OUT * CONV_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_conv_w,  NUM_FILTERS * 9 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_d_conv_b,  NUM_FILTERS     * sizeof(float)));

    float *d_losses; int *d_correct;
    CUDA_CHECK(cudaMalloc(&d_losses,  batch_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_correct, sizeof(int)));

    float *h_losses_pinned; int *h_correct_pinned;
    CUDA_CHECK(cudaMallocHost(&h_losses_pinned,  batch_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_correct_pinned, sizeof(int)));

    cudaStream_t copy_stream;
    CUDA_CHECK(cudaStreamCreate(&copy_stream));

    // ✅ FIXED: Using the same seed everywhere for strict reproducibility
    unsigned long long SEED = 42ULL;
    float xavier_dense = sqrtf(1.0f / DENSE_IN);
    init_weights_fp32<<<(NUM_FILTERS*9+255)/256,      256>>>(d_conv_w_fp32,  NUM_FILTERS*9,      0.05f,        SEED);
    init_weights_fp32<<<(DENSE_OUT_PAD*DENSE_IN+255)/256, 256>>>(d_dense_w_fp32, DENSE_OUT_PAD*DENSE_IN, xavier_dense, SEED);
    CUDA_CHECK(cudaMemset(d_conv_b,  0, NUM_FILTERS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dense_b, 0, DENSE_OUT_PAD * sizeof(float)));

    fp32_to_fp16<<<(NUM_FILTERS*9+255)/256,      256>>>(d_conv_w_fp32,  d_conv_w_fp16,  NUM_FILTERS*9);
    fp32_to_fp16<<<(DENSE_OUT_PAD*DENSE_IN+255)/256, 256>>>(d_dense_w_fp32, d_dense_w_fp16, DENSE_OUT_PAD*DENSE_IN);
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 conv_fwd_grid(batch_size, 1, NUM_FILTERS);
    dim3 conv_fwd_block(16, 16);
    dim3 wmma_grid(DENSE_OUT_PAD / WMMA_N, batch_size / WMMA_M);
    dim3 wmma_block(32);
    dim3 dense_bwd_w_block(256);
    dim3 dense_bwd_w_grid((DENSE_IN + 255) / 256, DENSE_OUT_PAD);
    dim3 dense_bwd_in_grid((DENSE_IN + 255) / 256, batch_size);

    int pool_bwd_blocks  = (total_pool + 255) / 256;
    int sgd_dense_blocks = (DENSE_OUT_PAD * DENSE_IN + 255) / 256;
    int sgd_conv_blocks  = (NUM_FILTERS * 9 + 31) / 32;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start));

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss    = 0.0f;
        int   total_correct = 0;
        float epoch_lr      = lr * powf(0.95f, (float)epoch);

        for (int b = 0; b < num_batches; b++) {
            long long batch_start = (long long)b * batch_size;
            float* batch_images   = d_images + batch_start * IN_DIM * IN_DIM;
            int* batch_labels   = d_labels + batch_start;

            CUDA_CHECK(cudaMemset(d_d_conv_out, 0, (long long)batch_size * NUM_FILTERS * CONV_OUT * CONV_OUT * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_d_conv_w,  0, NUM_FILTERS * 9 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_d_dense_w, 0, DENSE_OUT_PAD * DENSE_IN * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_d_pool_in, 0, (long long)batch_size * NUM_FILTERS * POOL_OUT * POOL_OUT * sizeof(float)));

            // ---- FORWARD ----
            batched_fused_conv_relu_pool_fp16<<<conv_fwd_grid, conv_fwd_block>>>(
                batch_images, d_conv_w_fp16, d_conv_b, d_pool_out, d_max_indices);

            cast_pool_to_fp16<<<(total_pool+255)/256, 256>>>(d_pool_out, d_pool_out_fp16, total_pool);

            batched_dense_forward_tensor_cores<<<wmma_grid, wmma_block>>>(
                d_pool_out_fp16, d_dense_w_fp16, d_dense_b, d_dense_out, DENSE_IN);

            batched_log_softmax_cross_entropy<<<batch_size, 1>>>(
                d_dense_out, batch_labels, d_d_out, d_losses, DENSE_OUT_PAD, DENSE_OUT);

            CUDA_CHECK(cudaMemcpyAsync(h_losses_pinned, d_losses, batch_size * sizeof(float), cudaMemcpyDeviceToHost, copy_stream));
            compute_predictions<<<1, 1>>>(
                d_dense_out, batch_labels, d_correct, batch_size, DENSE_OUT_PAD, DENSE_OUT);
            CUDA_CHECK(cudaMemcpyAsync(h_correct_pinned, d_correct, sizeof(int), cudaMemcpyDeviceToHost, copy_stream));

            // ---- BACKWARD ----
            batched_dense_backward_weights<<<dense_bwd_w_grid, dense_bwd_w_block>>>(
                d_d_out, d_pool_out, d_d_dense_w, d_d_dense_b, DENSE_IN, DENSE_OUT_PAD, DENSE_OUT, batch_size);

            batched_dense_backward_input<<<dense_bwd_in_grid, 256>>>(
                d_d_out, d_dense_w_fp32, d_d_pool_in, DENSE_IN, DENSE_OUT_PAD, DENSE_OUT, batch_size);

            batched_pool_relu_backward<<<pool_bwd_blocks, 256>>>(
                d_d_pool_in, d_pool_out, d_max_indices, d_d_conv_out, total_pool);

            batched_conv2d_backward<<<sgd_conv_blocks, 32>>>(
                d_d_conv_out, batch_images, d_d_conv_w, d_d_conv_b, batch_size);

            // ---- SGD ----
            sgd_update_and_cast<<<sgd_dense_blocks, 256>>>(
                d_dense_w_fp32, d_dense_w_fp16, d_d_dense_w, epoch_lr, DENSE_OUT_PAD * DENSE_IN, batch_size);
            sgd_update_bias<<<1, 32>>>(
                d_dense_b, d_d_dense_b, epoch_lr, DENSE_OUT_PAD, batch_size);
            sgd_update_and_cast<<<sgd_conv_blocks, 32>>>(
                d_conv_w_fp32, d_conv_w_fp16, d_d_conv_w, epoch_lr, NUM_FILTERS * 9, batch_size);
            sgd_update_bias<<<1, 32>>>(
                d_conv_b, d_d_conv_b, epoch_lr, NUM_FILTERS, batch_size);

            CUDA_CHECK(cudaStreamSynchronize(copy_stream));
            for (int s = 0; s < batch_size; s++) total_loss += h_losses_pinned[s];
            total_correct += *h_correct_pinned;
        }

        printf("Epoch %d | LR: %.5f | Loss: %.4f | Accuracy: %.2f%%\n",
               epoch + 1, epoch_lr, total_loss / train_samples, (total_correct / (float)train_samples) * 100.0f);
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
    printf("Total Execution Time: %.3f seconds\n", ms / 1000.0f);

    // ---- Cleanup ----
    cudaFree(d_images); cudaFree(d_labels); cudaFree(d_conv_w_fp32); cudaFree(d_conv_b);
    cudaFree(d_dense_w_fp32); cudaFree(d_dense_b); cudaFree(d_conv_w_fp16); cudaFree(d_dense_w_fp16);
    cudaFree(d_pool_out); cudaFree(d_pool_out_fp16); cudaFree(d_max_indices); cudaFree(d_dense_out);
    cudaFree(d_d_out); cudaFree(d_d_dense_w); cudaFree(d_d_dense_b); cudaFree(d_d_pool_in);
    cudaFree(d_d_conv_out); cudaFree(d_d_conv_w); cudaFree(d_d_conv_b); cudaFree(d_losses); cudaFree(d_correct);
    cudaFreeHost(h_losses_pinned); cudaFreeHost(h_correct_pinned);
    cudaStreamDestroy(copy_stream);
    free(h_images); free(h_labels); free(h_labels_int);

    free(test_images); free(test_labels);

    return 0;
}
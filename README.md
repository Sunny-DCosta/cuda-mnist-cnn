# CUDA MNIST CNN

A convolutional neural network for MNIST digit classification written entirely in CUDA C from scratch — no deep learning frameworks. The focus is on GPU-level optimizations: fused kernels, Tensor Cores, mixed precision, and shared memory tuning.

---

## Architecture

```
Input (28×28) → Conv2D (16 filters, 3×3) → ReLU → MaxPool (2×2) → Dense (2704→10) → Softmax
```

| Layer       | Output Shape         | Notes                          |
|-------------|----------------------|--------------------------------|
| Input       | 1 × 28 × 28          | Normalized to [0, 1]           |
| Conv2D      | 16 × 26 × 26         | 3×3 filters, no padding        |
| ReLU        | 16 × 26 × 26         | Fused into conv kernel         |
| MaxPool 2×2 | 16 × 13 × 13         | Fused into conv kernel         |
| Dense       | 2704 → 10            | Tensor Cores (WMMA, FP16)      |
| Softmax     | 10                   | Log-softmax + cross-entropy    |

Training: SGD with per-epoch learning rate decay (`lr × 0.95^epoch`).

---

## GPU Optimizations

### Fused Conv + ReLU + MaxPool kernel
Conv, ReLU, and 2×2 max-pooling are merged into a single kernel. This eliminates two intermediate global memory writes and reads that a naive implementation would require.

### Tensor Cores (WMMA)
The dense layer forward pass uses the `nvcuda::wmma` API for FP16 matrix multiplication on Tensor Core hardware. Weights are padded to 16 (`DENSE_OUT_PAD`) to satisfy WMMA tile alignment.

### Mixed precision
Convolutional and dense weights are maintained as FP32 master copies for numerically stable SGD updates, and separately cast to FP16 for forward compute. The cast is fused into the SGD update kernel so there's no extra pass over weight memory.

### Shared memory with bank-conflict padding
The fused conv kernel loads the 3×3 filter bank, biases, and the 28×28 image into shared memory before compute begins. Arrays are padded (`+1` for weights, `+3` for image) to offset 32-bank alignment and avoid shared memory bank conflicts.

### Warp shuffle reductions
Bias gradient accumulation in the dense backward pass uses `__shfl_down_sync` warp reductions instead of atomic operations or shared memory reductions.

### Pinned memory + async D→H transfers
Loss values and per-batch accuracy are transferred on a dedicated `cudaStream_t` using `cudaMallocHost` (pinned) buffers and `cudaMemcpyAsync`, overlapping the host-side accumulation with GPU backward pass execution.

### Read-only cache (`__ldg`)
All read-only global memory accesses in the backward pass use `__ldg` to route through the texture cache.

### Compiler-level
- `#pragma unroll` on the fully unrolled 3×3 convolution inner loops
- `-O3 --use_fast_math` at compile time

---

## Requirements

- NVIDIA GPU with **compute capability ≥ 8.6** (RTX 30xx or later; Tensor Cores required)
- CUDA Toolkit ≥ 11.x
- MNIST dataset files (see below)

Tested on `sm_86` (RTX 3070) and `sm_89` (RTX 4060).

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/cuda-mnist-cnn.git
cd cuda-mnist-cnn
```

### 2. Download MNIST

Download the four binary files from [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist) and place them in a `data/` folder:

```
data/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

### 3. Compile

```bash
nvcc main.cu -o mnist_cnn     \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -O3 --use_fast_math         \
  -lcurand                    \
  -lineinfo
```

### 4. Run

```bash
./mnist_cnn
```

Expected output:

```
✅ MNIST Loaded Successfully!
Train Samples: 5000 | Test Samples: 10000

Batch: 32 | Filters: 16 | Samples: 5000 | Batches/epoch: 156
Epoch 1 | LR: 0.32000 | Loss: 0.7517 | Accuracy: 77.52%
Epoch 2 | LR: 0.30400 | Loss: 0.2165 | Accuracy: 93.66%
Epoch 3 | LR: 0.28880 | Loss: 0.1495 | Accuracy: 95.62%
Epoch 4 | LR: 0.27436 | Loss: 0.1101 | Accuracy: 96.76%
Epoch 5 | LR: 0.26064 | Loss: 0.0828 | Accuracy: 97.60%
Total Execution Time: 2.365 seconds
```

---

## Project Structure

```
cuda-mnist-cnn/
├── main.cu          # All CUDA kernels and training loop
├── mnist_loader.h   # Minimal MNIST binary file reader
├── data/            # MNIST files go here (gitignored)
└── README.md
```

---

## What this is not

This is a learning and experimentation project, not a production CNN framework. There is no test-set evaluation loop, no model checkpointing, and no multi-GPU support. The training uses 5,000 samples by default (configurable via `train_samples` in `main()`).

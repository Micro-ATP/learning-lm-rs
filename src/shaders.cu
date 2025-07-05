#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// 矩阵乘法: C = beta * C + alpha * A @ B^T
__global__ void matmul_kernel(
    const float* a, const float* b, float* c,
    int m, int n, int k, float alpha, float beta
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[col * k + i];
        }
        c[row * n + col] = beta * c[row * n + col] + alpha * sum;
    }
}

// RMS归一化
__global__ void rms_norm_kernel(
    const float* x, const float* w, float* y,
    int batch_size, int hidden_size, float eps
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float sum_squares;
    if (tid == 0) sum_squares = 0.0f;
    __syncthreads();
    
    // 计算平方和
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[batch * hidden_size + i];
        atomicAdd(&sum_squares, val * val);
    }
    __syncthreads();
    
    // 计算RMS
    float rms = sqrtf(sum_squares / hidden_size + eps);
    
    // 应用归一化和权重
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        y[batch * hidden_size + i] = (w[i] * x[batch * hidden_size + i]) / rms;
    }
}

// SwiGLU激活函数
__global__ void swiglu_kernel(
    const float* x, float* y, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        float y_val = y[idx];
        
        // sigmoid(x) = 1 / (1 + exp(-x))
        float sigmoid = 1.0f / (1.0f + expf(-x_val));
        // silu(x) = sigmoid(x) * x
        float silu = sigmoid * x_val;
        // y = silu(x) * y
        y[idx] = silu * y_val;
    }
}

// Softmax
__global__ void softmax_kernel(
    float* x, int seq_len, int batch_size
) {
    int batch = blockIdx.x;
    int tid = threadIdx.x;
    
    __shared__ float max_val;
    __shared__ float sum_exp;
    
    // 找到最大值
    if (tid == 0) {
        max_val = x[batch * seq_len];
        for (int i = 1; i < seq_len; i++) {
            max_val = max(max_val, x[batch * seq_len + i]);
        }
    }
    __syncthreads();
    
    // 计算exp和sum
    if (tid == 0) sum_exp = 0.0f;
    __syncthreads();
    
    for (int i = tid; i < seq_len; i += blockDim.x) {
        float exp_val = expf(x[batch * seq_len + i] - max_val);
        x[batch * seq_len + i] = exp_val;
        atomicAdd(&sum_exp, exp_val);
    }
    __syncthreads();
    
    // 归一化
    for (int i = tid; i < seq_len; i += blockDim.x) {
        x[batch * seq_len + i] /= sum_exp;
    }
}

// 辅助函数：启动矩阵乘法kernel
extern "C" void launch_matmul(
    const float* a, const float* b, float* c,
    int m, int n, int k, float alpha, float beta
) {
    dim3 block_size(16, 16);
    dim3 grid_size((m + 15) / 16, (n + 15) / 16);
    
    matmul_kernel<<<grid_size, block_size>>>(
        a, b, c, m, n, k, alpha, beta
    );
}

// 辅助函数：启动RMS归一化kernel
extern "C" void launch_rms_norm(
    const float* x, const float* w, float* y,
    int batch_size, int hidden_size, float eps
) {
    dim3 block_size(256);
    dim3 grid_size(batch_size);
    
    rms_norm_kernel<<<grid_size, block_size>>>(
        x, w, y, batch_size, hidden_size, eps
    );
}

// 辅助函数：启动SwiGLU kernel
extern "C" void launch_swiglu(
    const float* x, float* y, int size
) {
    dim3 block_size(256);
    dim3 grid_size((size + 255) / 256);
    
    swiglu_kernel<<<grid_size, block_size>>>(x, y, size);
}

// 辅助函数：启动Softmax kernel
extern "C" void launch_softmax(
    float* x, int seq_len, int batch_size
) {
    dim3 block_size(256);
    dim3 grid_size(batch_size);
    
    softmax_kernel<<<grid_size, block_size>>>(x, seq_len, batch_size);
} 
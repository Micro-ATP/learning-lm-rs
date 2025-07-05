#include <metal_stdlib>
using namespace metal;

// 矩阵乘法: C = beta * C + alpha * A @ B^T
kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   device const float* params [[buffer(3)]],
                   uint2 tid [[thread_position_in_grid]]) {
    
    uint m = tid.x;
    uint n = tid.y;
    uint k = 128; // 假设k=128，实际应该从参数获取
    
    float alpha = params[0];
    float beta = params[1];
    
    float sum = 0.0;
    for (uint i = 0; i < k; i++) {
        sum += a[m * k + i] * b[n * k + i];
    }
    
    uint idx = m * 128 + n; // 假设输出矩阵大小为128x128
    c[idx] = beta * c[idx] + alpha * sum;
}

// RMS归一化
kernel void rms_norm(device const float* x [[buffer(0)]],
                     device const float* w [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     device const float* params [[buffer(3)]],
                     uint tid [[thread_position_in_grid]]) {
    
    float eps = params[0];
    uint n = 128; // 假设归一化维度为128
    
    // 计算RMS
    float sum_squares = 0.0;
    for (uint i = 0; i < n; i++) {
        float val = x[tid * n + i];
        sum_squares += val * val;
    }
    float rms = sqrt(sum_squares / float(n) + eps);
    
    // 应用归一化和权重
    for (uint i = 0; i < n; i++) {
        y[tid * n + i] = (w[i] * x[tid * n + i]) / rms;
    }
}

// SwiGLU激活函数
kernel void swiglu(device const float* x [[buffer(0)]],
                   device float* y [[buffer(1)]],
                   uint tid [[thread_position_in_grid]]) {
    
    float x_val = x[tid];
    float y_val = y[tid];
    
    // sigmoid(x) = 1 / (1 + exp(-x))
    float sigmoid = 1.0 / (1.0 + exp(-x_val));
    // silu(x) = sigmoid(x) * x
    float silu = sigmoid * x_val;
    // y = silu(x) * y
    y[tid] = silu * y_val;
}

// Softmax
kernel void softmax(device float* x [[buffer(0)]],
                    uint tid [[thread_position_in_grid]]) {
    
    uint seq_len = 512; // 假设序列长度为512
    uint start_idx = tid * seq_len;
    
    // 找到最大值
    float max_val = x[start_idx];
    for (uint i = 1; i < seq_len; i++) {
        max_val = max(max_val, x[start_idx + i]);
    }
    
    // 计算exp和sum
    float sum = 0.0;
    for (uint i = 0; i < seq_len; i++) {
        float exp_val = exp(x[start_idx + i] - max_val);
        x[start_idx + i] = exp_val;
        sum += exp_val;
    }
    
    // 归一化
    for (uint i = 0; i < seq_len; i++) {
        x[start_idx + i] /= sum;
    }
} 
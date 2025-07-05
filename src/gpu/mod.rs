#[cfg(target_os = "macos")]
pub mod metal_backend;

#[cfg(target_os = "windows")]
pub mod cuda_backend;

use crate::tensor::Tensor;

pub trait GPUBackend {
    fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32);
    fn rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32);
    fn swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>);
    fn softmax(&self, x: &mut Tensor<f32>);
}

pub struct GPUContext {
    #[cfg(target_os = "macos")]
    backend: Box<dyn GPUBackend>,
    
    #[cfg(target_os = "windows")]
    backend: Box<dyn GPUBackend>,
}

impl GPUContext {
    #[cfg(target_os = "macos")]
    pub fn new() -> Result<Self, String> {
        // 尝试创建Metal后端
        if metal_backend::MetalBackend::is_available() {
            let backend = metal_backend::MetalBackend::new()
                .map_err(|e| format!("Failed to create Metal backend: {}", e))?;
            Ok(GPUContext {
                backend: Box::new(backend),
            })
        } else {
            Err("No GPU backend available".to_string())
        }
    }
    
    #[cfg(target_os = "windows")]
    pub fn new() -> Result<Self, String> {
        // 尝试创建CUDA后端
        if cuda_backend::CudaBackend::is_available() {
            let backend = cuda_backend::CudaBackend::new()
                .map_err(|e| format!("Failed to create CUDA backend: {}", e))?;
            Ok(GPUContext {
                backend: Box::new(backend),
            })
        } else {
            Err("No GPU backend available".to_string())
        }
    }
    
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    pub fn new() -> Result<Self, String> {
        Err("GPU acceleration not supported on this platform".to_string())
    }
    
    #[cfg(target_os = "macos")]
    pub fn is_available() -> bool {
        metal_backend::MetalBackend::is_available()
    }
    
    #[cfg(target_os = "windows")]
    pub fn is_available() -> bool {
        cuda_backend::CudaBackend::is_available()
    }
    
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    pub fn is_available() -> bool {
        false
    }
    
    #[cfg(target_os = "macos")]
    pub fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32) {
        self.backend.matmul(a, b, c, alpha, beta);
    }
    
    #[cfg(target_os = "macos")]
    pub fn rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32) {
        self.backend.rms_norm(x, w, y, eps);
    }
    
    #[cfg(target_os = "macos")]
    pub fn swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>) {
        self.backend.swiglu(x, y);
    }
    
    #[cfg(target_os = "macos")]
    pub fn softmax(&self, x: &mut Tensor<f32>) {
        self.backend.softmax(x);
    }
    
    #[cfg(target_os = "windows")]
    pub fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32) {
        self.backend.matmul(a, b, c, alpha, beta);
    }
    
    #[cfg(target_os = "windows")]
    pub fn rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32) {
        self.backend.rms_norm(x, w, y, eps);
    }
    
    #[cfg(target_os = "windows")]
    pub fn swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>) {
        self.backend.swiglu(x, y);
    }
    
    #[cfg(target_os = "windows")]
    pub fn softmax(&self, x: &mut Tensor<f32>) {
        self.backend.softmax(x);
    }
} 
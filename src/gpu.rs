use crate::tensor::Tensor;

#[cfg(target_os = "macos")]
mod metal_backend;

#[cfg(target_os = "macos")]
use metal_backend::MetalBackend;

pub trait GPUBackend {
    fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32);
    fn rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32);
    fn swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>);
    fn softmax(&self, x: &mut Tensor<f32>);
}

pub struct GPUContext {
    #[cfg(target_os = "macos")]
    backend: MetalBackend,
}

impl GPUContext {
    pub fn new() -> Result<Self, String> {
        #[cfg(target_os = "macos")]
        {
            let backend = MetalBackend::new()?;
            Ok(GPUContext { backend })
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Err("GPU not supported on this platform".to_string())
        }
    }
    
    pub fn is_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            MetalBackend::is_available()
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

impl GPUBackend for GPUContext {
    fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32) {
        #[cfg(target_os = "macos")]
        {
            self.backend.matmul(a, b, c, alpha, beta);
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            // Fallback to CPU
            crate::operators::matmul_transb(c, beta, a, b, alpha);
        }
    }
    
    fn rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32) {
        #[cfg(target_os = "macos")]
        {
            self.backend.rms_norm(x, w, y, eps);
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            // Fallback to CPU
            crate::operators::rms_norm(y, x, w, eps);
        }
    }
    
    fn swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>) {
        #[cfg(target_os = "macos")]
        {
            self.backend.swiglu(x, y);
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            // Fallback to CPU
            crate::operators::swiglu(y, x);
        }
    }
    
    fn softmax(&self, x: &mut Tensor<f32>) {
        #[cfg(target_os = "macos")]
        {
            self.backend.softmax(x);
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            // Fallback to CPU
            crate::operators::masked_softmax(x);
        }
    }
} 
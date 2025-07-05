use crate::tensor::Tensor;
use crate::gpu::GPUBackend;
use metal::{Device, CommandQueue, Buffer};
use std::sync::Arc;

pub struct MetalBackend {
    device: Arc<Device>,
    command_queue: Arc<CommandQueue>,
}

impl GPUBackend for MetalBackend {
    fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32) {
        // 暂时使用CPU实现，后续可以优化为GPU实现
        crate::operators::matmul_transb(c, beta, a, b, alpha);
    }
    
    fn rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32) {
        // 暂时使用CPU实现
        crate::operators::rms_norm(y, x, w, eps);
    }
    
    fn swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>) {
        // 暂时使用CPU实现
        crate::operators::swiglu(y, x);
    }
    
    fn softmax(&self, x: &mut Tensor<f32>) {
        // 暂时使用CPU实现
        crate::operators::masked_softmax(x);
    }
}

impl MetalBackend {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device available")?;
        
        let command_queue = device.new_command_queue();
        
        Ok(MetalBackend {
            device: Arc::new(device),
            command_queue: Arc::new(command_queue),
        })
    }
    
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }
    
    fn create_buffer(&self, data: &[f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            (data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared
        )
    }
    
    fn create_buffer_mut(&self, data: &mut [f32]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_mut_ptr() as *mut std::ffi::c_void,
            (data.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared
        )
    }
} 
use crate::tensor::Tensor;
use metal::{Device, CommandQueue, Buffer};
use std::sync::Arc;

pub struct MetalBackend {
    device: Arc<Device>,
    command_queue: Arc<CommandQueue>,
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

impl MetalBackend {
    pub fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32) {
        // 使用MPS矩阵乘法
        let a_buffer = self.create_buffer(a.data());
        let b_buffer = self.create_buffer(b.data());
        let c_buffer = self.create_buffer_mut(unsafe { c.data_mut() });
        
        // 执行矩阵乘法
        let command_buffer = self.command_queue.new_command_buffer();
        let compute_encoder = command_buffer.new_compute_command_encoder();
        
        // 这里需要实现具体的MPS矩阵乘法
        // 暂时使用CPU实现作为fallback
        crate::operators::matmul_transb(c, beta, a, b, alpha);
        
        compute_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
    
    pub fn rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32) {
        // 使用MPS归一化
        let x_buffer = self.create_buffer(x.data());
        let w_buffer = self.create_buffer(w.data());
        let y_buffer = self.create_buffer_mut(unsafe { y.data_mut() });
        
        // 暂时使用CPU实现
        crate::operators::rms_norm(y, x, w, eps);
    }
    
    pub fn swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>) {
        // 使用MPS激活函数
        let x_buffer = self.create_buffer(x.data());
        let y_buffer = self.create_buffer_mut(unsafe { y.data_mut() });
        
        // 暂时使用CPU实现
        crate::operators::swiglu(y, x);
    }
    
    pub fn softmax(&self, x: &mut Tensor<f32>) {
        // 使用MPS softmax
        let x_buffer = self.create_buffer_mut(unsafe { x.data_mut() });
        
        // 暂时使用CPU实现
        crate::operators::masked_softmax(x);
    }
} 
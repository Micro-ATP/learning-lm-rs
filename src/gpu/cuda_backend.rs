use crate::tensor::Tensor;
use crate::gpu::GPUBackend;
use std::sync::Arc;

pub struct CudaBackend {
    device: cuda::Device,
    context: cuda::Context,
}

impl GPUBackend for CudaBackend {
    fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32) {
        // 暂时使用CPU实现，后续可以优化为CUDA实现
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

impl CudaBackend {
    pub fn new() -> Result<Self, String> {
        // 初始化CUDA
        cuda::init().map_err(|e| format!("Failed to initialize CUDA: {:?}", e))?;
        
        // 获取设备数量
        let device_count = cuda::device::count()
            .map_err(|e| format!("Failed to get CUDA device count: {:?}", e))?;
        
        if device_count == 0 {
            return Err("No CUDA devices found".to_string());
        }
        
        // 选择第一个设备（通常是性能最好的）
        let device = cuda::Device::get(0)
            .map_err(|e| format!("Failed to get CUDA device: {:?}", e))?;
        
        // 创建CUDA上下文
        let context = device.create_context()
            .map_err(|e| format!("Failed to create CUDA context: {:?}", e))?;
        
        // 设置当前上下文
        context.set_current()
            .map_err(|e| format!("Failed to set CUDA context: {:?}", e))?;
        
        Ok(CudaBackend {
            device,
            context,
        })
    }
    
    pub fn is_available() -> bool {
        match cuda::init() {
            Ok(_) => {
                match cuda::device::count() {
                    Ok(count) => count > 0,
                    Err(_) => false,
                }
            }
            Err(_) => false,
        }
    }
    
    pub fn get_device_info(&self) -> Result<String, String> {
        let name = self.device.name()
            .map_err(|e| format!("Failed to get device name: {:?}", e))?;
        
        let compute_capability = self.device.compute_capability()
            .map_err(|e| format!("Failed to get compute capability: {:?}", e))?;
        
        let total_memory = self.device.total_memory()
            .map_err(|e| format!("Failed to get total memory: {:?}", e))?;
        
        Ok(format!("{} (Compute Capability: {}.{}, Memory: {} MB)", 
                   name, compute_capability.major, compute_capability.minor, 
                   total_memory / 1024 / 1024))
    }
    
    fn create_buffer(&self, data: &[f32]) -> Result<cuda::memory::DeviceBuffer<f32>, String> {
        cuda::memory::DeviceBuffer::from_slice(data)
            .map_err(|e| format!("Failed to create CUDA buffer: {:?}", e))
    }
    
    fn create_buffer_mut(&self, data: &mut [f32]) -> Result<cuda::memory::DeviceBuffer<f32>, String> {
        cuda::memory::DeviceBuffer::from_slice(data)
            .map_err(|e| format!("Failed to create CUDA buffer: {:?}", e))
    }
} 
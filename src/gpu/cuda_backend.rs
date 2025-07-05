use crate::tensor::Tensor;
use crate::gpu::GPUBackend;
use std::sync::Once;

static INIT: Once = Once::new();
static mut LAUNCH_MATMUL_FN: Option<LaunchMatmulFn> = None;
static mut LAUNCH_RMS_NORM_FN: Option<LaunchRmsNormFn> = None;
static mut LAUNCH_SWIGLU_FN: Option<LaunchSwigluFn> = None;
static mut LAUNCH_SOFTMAX_FN: Option<LaunchSoftmaxFn> = None;

type LaunchMatmulFn = unsafe extern "C" fn(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    beta: f32,
);

type LaunchRmsNormFn = unsafe extern "C" fn(
    x: *const f32,
    w: *const f32,
    y: *mut f32,
    batch_size: i32,
    hidden_size: i32,
    eps: f32,
);

type LaunchSwigluFn = unsafe extern "C" fn(
    x: *const f32,
    y: *mut f32,
    size: i32,
);

type LaunchSoftmaxFn = unsafe extern "C" fn(
    x: *mut f32,
    seq_len: i32,
    batch_size: i32,
);

pub struct CudaBackend {
    device_id: i32,
}

impl GPUBackend for CudaBackend {
    fn matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32) {
        self.cuda_matmul(a, b, c, alpha, beta);
    }
    
    fn rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32) {
        self.cuda_rms_norm(x, w, y, eps);
    }
    
    fn swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>) {
        self.cuda_swiglu(x, y);
    }
    
    fn softmax(&self, x: &mut Tensor<f32>) {
        self.cuda_softmax(x);
    }
}

impl CudaBackend {
    pub fn new() -> Result<Self, String> {
        if !Self::is_available() {
            return Err("CUDA not available".to_string());
        }
        
        // 初始化CUDA函数指针
        unsafe {
            INIT.call_once(|| {
                if let Ok(lib) = libloading::Library::new("src/shaders.dll") {
                    // 加载矩阵乘法kernel
                    if let Ok(func) = lib.get::<LaunchMatmulFn>(b"launch_matmul") {
                        LAUNCH_MATMUL_FN = Some(*func);
                        println!("✅ CUDA矩阵乘法kernel加载成功");
                    } else {
                        println!("⚠️  CUDA矩阵乘法kernel加载失败");
                    }
                    
                    // 加载RMS归一化kernel
                    if let Ok(func) = lib.get::<LaunchRmsNormFn>(b"launch_rms_norm") {
                        LAUNCH_RMS_NORM_FN = Some(*func);
                        println!("✅ CUDA RMS归一化kernel加载成功");
                    } else {
                        println!("⚠️  CUDA RMS归一化kernel加载失败");
                    }
                    
                    // 加载SwiGLU kernel
                    if let Ok(func) = lib.get::<LaunchSwigluFn>(b"launch_swiglu") {
                        LAUNCH_SWIGLU_FN = Some(*func);
                        println!("✅ CUDA SwiGLU kernel加载成功");
                    } else {
                        println!("⚠️  CUDA SwiGLU kernel加载失败");
                    }
                    
                    // 加载Softmax kernel
                    if let Ok(func) = lib.get::<LaunchSoftmaxFn>(b"launch_softmax") {
                        LAUNCH_SOFTMAX_FN = Some(*func);
                        println!("✅ CUDA Softmax kernel加载成功");
                    } else {
                        println!("⚠️  CUDA Softmax kernel加载失败");
                    }
                } else {
                    println!("⚠️  无法加载CUDA动态库 (src/shaders.dll)");
                }
            });
        }
        
        Ok(CudaBackend { device_id: 0 })
    }
    
    pub fn is_available() -> bool {
        std::env::var("CUDA_PATH").is_ok() || std::env::var("CUDA_HOME").is_ok()
    }
    
    pub fn get_device_info(&self) -> Result<String, String> {
        let output = std::process::Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .output();
        
        match output {
            Ok(output) => {
                if output.status.success() {
                    let info = String::from_utf8_lossy(&output.stdout);
                    let lines: Vec<&str> = info.lines().collect();
                    if !lines.is_empty() {
                        let parts: Vec<&str> = lines[0].split(", ").collect();
                        if parts.len() >= 2 {
                            return Ok(format!("{} (Memory: {} MB)", parts[0], parts[1]));
                        }
                    }
                }
                Ok("CUDA GPU detected".to_string())
            }
            Err(_) => Ok("CUDA backend available (using CPU fallback for now)".to_string())
        }
    }
    
    // CUDA矩阵乘法实现
    fn cuda_matmul(&self, a: &Tensor<f32>, b: &Tensor<f32>, c: &mut Tensor<f32>, alpha: f32, beta: f32) {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let m = a_shape[0] as i32;
        let k = a_shape[1] as i32;
        let n = b_shape[0] as i32;
        
        unsafe {
            if let Some(func) = LAUNCH_MATMUL_FN {
                func(
                    a.data().as_ptr(),
                    b.data().as_ptr(),
                    c.data_mut().as_mut_ptr(),
                    m, n, k, alpha, beta,
                );
                // 只在调试模式下显示GPU使用确认
                #[cfg(debug_assertions)]
                println!("🚀 GPU矩阵乘法: {}x{} @ {}x{}", m, k, n, k);
            } else {
                // Fallback to CPU implementation
                println!("⚠️  CUDA kernel未加载，使用CPU回退");
                crate::operators::matmul_transb(c, beta, a, b, alpha);
            }
        }
    }
    
    // CUDA RMS归一化实现
    fn cuda_rms_norm(&self, x: &Tensor<f32>, w: &Tensor<f32>, y: &mut Tensor<f32>, eps: f32) {
        let x_shape = x.shape();
        let batch_size = x.size() / x_shape[x_shape.len() - 1];
        let hidden_size = x_shape[x_shape.len() - 1];
        
        unsafe {
            if let Some(func) = LAUNCH_RMS_NORM_FN {
                func(
                    x.data().as_ptr(),
                    w.data().as_ptr(),
                    y.data_mut().as_mut_ptr(),
                    batch_size as i32,
                    hidden_size as i32,
                    eps,
                );
                // 只在调试模式下显示GPU使用确认
                #[cfg(debug_assertions)]
                println!("🚀 GPU RMS归一化: batch={}, hidden_size={}", batch_size, hidden_size);
            } else {
                // Fallback to CPU implementation
                println!("⚠️  CUDA RMS归一化kernel未加载，使用CPU回退");
                crate::operators::rms_norm(y, x, w, eps);
            }
        }
    }
    
    // CUDA SwiGLU实现
    fn cuda_swiglu(&self, x: &Tensor<f32>, y: &mut Tensor<f32>) {
        let size = x.size() as i32;
        
        unsafe {
            if let Some(func) = LAUNCH_SWIGLU_FN {
                func(
                    x.data().as_ptr(),
                    y.data_mut().as_mut_ptr(),
                    size,
                );
                // 只在调试模式下显示GPU使用确认
                #[cfg(debug_assertions)]
                println!("🚀 GPU SwiGLU: size={}", size);
            } else {
                // Fallback to CPU implementation
                println!("⚠️  CUDA SwiGLU kernel未加载，使用CPU回退");
                crate::operators::swiglu(y, x);
            }
        }
    }
    
    // CUDA Softmax实现
    fn cuda_softmax(&self, x: &mut Tensor<f32>) {
        let shape = x.shape();
        let seq_len = shape[shape.len() - 1] as i32;
        let batch_size = (x.size() / shape[shape.len() - 1]) as i32;
        
        unsafe {
            if let Some(func) = LAUNCH_SOFTMAX_FN {
                func(
                    x.data_mut().as_mut_ptr(),
                    seq_len,
                    batch_size,
                );
                // 只在调试模式下显示GPU使用确认
                #[cfg(debug_assertions)]
                println!("🚀 GPU Softmax: seq_len={}, batch_size={}", seq_len, batch_size);
            } else {
                // Fallback to CPU implementation
                println!("⚠️  CUDA Softmax kernel未加载，使用CPU回退");
                crate::operators::masked_softmax(x);
            }
        }
    }
} 
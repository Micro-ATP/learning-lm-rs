mod config;
mod gpu;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::io::{self, Write};

fn debug_model_params() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let model_file = std::fs::read(model_dir.join("model.safetensors")).unwrap();
    let safetensor = safetensors::SafeTensors::deserialize(&model_file).unwrap();
    
    println!("Available tensor names:");
    for name in safetensor.tensors() {
        println!("  {}", name.0);
    }
}

fn check_gpu_status() {
    println!("🔍 检查GPU状态...");
    
    #[cfg(target_os = "macos")]
    {
        // 检查Metal是否可用
        if crate::gpu::metal_backend::MetalBackend::is_available() {
            println!("✅ Metal GPU后端可用");
            
            // 尝试创建GPU上下文
            match crate::gpu::GPUContext::new() {
                Ok(_) => println!("✅ GPU上下文创建成功"),
                Err(e) => println!("❌ GPU上下文创建失败: {}", e),
            }
        } else {
            println!("❌ Metal GPU后端不可用");
        }
        
        // 显示系统信息
        println!("💻 系统信息:");
        println!("   - 操作系统: {}", std::env::consts::OS);
        println!("   - 架构: {}", std::env::consts::ARCH);
        
        // 检查是否有Metal设备
        if let Some(device) = metal::Device::system_default() {
            println!("✅ 检测到Metal设备: {}", device.name());
        } else {
            println!("❌ 未检测到Metal设备");
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        // 检查CUDA是否可用
        if crate::gpu::cuda_backend::CudaBackend::is_available() {
            println!("✅ CUDA GPU后端可用");
            
            // 尝试创建GPU上下文
            match crate::gpu::GPUContext::new() {
                Ok(_) => println!("✅ GPU上下文创建成功"),
                Err(e) => println!("❌ GPU上下文创建失败: {}", e),
            }
            
            // 获取CUDA设备信息
            if let Ok(backend) = crate::gpu::cuda_backend::CudaBackend::new() {
                if let Ok(info) = backend.get_device_info() {
                    println!("✅ 检测到CUDA设备: {}", info);
                }
            }
        } else {
            println!("❌ CUDA GPU后端不可用");
        }
        
        // 显示系统信息
        println!("💻 系统信息:");
        println!("   - 操作系统: {}", std::env::consts::OS);
        println!("   - 架构: {}", std::env::consts::ARCH);
    }
    
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        println!("💻 系统信息:");
        println!("   - 操作系统: {}", std::env::consts::OS);
        println!("   - 架构: {}", std::env::consts::ARCH);
        println!("ℹ️  GPU加速: 当前平台不支持GPU加速，使用CPU推理");
    }
    
    println!();
}

fn main() {
    // 检查GPU状态
    check_gpu_status();
    
    // 调试参数名称
    // debug_model_params();
    // return;
    
    println!("请选择模式：1. AI对话  2. AI故事  (输入1或2)");
    print!("你的选择: ");
    io::stdout().flush().unwrap();
    let mut mode = String::new();
    io::stdin().read_line(&mut mode).unwrap();
    let mode = mode.trim();

    if mode == "1" {
        // AI对话模式
        println!("🚀 启动AI对话模式...");
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("chat");
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        println!("欢迎使用AI聊天机器人！输入 'quit' 退出。");
        println!("💡 提示: 当前使用CPU推理，如需GPU加速请完善Metal实现");
        let mut messages = Vec::new();
        loop {
            print!("用户: ");
            io::stdout().flush().unwrap();
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();
            if input == "quit" {
                println!("再见！");
                break;
            }
            messages.push(model::ChatMessage::new("user", input));
            let response = llama.chat(
                &messages,
                200,
                0.8,
                30,
                1.0,
            );
            println!("AI: {}", response);
            messages.push(model::ChatMessage::new("assistant", &response));
        }
    } else if mode == "2" {
        // AI故事模式
        println!("🚀 启动AI故事模式...");
        let project_dir = env!("CARGO_MANIFEST_DIR");
        let model_dir = PathBuf::from(project_dir).join("models").join("story");
        let llama = model::Llama::<f32>::from_safetensors(&model_dir);
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let input = "Once upon a time";
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        let output_ids = llama.generate(
            input_ids,
            500,
            0.8,
            30,
            1.,
        );
        print!("{}", input);
        let generated_ids = &output_ids[input_ids.len()..];
        println!("{}", tokenizer.decode(generated_ids, true).unwrap());
    } else {
        println!("无效选择，程序退出。");
    }
}

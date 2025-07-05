use std::env;
use std::path::Path;

fn main() {
    #[cfg(target_os = "macos")]
    {
        let out_dir = env::var("OUT_DIR").unwrap();
        let shader_src = "src/shaders.metal";
        let shader_lib = format!("{}/shaders.metallib", out_dir);
        
        // 编译Metal shader
        let status = std::process::Command::new("xcrun")
            .args(&["metal", "-c", shader_src, "-o", &format!("{}/shaders.air", out_dir)])
            .status()
            .expect("Failed to compile Metal shader");
        
        if !status.success() {
            panic!("Metal shader compilation failed");
        }
        
        // 链接Metal library
        let status = std::process::Command::new("xcrun")
            .args(&["metallib", &format!("{}/shaders.air", out_dir), "-o", &shader_lib])
            .status()
            .expect("Failed to link Metal library");
        
        if !status.success() {
            panic!("Metal library linking failed");
        }
        
        // 将编译好的库文件复制到输出目录
        std::fs::copy(&shader_lib, "src/shaders.metallib")
            .expect("Failed to copy Metal library");
        
        println!("cargo:rerun-if-changed=src/shaders.metal");
    }
    
    #[cfg(target_os = "windows")]
    {
        // 检查CUDA工具链
        let nvcc_output = std::process::Command::new("nvcc")
            .arg("--version")
            .output();
        
        match nvcc_output {
            Ok(output) => {
                println!("✅ 检测到CUDA工具链");
                if let Ok(version) = String::from_utf8(output.stdout) {
                    println!("   {}", version.lines().next().unwrap_or(""));
                }
                // 编译CUDA shader为DLL
                let shader_src = "src/shaders.cu";
                let shader_lib = "src/shaders.dll";
                let status = std::process::Command::new("nvcc")
                    .args(&[
                        "--shared",
                        "-o", shader_lib,
                        shader_src,
                        "-arch=sm_89",
                        "-O3"
                    ])
                    .status();
                match status {
                    Ok(exit_status) => {
                        if exit_status.success() {
                            println!("✅ CUDA shader编译成功");
                            println!("cargo:rustc-link-lib=dylib=shaders");
                            println!("cargo:rustc-link-search=native=src");
                        } else {
                            println!("⚠️  CUDA shader编译失败，将使用CPU实现");
                        }
                    }
                    Err(_) => {
                        println!("⚠️  无法编译CUDA shader，将使用CPU实现");
                    }
                }
                println!("cargo:rerun-if-changed=src/shaders.cu");
            }
            Err(_) => {
                println!("⚠️  未检测到CUDA工具链，将使用CPU实现");
            }
        }
    }
    
    println!("cargo:rerun-if-changed=src/gpu.rs");
    println!("cargo:rerun-if-changed=src/gpu/metal_backend.rs");
    println!("cargo:rerun-if-changed=src/gpu/cuda_backend.rs");
} 
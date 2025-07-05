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
    
    println!("cargo:rerun-if-changed=src/gpu.rs");
    println!("cargo:rerun-if-changed=src/gpu/metal_backend.rs");
} 
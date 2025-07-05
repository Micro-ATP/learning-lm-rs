use std::env;
use std::path::Path;

fn main() {
    // GPU支持构建脚本
    println!("cargo:rerun-if-changed=src/gpu.rs");
    println!("cargo:rerun-if-changed=src/metal_backend.rs");
} 
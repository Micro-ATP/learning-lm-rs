mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use std::env;
use tokenizers::Tokenizer;
use std::io::{self, Write};

fn main() {
    // 打印模式选择菜单
    println!("请选择模式：");
    println!("1. 聊天模式");
    println!("2. 生成故事模式");

    print!("输入你的选择 (1/2): ");
    io::stdout().flush().unwrap();

    let mut mode = String::new();
    io::stdin().read_line(&mut mode).unwrap();
    let mode = mode.trim();

    match mode {
        "1" => run_chat_mode(),
        "2" => run_story_mode(),
        _ => println!("无效的选择，请重新运行程序并选择1或2。"),
    }
}

// 生成故事模式
fn run_story_mode() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);

    // 生成结果，收集为 Vec<u32>
    let output_ids = llama.generate(
        input_ids,
        500,
        0.9,
        4,
        1.,
    ).collect::<Vec<u32>>(); // 修正：收集生成器输出为 Vec<u32>

    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

// 聊天模式
fn run_chat_mode() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let mut messages = Vec::new();

    loop {
        print!("User: ");
        io::stdout().flush().unwrap();

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim().to_string();

        if user_input.to_lowercase() == "exit" {
            break;
        }

        // 构建聊天模板
        messages.push(("user".to_string(), user_input.clone()));
        let chat_input = format_messages(&messages, true);

        // Tokenizer 编码
        let binding = tokenizer.encode(chat_input.as_str(), true).unwrap();
        let input_ids = binding.get_ids();
        println!("Encoded input: {:?}", input_ids); // 调试信息，检查编码后的 token

        // 生成回复
        let output_ids = llama.generate(
            input_ids,
            256,    // 最大生成长度
            0.9,    // top_p，控制多样性
            50,     // top_k，增大采样范围
            0.8     // temperature，提高生成随机性
        ).collect::<Vec<u32>>();

        println!("Generated token IDs: {:?}", output_ids); // 调试信息，检查生成的 token

        // Tokenizer 解码
        let response_text = tokenizer.decode(&output_ids, true).unwrap();
        println!("Decoded response: {:?}", response_text); // 调试信息，检查解码后的文本
        println!("Assistant: {}", response_text.trim()); // 输出最终结果

        // 保存assistant的回复到messages中
        messages.push(("assistant".to_string(), response_text));
    }
}


fn format_messages(messages: &[(String, String)], add_generation_prompt: bool) -> String {
    let mut chat_input = String::new();

    for (role, content) in messages {
        chat_input.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
    }

    if add_generation_prompt {
        chat_input.push_str("<|im_start|>assistant\n");
    }

    chat_input
}

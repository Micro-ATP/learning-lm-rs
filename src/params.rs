use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            // 根据数据类型正确读取数据
            let data: Vec<f32> = match tensor.dtype() {
                safetensors::Dtype::F32 => {
                    let raw_data = tensor.data();
                    let f32_data: &[f32] = unsafe { std::slice::from_raw_parts(raw_data.as_ptr() as *const f32, raw_data.len() / 4) };
                    f32_data.to_vec()
                },
                _ => {
                    tensor.data().iter().map(|&x| x as f32).collect()
                }
            };
            Tensor::new(data, &tensor.shape().iter().map(|&x| x as usize).collect())
        };
        
        let n_layers = config.num_hidden_layers;
        let mut rms_att_w = Vec::new();
        let mut wq = Vec::new();
        let mut wk = Vec::new();
        let mut wv = Vec::new();
        let mut wo = Vec::new();
        let mut rms_ffn_w = Vec::new();
        let mut w_up = Vec::new();
        let mut w_gate = Vec::new();
        let mut w_down = Vec::new();
        
        // 加载embedding_table和lm_head，兼容Mistral和Llama
        let embedding_table = if let Ok(tensor) = safetensor.tensor("model.embed_tokens.weight") {
            // Mistral结构
            let data: Vec<f32> = match tensor.dtype() {
                safetensors::Dtype::F32 => {
                    let raw_data = tensor.data();
                    let f32_data: &[f32] = unsafe { std::slice::from_raw_parts(raw_data.as_ptr() as *const f32, raw_data.len() / 4) };
                    f32_data.to_vec()
                },
                _ => {
                    tensor.data().iter().map(|&x| x as f32).collect()
                }
            };
            Tensor::new(data, &tensor.shape().iter().map(|&x| x as usize).collect())
        } else {
            // Llama结构，兼容tie_word_embeddings
            let lm_head = get_tensor("lm_head.weight");
            Tensor::new(lm_head.data().to_vec(), lm_head.shape())
        };
        let lm_head = get_tensor("lm_head.weight");
        
        // 加载各层的参数
        for layer in 0..n_layers {
            rms_att_w.push(get_tensor(&format!("model.layers.{}.input_layernorm.weight", layer)));
            wq.push(get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", layer)));
            wk.push(get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", layer)));
            wv.push(get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", layer)));
            wo.push(get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", layer)));
            rms_ffn_w.push(get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", layer)));
            w_up.push(get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", layer)));
            w_gate.push(get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", layer)));
            w_down.push(get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", layer)));
        }
        
        // 加载输出层参数
        let rms_out_w = get_tensor("model.norm.weight");
        
        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}

use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::gpu::{GPUContext, GPUBackend};
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
    model_type: String,     // model type: "llama" or "mistral"
    gpu_context: Option<GPUContext>, // GPU上下文
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        // 检测模型类型
        let model_type = if config.model_type == "mistral" {
            "mistral"
        } else {
            "llama"
        };

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
            model_type: model_type.to_string(),
            gpu_context: None,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                &q,
                &full_k,
                &full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            // 输出投影: out = attn_V @ O_weight.T
            let mut out_proj = Tensor::<f32>::default(&vec![seq_len, self.d]);
            OP::matmul_transb(&mut out_proj, 0., &hidden_states, &self.params.wo[layer], 1.0);
            
            // 残差连接: residual = out + residual
            let out_data = out_proj.data();
            let residual_data = unsafe { residual.data_mut() };
            for i in 0..out_proj.size() {
                residual_data[i] += out_data[i];
            }

            // MLP层
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        
        // 重新创建hidden_states和residual张量，避免slice问题
        let mut final_hidden = Tensor::<f32>::default(&vec![1, self.d]);
        let mut final_residual = Tensor::<f32>::default(&vec![1, self.d]);
        
        // 复制最后一个token的hidden_states和residual
        let hidden_data = hidden_states.data();
        let residual_data = residual.data();
        let final_hidden_data = unsafe { final_hidden.data_mut() };
        let final_residual_data = unsafe { final_residual.data_mut() };
        
        for i in 0..self.d {
            final_hidden_data[i] = hidden_data[(seq_len - 1) * self.d + i];
            final_residual_data[i] = residual_data[(seq_len - 1) * self.d + i];
        }

        OP::rms_norm(
            &mut final_hidden,
            &final_residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &final_hidden, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<u32> {
        let mut result = token_ids.to_vec();
        let mut cache = self.new_cache();
        let mut input_ids = token_ids.to_vec();

        for _ in 0..max_len {
            if result.len() >= self.max_seq_len {
                break;
            }
            let input_tensor = Tensor::new(input_ids.clone(), &vec![input_ids.len()]);
            let logits = self.forward(&input_tensor, &mut cache);
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);
            result.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            input_ids = vec![next_token]; // 只输入最新token
        }
        
        result
    }

    pub fn chat(
        &self,
        messages: &[ChatMessage],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> String {
        // 构建对话模板
        let mut prompt = String::new();
        
        // 添加所有历史消息
        for message in messages {
            prompt.push_str(&format!("<|im_start|>{}\n{}\n<|im_end|>\n", 
                message.role, message.content));
        }
        
        // 添加assistant的起始标记
        prompt.push_str("<|im_start|>assistant\n");
        
        // 根据模型类型选择tokenizer路径
        let tokenizer_path = if self.model_type == "mistral" {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join("chat")
                .join("tokenizer.json")
        } else {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models")
                .join("story")
                .join("tokenizer.json")
        };
        
        // 编码输入
        let binding = tokenizers::Tokenizer::from_file(tokenizer_path).unwrap();
        let encoding = binding.encode(&*prompt, true).unwrap();
        let input_ids = encoding.get_ids();
        
        // 生成回复
        let output_ids = self.generate(
            input_ids,
            max_len,
            top_p,
            top_k,
            temperature,
        );
        
        // 解码生成的内容（只解码新生成的部分）
        let generated_ids = &output_ids[input_ids.len()..];
        binding.decode(generated_ids, true).unwrap()
    }

    /// 批量文本生成
    pub fn generate_batch(
        &self,
        batch_token_ids: &[Vec<u32>],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Vec<Vec<u32>> {
        batch_token_ids
            .iter()
            .map(|input| self.generate(input, max_len, top_p, top_k, temperature))
            .collect()
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) {
    // 1. 计算注意力分数: score = Q @ K.T / sqrt(dim)
    // 对于每个KV头，计算对应的Q头组的注意力分数
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();
    let att_scores_data = unsafe { att_scores.data_mut() };
    
    // 计算缩放因子
    let scale = (dqkv as f32).sqrt();
    
    // 对每个KV头计算注意力分数
    for kv_head in 0..n_kv_h {
        for group in 0..n_groups {
            let q_head = kv_head * n_groups + group;
            
            // 对每个序列位置计算注意力分数
            for seq_pos in 0..seq_len {
                for total_pos in 0..total_seq_len {
                    let mut score = 0.0;
                    
                    // 计算Q和K的点积
                    for dim in 0..dqkv {
                        let q_idx = seq_pos * n_kv_h * n_groups * dqkv + q_head * dqkv + dim;
                        let k_idx = total_pos * n_kv_h * dqkv + kv_head * dqkv + dim;
                        score += q_data[q_idx] * k_data[k_idx];
                    }
                    
                    // 应用缩放
                    score /= scale;
                    
                    // 存储注意力分数
                    let att_idx = kv_head * n_groups * seq_len * total_seq_len 
                                + group * seq_len * total_seq_len 
                                + seq_pos * total_seq_len 
                                + total_pos;
                    att_scores_data[att_idx] = score;
                }
            }
        }
    }
    
    // 2. 应用softmax
    OP::masked_softmax(att_scores);
    
    // 3. 计算注意力输出: attn_V = attn @ V
    let attn_v_data = unsafe { hidden_states.data_mut() };
    let att_scores_data = att_scores.data(); // 重新获取不可变引用
    
    // 对每个KV头计算注意力输出
    for kv_head in 0..n_kv_h {
        for group in 0..n_groups {
            let q_head = kv_head * n_groups + group;
            
            // 对每个序列位置计算输出
            for seq_pos in 0..seq_len {
                for dim in 0..dqkv {
                    let mut output = 0.0;
                    
                    // 对每个总序列位置计算加权和
                    for total_pos in 0..total_seq_len {
                        let att_idx = kv_head * n_groups * seq_len * total_seq_len 
                                    + group * seq_len * total_seq_len 
                                    + seq_pos * total_seq_len 
                                    + total_pos;
                        let v_idx = total_pos * n_kv_h * dqkv + kv_head * dqkv + dim;
                        output += att_scores_data[att_idx] * v_data[v_idx];
                    }
                    
                    // 存储输出
                    let out_idx = seq_pos * n_kv_h * n_groups * dqkv + q_head * dqkv + dim;
                    attn_v_data[out_idx] = output;
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,
    hidden_states: &mut Tensor<f32>,
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    // hidden = rms_norm(residual)
    OP::rms_norm(hidden_states, residual, rms_w, eps);
    // gate = hidden @ gate_weight.T
    OP::matmul_transb(gate, 0., hidden_states, w_gate, 1.0);
    // up = hidden @ up_weight.T
    OP::matmul_transb(up, 0., hidden_states, w_up, 1.0);
    // up = silu(gate) * up
    let gate_x = gate.data().to_vec();
    let gate_x_tensor = Tensor::new(gate_x, gate.shape());
    OP::swiglu(up, &gate_x_tensor);
    // output = up @ down_weight.T
    OP::matmul_transb(hidden_states, 0., up, w_down, 1.0);
    // residual = output + residual
    let hidden_data = hidden_states.data();
    let residual_size = residual.size();
    let residual_mut = unsafe { residual.data_mut() };
    for i in 0..residual_size {
        residual_mut[i] += hidden_data[i];
    }
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}

// 聊天消息结构
#[derive(Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

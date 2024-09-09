use std::fs::File;
use std::vec;
use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::operators::{dot, masked_softmax, matmul_transb, rms_norm, silu};
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
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

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

        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = &mut q_buf.reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
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
                q, 
                full_k, 
                full_v, 
                self.n_kv_h, 
                n_groups, 
                seq_len, 
                total_seq_len, 
                self.dqkv);

            OP::matmul_transb(
                &mut residual, 1.0, &hidden_states, &self.params.wo[layer], 1.0);

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

        // 修复 hidden_states 和 residual 的可变性问题
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> LlamaGenerator<f32> {
        let mut cache = self.new_cache();
        let mut logits = Tensor::<u32>::new(token_ids.to_vec(), &vec![token_ids.len()]);
        LlamaGenerator {
            model: self,
            cache,
            logits,
            max_seq_len: self.max_seq_len,
            max_tokens: max_len,
            eos_token_id: self.eos_token_id,
            top_p,
            top_k,
            temperature,
            input_token_len: token_ids.len(),
            generated_count: 0,
            sample: 0,
        }
    }
}

pub struct LlamaGenerator<'a, T> {
    model: &'a Llama<T>,
    cache: KVCache<T>,
    logits: Tensor<u32>,
    max_seq_len: usize,
    max_tokens: usize,
    eos_token_id: u32,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    input_token_len: usize,
    generated_count: usize,
    sample: u32,
}

impl Iterator for LlamaGenerator<'_, f32> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.generated_count == self.max_tokens
            || self.max_seq_len == (self.generated_count + self.input_token_len)
        {
            None
        } else {
            let logits_f32 = self.model.forward(&self.logits, &mut self.cache);
            self.sample = OP::random_sample(&logits_f32, self.top_p, self.top_k, self.temperature);
            self.generated_count += 1;
            if self.sample == self.eos_token_id {
                return None;
            }
            self.logits = Tensor::<u32>::new(vec![self.sample], &vec![1]);
            Some(self.sample)
        }
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
    assert_eq!(
        hidden_states.shape(),
        &vec![seq_len, n_kv_h * n_groups * dqkv]
    );
    assert_eq!(
        att_scores.shape(),
        &vec![n_kv_h, n_groups, seq_len, total_seq_len]
    );
    assert_eq!(q.shape(), &vec![seq_len, n_kv_h * n_groups, dqkv]);
    assert_eq!(k.shape(), &vec![total_seq_len, n_kv_h * dqkv]);
    assert_eq!(v.shape(), &vec![total_seq_len, n_kv_h * dqkv]);

    let att_scores_data = unsafe { att_scores.data_mut() };
    let mut att_scores_data_offset = 0;
    let q_data = q.data();
    let k_data = k.data();
    let v_data = v.data();

    for head_index in 0..n_kv_h {
        for group_index in 0..n_groups {
            for seq_i in 0..seq_len {
                for total_index in 0..total_seq_len {
                    let q_start_index = head_index * n_groups * dqkv
                        + group_index * dqkv
                        + seq_i * n_kv_h * n_groups * dqkv;
                    let query_matrix_line = &q_data[q_start_index..q_start_index + dqkv];
                    let k_start_index = head_index * dqkv + total_index * n_kv_h * dqkv;
                    let k_matrix_line = &k_data[k_start_index..k_start_index + dqkv];
                    att_scores_data[att_scores_data_offset] = query_matrix_line
                        .iter()
                        .zip(k_matrix_line.iter())
                        .map(|(q, k)| q * k / (dqkv as f32).sqrt())
                        .sum();
                    att_scores_data_offset += 1;
                }
            }
        }
    }

    masked_softmax(att_scores);

    let att_scores_data = att_scores.data();
    let mut pre_transpose_hidden_states = vec![0.0f32; hidden_states.size()];
    let hidden_states_data = unsafe { hidden_states.data_mut() };
    let mut hidden_states_data_offset = 0;

    for head_index in 0..n_kv_h {
        for group_index in 0..n_groups {
            for seq_i in 0..seq_len {
                let hidden_states_start_index =
                    seq_i * dqkv + (head_index * n_groups + group_index) * seq_len * dqkv;
                for total_index in 0..dqkv {
                    let att_start_index = head_index * n_groups * seq_len * total_seq_len
                        + group_index * seq_len * total_seq_len
                        + seq_i * total_seq_len;
                    let v_start_index = head_index * dqkv + total_index;
                    pre_transpose_hidden_states[hidden_states_start_index + total_index] +=
                        att_scores_data[att_start_index..att_start_index + total_seq_len]
                            .iter()
                            .zip(v_data.iter().skip(v_start_index).step_by(n_kv_h * dqkv))
                            .map(|(q, k)| q * k)
                            .sum::<f32>();
                    hidden_states_data_offset += 1;
                }
            }
        }
    }

    hidden_states_data_offset = 0;
    for seq_i in 0..seq_len {
        for head_index in 0..n_kv_h {
            for group_index in 0..n_groups {
                let hidden_states_start_index =
                    seq_i * dqkv + (head_index * n_groups + group_index) * seq_len * dqkv;
                for total_index in 0..dqkv {
                    hidden_states_data[hidden_states_start_index + total_index] =
                        pre_transpose_hidden_states[hidden_states_data_offset];
                    hidden_states_data_offset += 1;
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
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0., hidden_states, w_gate, 1.0);
    matmul_transb(up, 0., hidden_states, w_up, 1.0);
    silu(up, gate);
    matmul_transb(hidden_states, 0., up, w_down, 1.0);
    let residual = unsafe { residual.data_mut() };
    for i in 0..residual.len() {
        residual[i] += hidden_states.data()[i];
    }
}

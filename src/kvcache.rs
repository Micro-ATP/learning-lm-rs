use std::{usize, vec};

use crate::tensor::Tensor;
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize,
    dim: usize,
    length: usize, // length of the current sequence
}

impl<T: Default + Copy> KVCache<T> {
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(),
            max_seq_len: max_seq_len,
            dim: dim,
            length: init_len,
        }
    }

    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        let remaining_len = if self.length > start { self.length - start } else { 0 };
        let actual_len = remaining_len.min(self.max_seq_len - start);
        self.k_cache[layer].slice(start * self.dim, &vec![actual_len, self.dim])
    }

    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        let remaining_len = if self.length > start { self.length - start } else { 0 };
        let actual_len = remaining_len.min(self.max_seq_len - start);
        self.v_cache[layer].slice(start * self.dim, &vec![actual_len, self.dim])
    }

    pub fn increment(&mut self, seq_len: usize){
        self.length += seq_len;
    }

    pub fn len(&self) -> usize {
        self.length
    }
}

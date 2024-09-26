use crate::tensor::Tensor;
use num_traits::Float;
use std::ops::DivAssign;


// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}


// pub fn masked_softmax(y: &mut Tensor<f32>) {
//     let ndim = y.shape().len();
//     assert!(ndim >= 2);
//     let seq_len = y.shape()[ndim - 2];
//     let total_seq_len = y.shape()[ndim - 1];
//     let batch = y.size() / (seq_len * total_seq_len);

//     // 先获取 shape
//     let y_shape = y.shape().clone();

//     // 获取可变数据引用
//     let data = unsafe { y.data_mut() };

//     // 使用获取的 shape 信息进行调试打印
//     // println!("masked_softmax: seq_len: {}, total_seq_len: {}, batch: {}", seq_len, total_seq_len, batch);
//     // println!("masked_softmax: y.shape: {:?}", y_shape);

//     for b in 0..batch {
//         let base = b * seq_len * total_seq_len;
//         for i in 0..seq_len {
//             let offset = base + i * total_seq_len;
//             let boundary = total_seq_len - seq_len + i + 1;

//             let max = data[offset..offset + boundary]
//                 .iter()
//                 .fold(data[offset], |a, b| a.max(*b));

//             let sum = (0..boundary)
//                 .map(|j| {
//                     let e = (data[offset + j] - max).exp();
//                     data[offset + j] = e;
//                     e
//                 })
//                 .sum::<f32>();

//             (0..boundary).for_each(|j| data[offset + j] /= sum);
//             (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
//         }
//     }
// }
// pub fn masked_softmax<P: Float + std::iter::Sum + DivAssign>(y: &mut Tensor<P>) {
    pub fn masked_softmax(y: &mut Tensor<f32>) {
        let ndim = y.shape().len();
        assert!(ndim >= 2);
        let seq_len = y.shape()[ndim - 2];
        let total_seq_len = y.shape()[ndim - 1];
        let batch = y.size() / (seq_len * total_seq_len);
        let data = unsafe { y.data_mut() };
        for b in 0..batch {
            let base = b * seq_len * total_seq_len;
            for i in 0..seq_len {
                let offset = base + i * total_seq_len;
                let boundary = total_seq_len - seq_len + i + 1;
    
                let max = data[offset..offset + boundary]
                    .iter()
                    .fold(data[offset], |a, b| a.max(*b));
    
                let sum = (0..boundary)
                    .map(|j| {
                        let e = (data[offset + j] - max).exp();
                        data[offset + j] = e;
                        e
                    })
                    .sum::<f32>();
    
                (0..boundary).for_each(|j| data[offset + j] /= sum);
                (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
            }
        }
    }
    
    pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
        let shape = y.shape().clone();
        let len = shape.len();
        let last_dim = len - 1;
        let _y = unsafe { y.data_mut() };
        let _x = x.data();
        let _w = w.data();
    
        let mut ext_loop = 1;
        for i in 0..(shape.len() - 1) {
            ext_loop *= shape[i];
        }
        let inner_size = shape[last_dim];
    
        for i in 0..ext_loop {
            let mut xp = 0f32;
            for j in 0..shape[last_dim] {
                xp += _x[i * inner_size + j] * _x[i * inner_size + j];
                _y[i * inner_size + j] = _w[j] * _x[i * inner_size + j];
            }
            xp = f32::sqrt(xp / inner_size as f32 + epsilon);
            for j in 0..shape[last_dim] {
                _y[i * inner_size + j] /= xp;
            }   
        }
    }

// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for i in 0..len {
        let sigmoid_x = 1.0 / (1.0 + (-_x[i]).exp());
        _y[i] = sigmoid_x * _x[i] * _y[i];
    }
}

pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 获取矩阵的形状
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let (n, k2) = (b.shape()[0], b.shape()[1]);
    
    // 确保矩阵的形状是符合预期的
    assert_eq!(k, k2, "matmul_transb: Incompatible shapes for matmul: a: {:?}, b: {:?}", a.shape(), b.shape());
    assert_eq!(c.shape(), &[m, n], "matmul_transb: Output shape c: {:?} is not compatible with a: {:?} and b: {:?}", c.shape(), a.shape(), b.shape());

    // 获取矩阵的数据
    let a_data = a.data();
    let b_data = b.data();
    let c_data = unsafe { c.data_mut() };

    // 对矩阵进行运算
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_data[i * k + l] * b_data[j * k + l];
            }
            c_data[i * n + j] = beta * c_data[i * n + j] + alpha * sum;
        }
    }
}



// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}



// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

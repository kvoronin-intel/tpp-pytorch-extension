#include <ATen/record_function.h>
#include <torch/extension.h>
#include "rtm.h"

#include "utils.h"
#include "xsmm_functors.h"

using namespace pcl;

#ifdef FP32_OUTPUT
#define out_scalar_t float
#else
#define out_scalar_t scalar_t
#endif

template <typename scalar_t>
void embedding_bag_forward_tmpl(
    torch::Tensor t_weight,
    torch::Tensor t_input,
    torch::Tensor t_offsets,
    torch::Tensor& t_output) {
  auto N = t_offsets.size(0);
  auto NS = t_input.size(0);
  auto E = t_weight.size(1);
  t_input = t_input.contiguous();
  t_offsets = t_offsets.contiguous();

  DECL_VLA_PTR_PT(scalar_t, weight, [E], t_weight);
  DECL_VLA_PTR_PT(out_scalar_t, output, [E], t_output);
  int64_t* input = t_input.data_ptr<int64_t>();
  int64_t* offsets = t_offsets.data_ptr<int64_t>();

  auto embbag = EmbBagFwdTPP<scalar_t, out_scalar_t, int64_t>(E);

#pragma omp parallel for
  for (int n = 0; n < N; n++) {
    auto start = offsets[n];
    auto end = (n < N - 1 ? offsets[n + 1] : NS);

    embbag(output[n], weight[0], &input[start], end - start);
  }
}

// kHalf, kBFloat16, kFloat
at::Tensor embedding_bag_forward(
    torch::Tensor weight,
    torch::Tensor input,
    torch::Tensor offsets) {
  auto N = offsets.size(0);
  // auto NS = input.size(0);
  auto E = weight.size(1);
#ifdef FP32_OUTPUT
  auto opts = weight.options().dtype(at::kFloat);
#else
  auto opts = weight.options();
#endif
  at::Tensor output = at::empty({N, E}, opts);
  if (weight.dtype() == at::kFloat) {
    embedding_bag_forward_tmpl<float>(weight, input, offsets, output);
  } else if (weight.dtype() == at::kBFloat16) {
    embedding_bag_forward_tmpl<bfloat16>(weight, input, offsets, output);
  } else if (weight.dtype() == at::kHalf) {
    embedding_bag_forward_tmpl<half>(weight, input, offsets, output);
  } else {
    PCL_ASSERT(0, "This datatype is not supported\n");
  }
  return output;
}

template <typename scalar_t>
inline void embedding_bag_backward_tmpl(
    torch::Tensor t_gradout,
    torch::Tensor t_weight,
    torch::Tensor t_input,
    torch::Tensor t_offsets,
    torch::Tensor t_values) {
  auto N = t_offsets.size(0);
  auto NS = t_input.size(0);
  auto E = t_gradout.size(1);

  // DECL_VLA_PTR_PT(scalar_t, weight, [E], t_weight);
  DECL_VLA_PTR_PT(scalar_t, values, [E], t_values);
  DECL_VLA_PTR_PT(out_scalar_t, gradout, [E], t_gradout);
  // int64_t *input = t_input.data_ptr<int64_t>();
  int64_t* offsets = t_offsets.data_ptr<int64_t>();

  auto embbag_bwd = EmbBagBwdTPP<out_scalar_t, scalar_t>(E);
#pragma omp parallel for
  for (int n = 0; n < N; n++) {
    auto start = offsets[n];
    auto end = (n < N - 1 ? offsets[n + 1] : NS);
    embbag_bwd(gradout[n], values[start], end - start);
  }
}

at::Tensor embedding_bag_backward(
    torch::Tensor gradout,
    torch::Tensor weight,
    torch::Tensor input,
    torch::Tensor offsets) {
  auto NS = input.size(0);
  auto E = gradout.size(1);
  auto values = at::empty({NS, E}, weight.options());
  auto indices = input.reshape({1, -1});
  if (weight.dtype() == at::kFloat) {
    embedding_bag_backward_tmpl<float>(gradout, weight, input, offsets, values);
  } else if (weight.dtype() == at::kBFloat16) {
    embedding_bag_backward_tmpl<bfloat16>(
        gradout, weight, input, offsets, values);
  } else if (weight.dtype() == at::kHalf) {
    embedding_bag_backward_tmpl<half>(gradout, weight, input, offsets, values);
  } else {
    PCL_ASSERT(0, "This datatype is not supported\n");
  }

  auto grad_weight =
      at::_sparse_coo_tensor_unsafe(indices, values, weight.sizes());

  return grad_weight;
}

template <typename scalar_t>
void dense_sparse_add_tmpl(
    torch::Tensor t_dense,
    torch::Tensor t_sparse,
    float alpha) {
  auto NS = t_sparse._nnz();
  auto M = t_dense.size(0);
  auto E = t_dense.size(1);
  auto t_values = t_sparse._values();
  auto t_indices = t_sparse._indices();

  PCL_ASSERT(t_dense.is_contiguous(), "dense tensor must be contiguous\n");
  // Not using below due to spurious compiler warnings
  // DECL_VLA_PTR_PT(scalar_t, dense, [E], t_dense);
  // DECL_VLA_PTR_PT(scalar_t, values, [E], t_values);
  auto dense = t_dense.data_ptr<scalar_t>();
  auto values = t_values.data_ptr<scalar_t>();
  auto indices = t_indices.data_ptr<long>();
  auto lr = alpha;

  auto embbag_upd = ScaleAddTPP<scalar_t, scalar_t>(E);

  int max_thr = omp_get_max_threads();
  if (use_lock_free) {
    int nthr = max_thr;
    if (M < nthr)
      nthr = M;
#pragma omp parallel num_threads(nthr)
    {
      int tid = omp_get_thread_num();
      long j_begin = (tid * M) / nthr;
      long j_end = ((tid + 1) * M) / nthr;
      for (long i = 0; i < NS; i++) {
        auto ind = indices[i];
        if (ind >= j_begin && ind < j_end) {
          auto wa = &dense[ind * E];
          auto va = &values[i * E];
          embbag_upd(va, wa, lr);
        }
      }
    }
  } else {
    SimpleSpinLock fallBackLock;
#pragma omp parallel for
    for (int i = 0; i < NS; i++) {
      auto ind = indices[i];
      auto wa = &dense[ind * E];
      auto va = &values[i * E];
      {
        TransactionScope guard(fallBackLock, 100);
        embbag_upd(va, wa, lr);
      }
    }
  }
}

void dense_sparse_add(
    torch::Tensor dense,
    torch::Tensor sparse,
    /*torch::Scalar*/ float alpha) {
  RECORD_FUNCTION(
      "dense_sparse_add", std::vector<c10::IValue>({dense, sparse, alpha}));
  if (dense.dtype() == at::kFloat) {
    dense_sparse_add_tmpl<float>(dense, sparse, alpha);
    //} else if (dense.dtype() == at::kBFloat16) {
    //  dense_sparse_add_tmpl<bfloat16>(dense, sparse, alpha);
    //} else if (dense.dtype() == at::kHalf) {
    //  dense_sparse_add_tmpl<half>(dense, sparse, alpha);
  } else {
    PCL_ASSERT(0, "This datatype is not supported\n");
  }
}

void bf16_update(
    torch::Tensor hi_bits,
    torch::Tensor lo_bits,
    torch::Tensor grad,
    float lr) {
  MYASSERT(hi_bits.is_contiguous() && lo_bits.is_contiguous());
  grad = grad.contiguous();
  if (grad.is_sparse()) {
    RECORD_FUNCTION(
        "bf16_sparse_update",
        std::vector<c10::IValue>({hi_bits, lo_bits, grad, lr}));
    auto sparse = grad;
    auto NS = sparse._nnz();
    auto M = hi_bits.size(0);
    auto E = hi_bits.size(1);
    auto values_tensor = sparse._values();
    auto indices = sparse._indices();
    auto indices_data = indices.data_ptr<long>();
    auto split_sgd_kernel = SplitSGDTPP(E);

    auto hi_data = (unsigned short*)hi_bits.data_ptr();
    auto lo_data = (unsigned short*)lo_bits.data_ptr();
    auto values_data = values_tensor.data_ptr<at::BFloat16>();
    int max_thr = omp_get_max_threads();
    if (use_lock_free) {
      int nthr = max_thr;
      if (M < nthr)
        nthr = M;
#pragma omp parallel num_threads(nthr)
      {
        int tid = omp_get_thread_num();
        long j_begin = (tid * M) / nthr;
        long j_end = ((tid + 1) * M) / nthr;
        for (long i = 0; i < NS; i++) {
          auto ind = indices_data[i];
          if (ind >= j_begin && ind < j_end) {
            auto ha = &hi_data[ind * E];
            auto la = &lo_data[ind * E];
            auto va = &values_data[i * E];
            split_sgd_kernel((at::BFloat16*)ha, (at::BFloat16*)la, va, lr);
          }
        }
      }
    } else {
      SimpleSpinLock fallBackLock;
#pragma omp parallel for
      for (long i = 0; i < NS; i++) {
        auto ind = indices_data[i];
        auto ha = &hi_data[ind * E];
        auto la = &lo_data[ind * E];
        auto va = &values_data[i * E];
        {
          TransactionScope guard(fallBackLock, 100);
          split_sgd_kernel((at::BFloat16*)ha, (at::BFloat16*)la, va, lr);
        }
      }
    }
  } else {
    RECORD_FUNCTION(
        "bf16_dense_update",
        std::vector<c10::IValue>({hi_bits, lo_bits, grad, lr}));
    auto hi_ptr = (unsigned short*)hi_bits.data_ptr();
    auto lo_ptr = (unsigned short*)lo_bits.data_ptr();
    auto grad_ptr = grad.data_ptr<at::BFloat16>();
    long sz = hi_bits.numel();
    constexpr int block_size = 64;
    auto split_sgd_kernel = SplitSGDTPP(block_size);
    long i = 0;
#pragma omp parallel for lastprivate(i)
    for (i = 0; i < ALIGNDOWN(sz, block_size); i += block_size) {
      split_sgd_kernel(
          (at::BFloat16*)(hi_ptr + i),
          (at::BFloat16*)(lo_ptr + i),
          grad_ptr + i,
          lr);
    }
    if (i < sz) {
      auto split_sgd_kernel = SplitSGDTPP(sz - i);
      split_sgd_kernel(
          (at::BFloat16*)(hi_ptr + i),
          (at::BFloat16*)(lo_ptr + i),
          grad_ptr + i,
          lr);
    }
  }
}

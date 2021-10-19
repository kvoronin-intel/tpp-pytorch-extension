#include "init.h"
#ifdef ENABLE_RTM
#include "rtm.h"
#endif
#include "timing.h"
#include "xsmm_functors.h"

using namespace pcl;

#define MYASSERT(x)                     \
  do {                                  \
    if (!(x)) {                         \
      printf("Assert failed %s\n", #x); \
      exit(1);                          \
    }                                   \
  } while (0)

REGISTER_SCOPE(split_sgd_sparse, "splitsgd_s");
REGISTER_SCOPE(split_sgd_dense, "splitsgd_d");
REGISTER_SCOPE(dense_sparse_add, "sprse_add");
REGISTER_SCOPE(fused_adamw, "fused_adamw");
REGISTER_SCOPE(fused_lamb, "fused_lamb");
REGISTER_SCOPE(splt_adamw, "splt_adamw");
REGISTER_SCOPE(splt_lamb, "splt_lamb");
REGISTER_SCOPE(grad_norm, "grad_norm");

static int sparse_add_use_lock_free() {
  static int lock_free = -1;
  if (lock_free != -1)
    return lock_free;
#ifdef ENABLE_RTM
  char* str = getenv("PCL_USE_RTM_UPDATE");
#else
  char* str = NULL;
#endif
  if (str && atoi(str) > 0) {
    lock_free = 0;
    printf("PCL_SPARSE_ADD: Using RTM Based Update\n");
  } else {
    lock_free = 1;
    printf("PCL_SPARSE_ADD: Using Lock Free Update\n");
  }
  return lock_free;
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
  int use_lock_free = sparse_add_use_lock_free();
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
#ifdef ENABLE_RTM
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
#else
    printf("Please compile with ENABLE_RTM set\n");
    exit(1);
#endif
  }
}

void dense_sparse_add_(
    torch::Tensor dense,
    torch::Tensor sparse,
    /*torch::Scalar*/ float alpha) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(dense_sparse_add, {dense, sparse, alpha});
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

void bf16_split_add_(
    torch::Tensor hi_bits,
    torch::Tensor lo_bits,
    torch::Tensor grad,
    float lr) {
  GlobalPass _gp(UPD);
  MYASSERT(hi_bits.is_contiguous() && lo_bits.is_contiguous());
  grad = grad.contiguous();
  if (grad.is_sparse()) {
    RECORD_SCOPE(split_sgd_sparse, {hi_bits});
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
    int use_lock_free = sparse_add_use_lock_free();
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
#ifdef ENABLE_RTM
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
#else
      printf("Please compile with ENABLE_RTM set\n");
      exit(1);
#endif
    }
  } else {
    RECORD_SCOPE(split_sgd_dense, {hi_bits});
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

void fused_adamw(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float step_size,
    float lr,
    float weight_decay,
    float eps) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_adamw, {t_data});
  typedef float T;
  auto data = t_data.data_ptr<T>();
  auto grad = t_grad.data_ptr<T>();
  auto exp_avg = t_exp_avg.data_ptr<T>();
  auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
  long sz = t_data.numel();
  constexpr int BS = 64;

  auto adamw_tpp =
      SCOPEIT(FusedAdamWTPP<T>(BS, beta1, beta2, weight_decay, eps), OPTIM);

  long i;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    adamw_tpp(&data[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], step_size, lr);
  }
  if (i < sz) {
    auto adamw_tpp = SCOPEIT(
        FusedAdamWTPP<T>(sz - i, beta1, beta2, weight_decay, eps), OPTIM);
    adamw_tpp(&data[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], step_size, lr);
  }
}

void fused_split_adamw(
    at::Tensor& t_data_hi,
    at::Tensor& t_data_lo,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float step_size,
    float lr,
    float weight_decay,
    float eps) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(splt_adamw, {t_data_hi});
  typedef bfloat16 T;
  auto data_hi = t_data_hi.data_ptr<T>();
  auto data_lo = t_data_lo.data_ptr<T>();
  auto grad = t_grad.data_ptr<T>();
  auto exp_avg = t_exp_avg.data_ptr<T>();
  auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
  long sz = t_data_hi.numel();
  constexpr int BS = 64;

  auto split_adamw_tpp =
      SCOPEIT(FusedSplitAdamWTPP(BS, beta1, beta2, weight_decay, eps), OPTIM);

  long i;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    split_adamw_tpp(
        &data_hi[i],
        &data_lo[i],
        &grad[i],
        &exp_avg[i],
        &exp_avg_sq[i],
        step_size,
        lr);
  }
  if (i < sz) {
    auto split_adamw_tpp = SCOPEIT(
        FusedSplitAdamWTPP(sz - i, beta1, beta2, weight_decay, eps), OPTIM);
    split_adamw_tpp(
        &data_hi[i],
        &data_lo[i],
        &grad[i],
        &exp_avg[i],
        &exp_avg_sq[i],
        step_size,
        lr);
  }
}

template <typename T>
float norm2(T* ptr, long N) {
  constexpr int BS = 256;
  auto norm_tpp = SCOPEIT(Norm2TPP<T>(BS), OPTIM);
  float sum = 0.0f;
  long i;
#pragma omp parallel for reduction(+ : sum) lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i += BS) {
    norm_tpp(&ptr[i], &sum);
  }
  if (i < N) {
    auto norm_tpp = SCOPEIT(Norm2TPP<T>(N - i), OPTIM);
    norm_tpp(&ptr[i], &sum);
  }
  return sum;
}

template <typename T>
void tensor_scale(T* ptr, long N, float scale) {
  constexpr int BS = 256;
  auto scale_tpp = SCOPEIT((ScaleTPP<T, T>(BS)), EW_SCL);
  long i = 0;
#pragma omp parallel for lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i += BS) {
    scale_tpp(&ptr[i], &ptr[i], scale);
  }
  if (i < N) {
    auto scale_tpp = SCOPEIT((ScaleTPP<T, T>(N - i)), EW_SCL);
    scale_tpp(&ptr[i], &ptr[i], scale);
  }
}

at::Tensor clip_grad_norm(std::vector<at::Tensor>& grads, float max_norm) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(grad_norm);
  float total_norm = 0.0;
  int N = grads.size();

  for (int i = 0; i < N; i++) {
    if (grads[i].dtype() == at::kFloat) {
      total_norm += norm2(grads[i].data_ptr<float>(), grads[i].numel());
    } else if (grads[i].dtype() == at::kBFloat16) {
      total_norm += norm2(grads[i].data_ptr<bfloat16>(), grads[i].numel());
    } else {
      PCL_ASSERT(0, "Unsupported data type");
    }
  }

  total_norm = sqrtf(total_norm);
  float clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1.0) {
    for (int i = 0; i < N; i++) {
      if (grads[i].dtype() == at::kFloat) {
        tensor_scale(grads[i].data_ptr<float>(), grads[i].numel(), clip_coef);
      } else if (grads[i].dtype() == at::kBFloat16) {
        tensor_scale(
            grads[i].data_ptr<bfloat16>(), grads[i].numel(), clip_coef);
      } else {
        PCL_ASSERT(0, "Unsupported data type");
      }
    }
  }
  // printf("total_norm = %g\n", total_norm);
  return at::tensor(total_norm);
}

float fused_lamb(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    float beta1,
    float beta2,
    float weight_norm,
    float lr,
    float weight_decay,
    float eps) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_lamb, {t_data});
  typedef float T;
  auto t_adam_step = at::empty_like(t_data);
  auto data = t_data.data_ptr<T>();
  auto grad = t_grad.data_ptr<T>();
  auto exp_avg = t_exp_avg.data_ptr<T>();
  auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
  auto adam_step = t_adam_step.data_ptr<T>();
  long sz = t_data.numel();
  constexpr int BS = 64;

  auto adam_step_tpp =
      SCOPEIT(FusedAdamStepTPP<T>(BS, beta1, beta2, weight_decay, eps), OPTIM);
  auto norm_tpp = SCOPEIT(Norm2TPP<T>(BS), OPTIM);
  auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(BS)), OPTIM);

  long i;
  float adam_norm = 0.0f;
#pragma omp parallel for lastprivate(i) reduction(+ : adam_norm)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    adam_step_tpp(
        &data[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], &adam_step[i]);
    norm_tpp(&adam_step[i], &adam_norm);
  }
  if (i < sz) {
    auto adam_step_tpp = SCOPEIT(
        FusedAdamStepTPP<T>(sz - i, beta1, beta2, weight_decay, eps), OPTIM);
    auto norm_tpp = SCOPEIT(Norm2TPP<T>(sz - i), OPTIM);
    adam_step_tpp(
        &data[i], &grad[i], &exp_avg[i], &exp_avg_sq[i], &adam_step[i]);
    norm_tpp(&adam_step[i], &adam_norm);
  }

  adam_norm = sqrtf(adam_norm);
  if (weight_norm == -1.0) {
    weight_norm = sqrtf(norm2(data, sz));
  }

  auto trust_ratio = 1.0;
  if (weight_norm != 0 && adam_norm != 0) {
    trust_ratio = weight_norm / adam_norm;
  }

  lr = -lr * trust_ratio;

  float new_weight_norm = 0.0;

#pragma omp parallel for lastprivate(i) reduction(+ : new_weight_norm)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    scale_add_tpp(&adam_step[i], &data[i], lr);
    norm_tpp(&data[i], &new_weight_norm);
  }
  if (i < sz) {
    auto norm_tpp = SCOPEIT(Norm2TPP<T>(sz - i), OPTIM);
    auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(sz - i)), OPTIM);
    scale_add_tpp(&adam_step[i], &data[i], lr);
    norm_tpp(&data[i], &new_weight_norm);
  }
  new_weight_norm = sqrtf(new_weight_norm);
  if (new_weight_norm > 10.0)
    new_weight_norm = 10.0;
  return new_weight_norm;
}

REGISTER_SUBMODULE(_optim, m) {
  m.def("dense_sparse_add_", &dense_sparse_add_, "Pcl pcl_dense_sparse_add");
  m.def("bf16_split_add_", &bf16_split_add_, "Pcl pcl_bf16_update");
  m.def("fused_adamw", &fused_adamw, "Fused AdamW optimizer");
  m.def(
      "fused_split_adamw",
      &fused_split_adamw,
      "Fused AdamW optimizer for BF16");
  m.def("clip_grad_norm", &clip_grad_norm, "Pcl BERT clip_grad_norm");
  m.def("fused_lamb", &fused_lamb, "Fused LAMB optimizer");
}

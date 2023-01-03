#include "init.h"
#ifdef ENABLE_RTM
#include "rtm.h"
#endif
#include "timing.h"
#include "xsmm_functors.h"

#if (defined(__x86_64__) || defined(__i386__))
#include "ATen/native/cpu/Intrinsics.h"
#else
#define _mm_pause()
#endif

#include <atomic>

static inline void atomic_add_float(double* dst, double fvalue) {
  typedef union {
    unsigned long long intV;
    double floatV;
  } uf64_t;

  uf64_t new_value, old_value;
  std::atomic<unsigned long long>* dst_intV =
      (std::atomic<unsigned long long>*)(dst);

  old_value.floatV = *dst;
  new_value.floatV = old_value.floatV + fvalue;

  unsigned long long* old_intV = (unsigned long long*)(&old_value.intV);
  while (!std::atomic_compare_exchange_strong(
      dst_intV, old_intV, new_value.intV)) {
    _mm_pause();
    old_value.floatV = *dst;
    new_value.floatV = old_value.floatV + fvalue;
  }
}

static inline void atomic_add_float(float* dst, float fvalue) {
  typedef union {
    unsigned intV;
    float floatV;
  } uf32_t;

  uf32_t new_value, old_value;
  std::atomic<unsigned>* dst_intV = (std::atomic<unsigned>*)(dst);

  old_value.floatV = *dst;
  new_value.floatV = old_value.floatV + fvalue;

  unsigned* old_intV = (unsigned*)(&old_value.intV);
  while (!std::atomic_compare_exchange_strong(
      dst_intV, old_intV, new_value.intV)) {
    _mm_pause();
    old_value.floatV = *dst;
    new_value.floatV = old_value.floatV + fvalue;
  }
}

using namespace tpp;

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
REGISTER_SCOPE(fused_sgd,  "fused_sgd");
REGISTER_SCOPE(splt_adamw, "splt_adamw");
REGISTER_SCOPE(splt_lamb, "splt_lamb");
REGISTER_SCOPE(grad_norm, "grad_norm");

static int sparse_add_use_lock_free() {
  static int lock_free = -1;
  if (lock_free != -1)
    return lock_free;
#ifdef ENABLE_RTM
  char* str = getenv("TPP_USE_RTM_UPDATE");
#else
  char* str = NULL;
#endif
  if (str && atoi(str) > 0) {
    lock_free = 0;
    printf("TPP_SPARSE_ADD: Using RTM Based Update\n");
  } else {
    lock_free = 1;
    printf("TPP_SPARSE_ADD: Using Lock Free Update\n");
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

  TPP_ASSERT(t_dense.is_contiguous(), "dense tensor must be contiguous\n");
  // Not using below due to spurious compiler warnings
  // auto  dense = GetVLAPtr<scalar_t>( t_dense, { E});
  // auto  values = GetVLAPtr<scalar_t>( t_values, { E});
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
    TPP_ASSERT(0, "This datatype is not supported\n");
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
  if (!t_grad.is_sparse()) {
    t_grad = t_grad.contiguous();
    auto data = t_data.data_ptr<T>();
    auto exp_avg = t_exp_avg.data_ptr<T>();
    auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
    auto grad = t_grad.data_ptr<T>();
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
  } else {
    TPP_ASSERT(t_data.dim() == 2, "Sparse Adam support only 2D params\n");
    t_grad = t_grad.coalesce();
    auto t_values = t_grad._values();
    auto t_indices = t_grad._indices();
    auto data = t_data.data_ptr<T>();
    auto exp_avg = t_exp_avg.data_ptr<T>();
    auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
    auto values = t_values.data_ptr<T>();
    auto indices = t_indices.data_ptr<long>();
    auto NS = t_grad._nnz();
    // auto M = t_data.size(0);
    auto E = t_data.size(1);

    auto adamw_tpp =
        SCOPEIT(FusedAdamWTPP<T>(E, beta1, beta2, weight_decay, eps), OPTIM);
#pragma omp parallel for
    for (int i = 0; i < NS; i++) {
      auto ind = indices[i];
      adamw_tpp(
          &data[ind * E],
          &values[i * E],
          &exp_avg[ind * E],
          &exp_avg_sq[ind * E],
          step_size,
          lr);
    }
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
  if (!t_grad.is_sparse()) {
    t_grad = t_grad.contiguous();
    auto data_hi = t_data_hi.data_ptr<T>();
    auto data_lo = t_data_lo.data_ptr<T>();
    auto exp_avg = t_exp_avg.data_ptr<T>();
    auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
    auto grad = t_grad.data_ptr<T>();
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
  } else {
    TPP_ASSERT(t_data_hi.dim() == 2, "Sparse Adam support only 2D params\n");
    // TODO: BFloat16 coalesce() might not be optimal in pytorch, implement our
    // own
    // t_grad = t_grad.coalesce();
    auto t_values = t_grad._values();
    auto t_indices = t_grad._indices();
    auto data_hi = t_data_hi.data_ptr<T>();
    auto data_lo = t_data_lo.data_ptr<T>();
    auto exp_avg = t_exp_avg.data_ptr<T>();
    auto exp_avg_sq = t_exp_avg_sq.data_ptr<T>();
    auto values = t_values.data_ptr<T>();
    auto indices = t_indices.data_ptr<long>();
    auto NS = t_grad._nnz();
    // auto M = t_data_hi.size(0);
    auto E = t_data_hi.size(1);

    auto split_adamw_tpp =
        SCOPEIT(FusedSplitAdamWTPP(E, beta1, beta2, weight_decay, eps), OPTIM);
#pragma omp parallel for
    for (int i = 0; i < NS; i++) {
      auto ind = indices[i];
      split_adamw_tpp(
          &data_hi[ind * E],
          &data_lo[ind * E],
          &values[i * E],
          &exp_avg[ind * E],
          &exp_avg_sq[ind * E],
          step_size,
          lr);
    }
  }
}

template <typename T>
double norm2(T* ptr, long N) {
  constexpr int BS = 256;
  auto norm_tpp = SCOPEIT((Norm2TPP<T, double>(BS)), OPTIM);
  double sum = 0.0f;
  long i;
#pragma omp parallel for reduction(+ : sum) lastprivate(i)
  for (i = 0; i < ALIGNDOWN(N, BS); i += BS) {
    norm_tpp(&ptr[i], &sum);
  }
  if (i < N) {
    auto norm_tpp = SCOPEIT((Norm2TPP<T, double>(N - i)), OPTIM);
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

double clip_grad_norm(std::vector<at::Tensor>& grads, double max_norm) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(grad_norm);
  double total_norm = 0.0;
  int N = grads.size();

  for (int i = 0; i < N; i++) {
    if (grads[i].dtype() == at::kFloat) {
      total_norm += norm2(pt_get_data_ptr<float>(grads[i]), grads[i].numel());
    } else if (grads[i].dtype() == at::kBFloat16) {
      total_norm +=
          norm2(pt_get_data_ptr<bfloat16>(grads[i]), grads[i].numel());
    } else if (grads[i].dtype() == at::kBFloat8) {
      total_norm += norm2(pt_get_data_ptr<bfloat8>(grads[i]), grads[i].numel());
    } else {
      TPP_ASSERT(0, "Unsupported data type");
    }
  }

  total_norm = sqrt(total_norm);
  float clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1.0) {
    for (int i = 0; i < N; i++) {
      if (grads[i].dtype() == at::kFloat) {
        tensor_scale(
            pt_get_data_ptr<float>(grads[i]), grads[i].numel(), clip_coef);
      } else if (grads[i].dtype() == at::kBFloat16) {
        tensor_scale(
            pt_get_data_ptr<bfloat16>(grads[i]), grads[i].numel(), clip_coef);
      } else if (grads[i].dtype() == at::kBFloat8) {
        tensor_scale(
            pt_get_data_ptr<bfloat8>(grads[i]), grads[i].numel(), clip_coef);
      } else {
        TPP_ASSERT(0, "Unsupported data type");
      }
    }
  }
  // printf("total_norm = %g\n", total_norm);
  return total_norm;
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

  auto adam_step_tpp = SCOPEIT(
      FusedAdamStepTPP<T>(BS, beta1, beta2, eps, weight_decay > 0.0, false),
      OPTIM);
  auto norm_tpp = SCOPEIT(Norm2TPP<T>(BS), OPTIM);
  auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(BS)), OPTIM);

  long i;
  float adam_norm = 0.0f;
#pragma omp parallel for lastprivate(i) reduction(+ : adam_norm)
  for (i = 0; i < ALIGNDOWN(sz, BS); i += BS) {
    adam_step_tpp(
        &data[i],
        &grad[i],
        &exp_avg[i],
        &exp_avg_sq[i],
        &adam_step[i],
        weight_decay);
    norm_tpp(&adam_step[i], &adam_norm);
  }
  if (i < sz) {
    auto adam_step_tpp = SCOPEIT(
        FusedAdamStepTPP<T>(
            sz - i, beta1, beta2, eps, weight_decay > 0.0, false),
        OPTIM);
    auto norm_tpp = SCOPEIT(Norm2TPP<T>(sz - i), OPTIM);
    adam_step_tpp(
        &data[i],
        &grad[i],
        &exp_avg[i],
        &exp_avg_sq[i],
        &adam_step[i],
        weight_decay);
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

template <typename T, typename TN>
void fused_lamb_v2_impl(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    at::Tensor& t_adam_step,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    at::Tensor& t_weight_norms,
    at::Tensor& t_update_norms,
    float weight_decay,
    float beta1,
    float beta2,
    float lr,
    float eps,
    int block_size,
    int step,
    bool fused_param_norm) {
  const int BS = block_size;
  auto num_blocks = t_data.numel() / block_size;
  auto d = GetVLAPtr<T>(t_data, {BS});
  auto g = GetVLAPtr<T>(t_grad, {BS});
  auto m = GetVLAPtr<T>(t_exp_avg, {BS});
  auto v = GetVLAPtr<T>(t_exp_avg_sq, {BS});
  auto u = GetVLAPtr<T>(t_adam_step, {BS});
  auto dl = GetVLAPtr<T>(t_data_low, {BS});
  // auto sz = t_block_sizes.data_ptr<int>();
  auto b2p = t_block2param.data_ptr<int>();
  auto wnorm = t_weight_norms.data_ptr<TN>();
  auto unorm = t_update_norms.data_ptr<TN>();

  auto adam_step_nwd_tpp =
      SCOPEIT(FusedAdamStepTPP<T>(BS, beta1, beta2, eps, false, true), OPTIM);
  auto adam_step_wwd_tpp =
      SCOPEIT(FusedAdamStepTPP<T>(BS, beta1, beta2, eps, true, true), OPTIM);
  auto norm_tpp = SCOPEIT((Norm2TPP<T, TN>(BS)), OPTIM);
  auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(BS)), OPTIM);
  auto scale_add_split_bf16_tpp = SCOPEIT((SplitSGDTPP(BS)), OPTIM);

  long i;
  float b1_scale = 1.0 / (1.0 - pow(beta1, step));
  float b2_scale = 1.0 / (1.0 - pow(beta2, step));
  if (!fused_param_norm) {
    t_weight_norms.zero_();
    t_update_norms.zero_();
  }
  TN fused_adam_norm = 0.0;
  TN fused_weight_norm = 0.0;
#pragma omp parallel for reduction(+ : fused_adam_norm, fused_weight_norm)
  for (i = 0; i < num_blocks; i++) {
    TN adam_norm = 0.0f;
    TN wt_norm = 0.0f;
    int p_i = b2p[i] + 1;
    float wd = weight_decay;
    bool use_wd = (wd > 0.0);
    if (use_wd) {
      adam_step_wwd_tpp(d[i], g[i], m[i], v[i], u[i], wd, b1_scale, b2_scale);
      norm_tpp(d[i], &wt_norm);
      norm_tpp(u[i], &adam_norm);
      if (!fused_param_norm) {
        atomic_add_float(&wnorm[p_i], wt_norm);
        atomic_add_float(&unorm[p_i], adam_norm);
      }
      fused_adam_norm += adam_norm;
      fused_weight_norm += wt_norm;
    } else {
      adam_step_nwd_tpp(d[i], g[i], m[i], v[i], u[i], wd, b1_scale, b2_scale);
    }
  }
  if (weight_decay > 0.0) {
    wnorm[0] = fused_weight_norm;
    unorm[0] = fused_adam_norm;
  }

#pragma omp parallel for
  for (i = 0; i < num_blocks; i++) {
    auto trust_ratio = 1.0;
    int p_i = b2p[i] + 1;
    float wd = weight_decay;
    bool use_wd = (wd > 0.0);
    if (use_wd) {
      float weight_norm = fused_weight_norm;
      float adam_norm = fused_adam_norm;
      if (!fused_param_norm) {
        weight_norm = wnorm[p_i];
        adam_norm = unorm[p_i];
      }
      adam_norm = sqrtf(adam_norm);
      weight_norm = sqrtf(weight_norm);
      if (weight_norm != 0 && adam_norm != 0) {
        trust_ratio = weight_norm / adam_norm;
      }
    }
    float final_lr = -lr * trust_ratio;
    if (std::is_same<T, float>::value) {
      scale_add_tpp(u[i], d[i], final_lr);
    } else {
      scale_add_split_bf16_tpp(
          (at::BFloat16*)d[i],
          (at::BFloat16*)dl[i],
          (at::BFloat16*)u[i],
          final_lr);
    }
  }
}

void fused_lamb_v2(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_exp_avg,
    at::Tensor& t_exp_avg_sq,
    at::Tensor& t_adam_step,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    at::Tensor& t_weight_norms,
    at::Tensor& t_update_norms,
    float weight_decay,
    float beta1,
    float beta2,
    float lr,
    float eps,
    int block_size,
    int step,
    bool fused_param_norm) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_lamb, {t_data});

  if (t_weight_norms.dtype() == at::kFloat) {
    if (t_data.dtype() == at::kFloat) {
      fused_lamb_v2_impl<float, float>(
          t_data,
          t_grad,
          t_exp_avg,
          t_exp_avg_sq,
          t_adam_step,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          t_weight_norms,
          t_update_norms,
          weight_decay,
          beta1,
          beta2,
          lr,
          eps,
          block_size,
          step,
          fused_param_norm);
    } else if (t_data.dtype() == at::kBFloat16) {
      fused_lamb_v2_impl<bfloat16, float>(
          t_data,
          t_grad,
          t_exp_avg,
          t_exp_avg_sq,
          t_adam_step,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          t_weight_norms,
          t_update_norms,
          weight_decay,
          beta1,
          beta2,
          lr,
          eps,
          block_size,
          step,
          fused_param_norm);
    } else {
      TPP_ASSERT(0, "Should not come here\n");
    }
  } else if (t_weight_norms.dtype() == at::kDouble) {
    if (t_data.dtype() == at::kFloat) {
      fused_lamb_v2_impl<float, double>(
          t_data,
          t_grad,
          t_exp_avg,
          t_exp_avg_sq,
          t_adam_step,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          t_weight_norms,
          t_update_norms,
          weight_decay,
          beta1,
          beta2,
          lr,
          eps,
          block_size,
          step,
          fused_param_norm);
    } else if (t_data.dtype() == at::kBFloat16) {
      fused_lamb_v2_impl<bfloat16, double>(
          t_data,
          t_grad,
          t_exp_avg,
          t_exp_avg_sq,
          t_adam_step,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          t_weight_norms,
          t_update_norms,
          weight_decay,
          beta1,
          beta2,
          lr,
          eps,
          block_size,
          step,
          fused_param_norm);
    } else {
      TPP_ASSERT(0, "Should not come here\n");
    }
  } else {
    TPP_ASSERT(0, "Should not come here\n");
  }
}

/* Stores momentum as T (bf16) and computes in bf16 except the lr update */
template <typename T>
void fused_sgd_v0_impl(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_moment,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    float weight_decay,
    float momentum,
    float dampening,
    float nesterov,
    float lr,
    int block_size,
    int step) {
  const int BS = block_size;
  auto num_blocks = t_data.numel() / block_size;
  DECL_VLA_PTR_PT(T, d, [BS], t_data);
  DECL_VLA_PTR_PT(T, g, [BS], t_grad);
  DECL_VLA_PTR_PT(T, m, [BS], t_moment);
  DECL_VLA_PTR_PT(T, dl, [BS], t_data_low);

  auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(BS)), OPTIM);

  auto copy_tpp = SCOPEIT(CpyTPP<T>(BS), OPTIM);

  auto scale_tpp = SCOPEIT((ScaleTPP<T,T>(BS)), OPTIM);

  auto scale_add_split_bf16_tpp = SCOPEIT((SplitSGDTPP(BS)), OPTIM);

  //printf("dbg: fused_sgd_v0_impl, step = %d\n", step);

#pragma omp parallel for
  for (long i = 0; i < num_blocks; i++) {
    // correct operations but missing conversion to f32 for all except last computes
    // 1. weight decay
    if (weight_decay != 0)
      scale_add_tpp(d[i], g[i], weight_decay);
    // 2. momentum computation
    if (momentum != 0) {
        if (step == 0) {
          copy_tpp(g[i], m[i]);
        } else {
          scale_tpp(m[i], m[i], momentum);
          scale_add_tpp(g[i], m[i], 1.0f - dampening);
        }

      // nesterov
      if (nesterov != 0) {
        printf("nesterov support has not been implemented for fused_sgd_v2_impl\n");
        exit(-1);
      } else {
        copy_tpp(m[i], g[i]);
      }
    }
    // 3. lr term
    if (std::is_same<T, float>::value) {
      scale_add_tpp(g[i], d[i], -lr);
    } else {
      scale_add_split_bf16_tpp(
          (at::BFloat16*)d[i],
          (at::BFloat16*)dl[i],
          (at::BFloat16*)g[i],
          -lr);
    }
  }
}


/* Stores momentum as T (bf16) and computes in fp32 */
template <typename T>
void fused_sgd_v1_impl(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_moment,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    float weight_decay,
    float momentum,
    float dampening,
    float nesterov,
    float lr,
    int block_size,
    int step) {
  const int BS = block_size;
  auto num_blocks = t_data.numel() / block_size;
  DECL_VLA_PTR_PT(T, d, [BS], t_data);
  DECL_VLA_PTR_PT(T, g, [BS], t_grad);
  DECL_VLA_PTR_PT(T, m, [BS], t_moment);
  DECL_VLA_PTR_PT(T, dl, [BS], t_data_low);

  auto scale_add_split_bf16_tpp = SCOPEIT((SplitSGDTPP(BS)), OPTIM);

  auto scale_add_f32_tpp = SCOPEIT((ScaleAddTPP<float, float>(BS)), OPTIM);
  auto copy_f32_tpp = SCOPEIT(CpyTPP<float>(BS), OPTIM);
  auto scale_f32_tpp = SCOPEIT((ScaleTPP<float,float>(BS)), OPTIM);

  auto upconvert_tpp = SCOPEIT((ConvertTPP<T, float>(BS)), OPTIM);
  auto upconvert_split_tpp = SCOPEIT_REF(ConvertSplitTPP(BS), OPTIM);
  auto downconvert_tpp = SCOPEIT((ConvertTPP<float, T>(BS)), OPTIM);

  //printf("dbg: fused_sgd_v1_impl, step = %d\n", step);

#pragma omp parallel for
  for (long i = 0; i < num_blocks; i++) {
    LIBXSMM_ALIGNED(float g_f32[BS], 64);
    LIBXSMM_ALIGNED(float d_f32[BS], 64);
    LIBXSMM_ALIGNED(float m_f32[BS], 64);
    LIBXSMM_ALIGNED(bfloat16 g_downconvert_bf16[BS], 64);

    if (std::is_same<T, float>::value) {
      copy_f32_tpp((float*)g[i], &g_f32[0]);
      copy_f32_tpp((float*)d[i], &d_f32[0]);
      copy_f32_tpp((float*)m[i], &m_f32[0]);
    } else {
      upconvert_tpp(g[i], &g_f32[0]);
      upconvert_tpp(m[i], &m_f32[0]);
      upconvert_split_tpp((bfloat16*)d[i], (bfloat16*)dl[i], &d_f32[0]);
    }

    // 1. weight decay
    if (weight_decay != 0)
      scale_add_f32_tpp(&d_f32[0], &g_f32[0], weight_decay);
    // 2. momentum computation
    if (momentum != 0) {
        if (step == 0) {
          copy_f32_tpp(&g_f32[0], &m_f32[0]);
        } else {
          scale_f32_tpp(&m_f32[0], &m_f32[0], momentum);
          scale_add_f32_tpp(&g_f32[0], &m_f32[0], 1.0f - dampening);
        }

      // nesterov
      if (nesterov != 0) {
        printf("nesterov support has not been implemented for fused_sgd_v2_impl\n");
        exit(-1);
      } else {
        copy_f32_tpp(&m_f32[0], &g_f32[0]);
      }
    }

    // 3. lr term
    if (std::is_same<T, float>::value) {
      //scale_add_tpp(g[i], d[i], -lr);
      scale_add_f32_tpp(&g_f32[0], (float*)d[i], -lr);
    } else {
      downconvert_tpp(&m_f32[0], m[i]);
      downconvert_tpp(&g_f32[0], (T*)&g_downconvert_bf16[0]);
      scale_add_split_bf16_tpp(
          (at::BFloat16*)d[i],
          (at::BFloat16*)dl[i],
          (at::BFloat16*)&g_downconvert_bf16[0],//g[i],
          -lr);
    }
  }
}

/* Stores momentum as fp32 and computes in fp32 but has scalar non-TPP code for lr update when T = bf16 */
template <typename T>
void fused_sgd_v2_impl(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_moment,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    float weight_decay,
    float momentum,
    float dampening,
    float nesterov,
    float lr,
    int block_size,
    int step) {
  const int BS = block_size;
  auto num_blocks = t_data.numel() / block_size;
  DECL_VLA_PTR_PT(T, d, [BS], t_data);
  DECL_VLA_PTR_PT(T, g, [BS], t_grad);
  DECL_VLA_PTR_PT(float, m, [BS], t_moment);
  DECL_VLA_PTR_PT(T, dl, [BS], t_data_low);

  auto scale_add_tpp = SCOPEIT((ScaleAddTPP<T, T>(BS)), OPTIM);

  auto copy_tpp = SCOPEIT(CpyTPP<T>(BS), OPTIM);

  auto scale_tpp = SCOPEIT((ScaleTPP<T,T>(BS)), OPTIM);

  auto scale_add_split_bf16_tpp = SCOPEIT((SplitSGDTPP(BS)), OPTIM);

  auto scale_add_f32_tpp = SCOPEIT((ScaleAddTPP<float, float>(BS)), OPTIM);
  auto copy_f32_tpp = SCOPEIT(CpyTPP<float>(BS), OPTIM);
  auto scale_f32_tpp = SCOPEIT((ScaleTPP<float,float>(BS)), OPTIM);

  auto upconvert_tpp = SCOPEIT((ConvertTPP<T, float>(BS)), OPTIM);
  auto upconvert_split_tpp = SCOPEIT_REF(ConvertSplitTPP(BS), OPTIM);
  auto downconvert_tpp = SCOPEIT((ConvertTPP<float, T>(BS)), OPTIM);

  //printf("dbg: fused_sgd_v2_impl, step = %d\n", step);

#pragma omp parallel for
  for (long i = 0; i < num_blocks; i++) {
    LIBXSMM_ALIGNED(float g_f32[BS], 64);
    LIBXSMM_ALIGNED(float d_f32[BS], 64);

    if (std::is_same<T, float>::value) {
      copy_f32_tpp((float*)g[i], &g_f32[0]);
      copy_f32_tpp((float*)d[i], &d_f32[0]);
    } else {
      upconvert_tpp(g[i], &g_f32[0]);
      upconvert_split_tpp((bfloat16*)d[i], (bfloat16*)dl[i], &d_f32[0]);
    }

    // 1. weight decay
    if (weight_decay != 0)
      scale_add_f32_tpp(&d_f32[0], &g_f32[0], weight_decay);
    // 2. momentum computation
    if (momentum != 0) {
        if (step == 0) {
          copy_f32_tpp(&g_f32[0], m[i]);
        } else {
          scale_f32_tpp(m[i], m[i], momentum);
          scale_add_f32_tpp(&g_f32[0], m[i], 1.0f - dampening);
        }

      // nesterov
      if (nesterov != 0) {
        printf("nesterov support has not been implemented for fused_sgd_v2_impl\n");
        exit(-1);
      } else {
        copy_f32_tpp(m[i], &g_f32[0]);
      }
    }

    // 3. lr term
    if (std::is_same<T, float>::value) {
      scale_add_f32_tpp(&g_f32[0], (float*)d[i], -lr);
    } else {
      auto in_hi_cast = (libxsmm_bfloat16*)d[i];
      auto in_lo_cast = (libxsmm_bfloat16*)dl[i];
      for (int j = 0; j < BS; j++) {
        union libxsmm_bfloat16_f32 bf16_w;
        bf16_w.i[0] = in_lo_cast[j];
        bf16_w.i[1] = in_hi_cast[j];
        union libxsmm_bfloat16_f32 bf16_hp;
        bf16_hp.f = bf16_w.f - (float)lr * g_f32[j];
        in_lo_cast[j] = bf16_hp.i[0];
        in_hi_cast[j] = bf16_hp.i[1];
      }
    }
  }
}

/* The most up-to-date SplitSGD implementation */
/* Stores momentum as fp32, computes in fp32 but keeps fp32 version of gradient split into two bf16 parts (overlapped with bf16 gradient) */
template <typename T>
void fused_sgd_v3_impl(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_moment,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    float weight_decay,
    float momentum,
    float dampening,
    float nesterov,
    float lr,
    int block_size,
    int step) {
  const int BS = block_size;
  auto num_blocks = t_data.numel() / block_size;
  DECL_VLA_PTR_PT(T, d, [BS], t_data);
  DECL_VLA_PTR_PT(T, g, [BS], t_grad);
  DECL_VLA_PTR_PT(float, m, [BS], t_moment);
  DECL_VLA_PTR_PT(T, dl, [BS], t_data_low);

  //printf("dbg: fused_sgd_v3_impl, step = %d\n", step);

  if (std::is_same<T, float>::value) {
    auto sgd_ext_tpp = SCOPEIT((SGDExtTPP<T>(BS)), OPTIM);

#   pragma omp parallel for
    for (long i = 0; i < num_blocks; i++) {
      sgd_ext_tpp(d[i], g[i], m[i], weight_decay, dampening, momentum, lr, step);
    }
  } else { /* bf16 */

    auto split_sgd_ext_tpp = SCOPEIT((SplitSGDExtTPP(BS)), OPTIM);

#   pragma omp parallel for
    for (long i = 0; i < num_blocks; i++) {
      LIBXSMM_ALIGNED(float g_f32[BS], 64);
      split_sgd_ext_tpp((bfloat16*)dl[i], (bfloat16*)d[i], (bfloat16*)g[i], m[i], &g_f32[0], weight_decay, dampening, momentum, lr, step);
    }
  }
}


void fused_sgd_v0(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_moment,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    float weight_decay,
    float momentum,
    float dampening,
    float nesterov,
    float lr,
    int block_size,
    int step) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_sgd, {t_data});

    if (t_data.dtype() == at::kFloat) {
      fused_sgd_v0_impl<float>(
          t_data,
          t_grad,
          t_moment,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          block_size,
          step);
    } else if (t_data.dtype() == at::kBFloat16) {
      fused_sgd_v0_impl<bfloat16>(
          t_data,
          t_grad,
          t_moment,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          block_size,
          step);
    } else {
      TPP_ASSERT(0, "Should not come here\n");
    }
}

void fused_sgd_v1(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_moment,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    float weight_decay,
    float momentum,
    float dampening,
    float nesterov,
    float lr,
    int block_size,
    int step) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_sgd, {t_data});

    if (t_data.dtype() == at::kFloat) {
      fused_sgd_v1_impl<float>(
          t_data,
          t_grad,
          t_moment,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          block_size,
          step);
    } else if (t_data.dtype() == at::kBFloat16) {
      fused_sgd_v1_impl<bfloat16>(
          t_data,
          t_grad,
          t_moment,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          block_size,
          step);
    } else {
      TPP_ASSERT(0, "Should not come here\n");
    }
}

void fused_sgd_v2(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_moment,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    float weight_decay,
    float momentum,
    float dampening,
    float nesterov,
    float lr,
    int block_size,
    int step) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_sgd, {t_data});

    if (t_data.dtype() == at::kFloat) {
      fused_sgd_v2_impl<float>(
          t_data,
          t_grad,
          t_moment,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          block_size,
          step);
    } else if (t_data.dtype() == at::kBFloat16) {
      fused_sgd_v2_impl<bfloat16>(
          t_data,
          t_grad,
          t_moment,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          block_size,
          step);
    } else {
      TPP_ASSERT(0, "Should not come here\n");
    }
}

void fused_sgd_v3(
    at::Tensor& t_data,
    at::Tensor& t_grad,
    at::Tensor& t_moment,
    at::Tensor& t_data_low,
    at::Tensor& t_offsets,
    at::Tensor& t_block_sizes,
    at::Tensor& t_block2param,
    float weight_decay,
    float momentum,
    float dampening,
    float nesterov,
    float lr,
    int block_size,
    int step) {
  GlobalPass _gp(UPD);
  RECORD_SCOPE(fused_sgd, {t_data});

    if (t_data.dtype() == at::kFloat) {
      fused_sgd_v3_impl<float>(
          t_data,
          t_grad,
          t_moment,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          block_size,
          step);
    } else if (t_data.dtype() == at::kBFloat16) {
      fused_sgd_v3_impl<bfloat16>(
          t_data,
          t_grad,
          t_moment,
          t_data_low,
          t_offsets,
          t_block_sizes,
          t_block2param,
          weight_decay,
          momentum,
          dampening,
          nesterov,
          lr,
          block_size,
          step);
    } else {
      TPP_ASSERT(0, "Should not come here\n");
    }
}

REGISTER_SUBMODULE(_optim, m) {
  m.def("dense_sparse_add_", &dense_sparse_add_, "Tpp tpp_dense_sparse_add");
  m.def("bf16_split_add_", &bf16_split_add_, "Tpp tpp_bf16_update");
  m.def("fused_adamw", &fused_adamw, "Fused AdamW optimizer");
  m.def(
      "fused_split_adamw",
      &fused_split_adamw,
      "Fused AdamW optimizer for BF16");
  m.def("clip_grad_norm", &clip_grad_norm, "Tpp BERT clip_grad_norm");
  m.def("fused_lamb", &fused_lamb, "Fused LAMB optimizer");
  m.def("fused_lamb_v2", &fused_lamb_v2, "Fused LAMB optimizer version 2");
  m.def("fused_sgd_v0",  &fused_sgd_v0,  "Fused SGD  optimizer version 0 (no TPP for intemediate updates, no fp32 conversion for intermediate, bf16 momentum");
  m.def("fused_sgd_v1",  &fused_sgd_v1,  "Fused SGD  optimizer version 1 (TPP for intermediate updates, temporary fp32 copies for intermediates, bf16 momentum");
  m.def("fused_sgd_v2",  &fused_sgd_v2,  "Fused SGD  optimizer version 2 (TPP/ad hoc for intermediate updates, temporary fp32 copies for intermediates, fp32 momentum");
  m.def("fused_sgd_v3",  &fused_sgd_v3,  "Fused SGD  optimizer version 3 (TPP equations for intermediate updates, fp32/split (for bf16) weights, fp32 momentum");
}

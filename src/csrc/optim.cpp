#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace pcl;

REGISTER_SCOPE(fused_adamw, "fused_adamw");
REGISTER_SCOPE(splt_adamw, "splt_adamw");
REGISTER_SCOPE(grad_norm, "grad_norm");

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

REGISTER_SUBMODULE(_optim, m) {
  m.def("fused_adamw", &fused_adamw, "Fused AdamW optimizer");
  m.def(
      "fused_split_adamw",
      &fused_split_adamw,
      "Fused AdamW optimizer for BF16");
  m.def("clip_grad_norm", &clip_grad_norm, "Pcl BERT clip_grad_norm");
}

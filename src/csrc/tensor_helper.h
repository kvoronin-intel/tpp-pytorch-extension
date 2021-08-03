#ifndef _TENSOR_HELPER_H_
#define _TENSOR_HELPER_H_

#include "utils.h"

inline at::Tensor wt_tensor_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
#if 0
  PCL_ASSERT(Hc % 2 == 0, "Uneven number for Hc\n");
  return input.view({Nk, Nc, Hc/2, 2, Hk}).permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto Hcp2 = (Hc + 1) / 2;
  auto output = input.new_empty({Nk, Nc, Hcp2, Hk, 2});
  DECL_VLA_PTR_PT(bfloat16, out, [Hcp2 * Hk * 2], output);
  DECL_VLA_PTR_PT(bfloat16, in, [Hc * Hk], input);
  auto n2v_tpp = SCOPEIT(
      XformExtTPP<bfloat16>(Hc, Hk, Hcp2 * 2, Hk, XformTPP::XFORM_N2V_TPP),
      VNNI);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

inline at::Tensor wt_tensor_trans_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
#if 0
  PCL_ASSERT(Hk % 2 == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc, Hk/2, 2}).permute({0, 1, 3, 2, 4}).contiguous();
#else
  auto Hkp2 = (Hk + 1) / 2;
  auto output = input.new_empty({Nk, Nc, Hkp2, Hc, 2});
  DECL_VLA_PTR_PT(bfloat16, out, [Hkp2 * Hc * 2], output);
  DECL_VLA_PTR_PT(bfloat16, in, [Hc * Hk], input);
  auto trans_n2v_tpp = SCOPEIT(
      XformExtTPP<bfloat16>(
          Hc, Hk, Hc, Hkp2 * 2, XformTPP::XFORM_XPOSE_N2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    trans_n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

inline at::Tensor wt_tensor_trans_v2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
#if 0
  PCL_ASSERT(Hc % 2 == 0, "Uneven number for Hc\n");
  PCL_ASSERT(Hk % 2 == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc/2, Hk/2, 2, 2}).permute({0, 1, 3, 2, 5, 4}).contiguous().view({Nk, Nc, Hk/2, Hc, 2});
#else
  PCL_ASSERT(Hc % 2 == 0, "Uneven number for Hc\n");
  auto Hkp2 = (Hk + 1) / 2;
  auto output = input.new_empty({Nk, Nc, Hkp2, Hc, 2});
  DECL_VLA_PTR_PT(bfloat16, out, [Hkp2 * Hc * 2], output);
  DECL_VLA_PTR_PT(bfloat16, in, [Hc * Hk], input);
  auto trans_v2v_tpp = SCOPEIT(
      XformExtTPP<bfloat16>(
          Hc, Hk, Hkp2 * 2, Hc, XformTPP::XFORM_XPOSE_V2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    trans_v2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

USING_SCOPE(w_vnni);

inline at::Tensor wt_tensor_for_fwd(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  RECORD_SCOPE(w_vnni, {input});
  if (input.dtype() == at::kBFloat16) {
    if (input.dim() == 5) {
      return input;
    } else {
      return wt_tensor_n2v(Nk, Hk, Nc, Hc, input);
    }
  } else {
    return input;
  }
}

USING_SCOPE(w_xpose);

inline at::Tensor wt_tensor_for_bwd(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  RECORD_SCOPE(w_xpose, {input});
  if (input.dtype() == at::kBFloat16) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v(Nk, Hk, Nc, Hc, input);
    }
  } else {
#if 0
    return input.permute({0, 1, 3, 2}).contiguous();
#else
    auto output = input.new_empty({Nk, Nc, Hk, Hc});
    DECL_VLA_PTR_PT(float, out, [Hk * Hc], output);
    DECL_VLA_PTR_PT(float, in, [Hc * Hk], input);
    auto trans_tpp =
        SCOPEIT(XformExtTPP<float>(Hc, Hk, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < Nk * Nc; n++) {
      trans_tpp(in[n], out[n]);
    }
    return output;
#endif
  }
}

USING_SCOPE(a_xpose);

inline at::Tensor act_tensor_trans(
    long B,
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto output = input.new_empty({B, S1, N, H, S2});
  DECL_VLA_PTR_PT(bfloat16, out, [H * S2], output);
  DECL_VLA_PTR_PT(bfloat16, in, [H * S2], input);
  auto trans_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < B * S1 * N; n++) {
      trans_tpp(in[n], out[n]);
    }
  }
  return output;
#endif
}

USING_SCOPE(a_vnni);

inline at::Tensor act_tensor_n2v(
    long B,
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  PCL_ASSERT(S2 % 2 == 0, "Uneven number for S2\n");
#if 0
  return input.view({B, S1, N, S2/2, 2, H}).permute({0,1,2,3,5,4}).contiguous();
#else
  auto output = input.new_empty({B, S1, N, S2 / 2, H, 2});
  DECL_VLA_PTR_PT(bfloat16, out, [H * S2], output);
  DECL_VLA_PTR_PT(bfloat16, in, [H * S2], input);
  auto n2v_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < B * S1 * N; n++) {
      n2v_tpp(in[n], out[n]);
    }
  }
  return output;
#endif
}

inline at::Tensor get_padded_activation_for_vnni(at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  if (input.dtype() == at::kFloat)
    return input;
  constexpr int align = 2;
  auto sizes = input.sizes();
  int ndims = input.dim();
  PCL_ASSERT(ndims >= 2, "Invalid shape\n");
  auto C = sizes[ndims - 1];
  int pad = C % align;
  if (pad == 0)
    return input;
  std::vector<int64_t> new_sizes(sizes.begin(), sizes.end());
  new_sizes[ndims - 1] = align - pad;
  auto output = at::cat({input, input.new_zeros(new_sizes)}, ndims - 1);
  return output;
}

USING_SCOPE(zero);

inline void tensor_set_zero(long N, long sz, at::Tensor& input) {
#if 0
  input.zero_();
#else
  RECORD_SCOPE(zero, {input});
  if (input.dtype() == at::kFloat) {
    DECL_VLA_PTR_PT(float, in, [sz], input);
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  } else {
    DECL_VLA_PTR_PT(bfloat16, in, [sz], input);
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<bfloat16>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  }
#endif
}

#endif // _TENSOR_HELPER_H_

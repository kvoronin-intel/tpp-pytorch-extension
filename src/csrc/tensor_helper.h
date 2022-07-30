#ifndef _TENSOR_HELPER_H_
#define _TENSOR_HELPER_H_

#include "utils.h"

template<typename T>
inline constexpr int get_vnni_block_size() {
  return 4 / sizeof(T);
}

inline int get_vnni_block_size(caffe2::TypeMeta dtype) {
  if (dtype == at::kBFloat16) return 2;
  else if (dtype == at::kBFloat8) return 4;
  else return 1;
}

template<typename T>
inline at::Tensor wt_tensor_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  constexpr int BS = get_vnni_block_size<T>();
#if 0
  PCL_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  return input.view({Nk, Nc, Hc/BS, BS, Hk}).permute({0, 1, 2, 4, 3}).contiguous();
#else
  auto Hcp2 = (Hc + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hcp2, Hk, BS});
  DECL_VLA_PTR_PT(T, out, [Hcp2 * Hk * BS], output);
  DECL_VLA_PTR_PT(T, in, [Hc * Hk], input);
  auto n2v_tpp = SCOPEIT(
      XformExtTPP<T>(Hc, Hk, Hcp2 * BS, Hk, XformTPP::XFORM_N2V_TPP),
      VNNI);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

template<typename T>
inline at::Tensor wt_tensor_trans_n2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  constexpr int BS = get_vnni_block_size<T>();
#if 0
  PCL_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc, Hk/BS, BS}).permute({0, 1, 3, 2, 4}).contiguous();
#else
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hkp2, Hc, BS});
  DECL_VLA_PTR_PT(T, out, [Hkp2 * Hc * BS], output);
  DECL_VLA_PTR_PT(T, in, [Hc * Hk], input);
  auto trans_n2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hc, Hkp2 * BS, XformTPP::XFORM_XPOSE_N2V_TPP, true),
      XPOSE);
  RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
  for (int n = 0; n < Nk * Nc; n++) {
    trans_n2v_tpp(in[n], out[n]);
  }
  return output;
#endif
}

template<typename T>
inline at::Tensor wt_tensor_trans_v2v(
    long Nk,
    long Hk,
    long Nc,
    long Hc,
    at::Tensor& input) {
  constexpr int BS = get_vnni_block_size<T>();
#if 0
  PCL_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  PCL_ASSERT(Hk % BS == 0, "Uneven number for Hk\n");
  return input.view({Nk, Nc, Hc/BS, Hk/BS, BS, BS}).permute({0, 1, 3, 2, 5, 4}).contiguous().view({Nk, Nc, Hk/BS, Hc, BS});
#else
  PCL_ASSERT(Hc % BS == 0, "Uneven number for Hc\n");
  auto Hkp2 = (Hk + BS - 1) / BS;
  auto output = input.new_empty({Nk, Nc, Hkp2, Hc, BS});
  DECL_VLA_PTR_PT(T, out, [Hkp2 * Hc * BS], output);
  DECL_VLA_PTR_PT(T, in, [Hc * Hk], input);
  auto trans_v2v_tpp = SCOPEIT(
      XformExtTPP<T>(
          Hc, Hk, Hkp2 * BS, Hc, XformTPP::XFORM_XPOSE_V2V_TPP, true),
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
  if (input.dtype() != at::kFloat) {
    if (input.dim() == 5) {
      return input;
    } else {
      if (input.dtype() == at::kBFloat16) {
        return wt_tensor_n2v<bfloat16>(Nk, Hk, Nc, Hc, input);
      } else if (input.dtype() == at::kBFloat8) {
        return wt_tensor_n2v<bfloat8>(Nk, Hk, Nc, Hc, input);
      } else {
        PCL_ASSERT(false, "Unsupported datatype!");
      }
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
      return wt_tensor_trans_v2v<bfloat16>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v<bfloat16>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kBFloat8) {
    if (input.dim() == 5) {
      return wt_tensor_trans_v2v<bfloat8>(Nk, Hk, Nc, Hc, input);
    } else {
      return wt_tensor_trans_n2v<bfloat8>(Nk, Hk, Nc, Hc, input);
    }
  } else if (input.dtype() == at::kFloat) {
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
  } else {
    PCL_ASSERT(false, "Unsupported datatype!");
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
  if (input.dtype() == at::kBFloat16) {
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
  } else if (input.dtype() == at::kBFloat8) {
    DECL_VLA_PTR_PT(bfloat8, out, [H * S2], output);
    DECL_VLA_PTR_PT(bfloat8, in, [H * S2], input);
    auto trans_tpp =
      SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < B * S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else {
    PCL_ASSERT(false, "Unsupported datatype!");
  }
  return output;
#endif
}

inline at::Tensor act_tensor_trans(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_xpose, {input});
#if 0
  return input.permute({0, 1, 3, 2}).contiguous();
#else
  auto output = input.new_empty({S1, N, H, S2});
  if (input.dtype() == at::kBFloat16) {
    DECL_VLA_PTR_PT(bfloat16, out, [H * S2], output);
    DECL_VLA_PTR_PT(bfloat16, in, [H * S2], input);
    auto trans_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else if (input.dtype() == at::kBFloat8) {
    DECL_VLA_PTR_PT(bfloat8, out, [H * S2], output);
    DECL_VLA_PTR_PT(bfloat8, in, [H * S2], input);
    auto trans_tpp =
      SCOPEIT(XformExtTPP<bfloat8>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < S1 * N; n++) {
        trans_tpp(in[n], out[n]);
      }
    }
  } else {
    PCL_ASSERT(false, "Unsupported datatype!");
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
  const int BS = get_vnni_block_size(input.dtype());
  PCL_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 1
  return input.view({B, S1, N, S2/BS, BS, H}).permute({0,1,2,3,5,4}).contiguous();
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

inline at::Tensor act_tensor_n2v(
    long S1,
    long N,
    long S2,
    long H,
    at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  const int BS = get_vnni_block_size(input.dtype());
  PCL_ASSERT(S2 % BS == 0, "Uneven number for S2\n");
#if 1
  return input.view({S1, N, S2/BS, BS, H}).permute({0,1,2,4,3}).contiguous();
#else
  auto output = input.new_empty({S1, N, S2 / BS, H, BS});
  DECL_VLA_PTR_PT(bfloat16, out, [H * S2], output);
  DECL_VLA_PTR_PT(bfloat16, in, [H * S2], input);
  auto n2v_tpp =
      SCOPEIT(XformExtTPP<bfloat16>(S2, H, XformTPP::XFORM_N2V_TPP), VNNI);
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < S1 * N; n++) {
      n2v_tpp(in[n], out[n]);
    }
  }
  return output;
#endif
}

inline at::Tensor get_padded_activation_for_vnni(at::Tensor& input) {
  RECORD_SCOPE(a_vnni, {input});
  auto dtype = input.dtype();
  if (dtype == at::kFloat)
    return input;
  const int align = get_vnni_block_size(dtype);
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
  } else if (input.dtype() == at::kBFloat16) {
    DECL_VLA_PTR_PT(bfloat16, in, [sz], input);
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<bfloat16>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  } else {
    DECL_VLA_PTR_PT(bfloat8, in, [sz], input);
    auto set_zero_tpp = SCOPEIT(SetZeroTPP<bfloat8>(sz), EW_ZERO);
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      set_zero_tpp(in[n]);
    }
  }
#endif
}

#endif // _TENSOR_HELPER_H_

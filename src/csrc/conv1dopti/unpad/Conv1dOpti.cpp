/******************************************************************************
 * Copyright (c) Intel Corporation - All rights reserved.                      *
 * This file is part of the LIBXSMM library.                                   *
 *                                                                             *
 * For information on the license, see the LICENSE file.                       *
 * Further information: https://github.com/hfp/libxsmm/                        *
 * SPDX-License-Identifier: BSD-3-Clause                                       *
 ******************************************************************************/
/* Narendra Chaudhary, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/

#include <immintrin.h>
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include <omp.h>
#include <stdio.h>
#include <torch/extension.h>
#include <iostream>
#include <tuple>

#include <ATen/record_function.h>

#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

// #define TPP_ASSERT(cond, x...) do { if(!(cond)) { printf(x); fflush(stdout);
// exit(1); } } while(0)

#define XS_TILE_FORWARD 64
#define XS_TILE_DBACKWARD 64
#define XS_TILE_WBACKWARD 64 /* 256 for peak performance */

REGISTER_LOCAL_SCOPE(forward_loop, "forward_loop");
REGISTER_LOCAL_SCOPE(virtul_copy_loop, "virtul_copy_loop");
REGISTER_LOCAL_SCOPE(backward_data_loop, "backward_data_loop");
REGISTER_LOCAL_SCOPE(backward_weight_loop, "backward_weight_loop");

REGISTER_LOCAL_SCOPE(forward_loop_bf16, "forward_loop_bf16");
REGISTER_LOCAL_SCOPE(virtul_copy_loop_bf16, "virtul_copy_loop_bf16");
REGISTER_LOCAL_SCOPE(backward_data_loop_bf16, "backward_data_loop_bf16");
REGISTER_LOCAL_SCOPE(backward_weight_loop_bf16, "backward_weight_loop_bf16");

at::Tensor Conv1dOpti_forward_libxsmm(
    at::Tensor& input,
    at::Tensor& weight,
    int dilation) {
  GlobalPass _gp(FWD);
  if (input[0].dtype() == at::kFloat) {
    typedef float T;
#include "Conv1dOpti_forward_libxsmm_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "Conv1dOpti_forward_libxsmm_tmpl_bf16.h"
  }
}

std::tuple<at::Tensor, at::Tensor> Conv1dOpti_backward_libxsmm(
    at::Tensor& grad,
    at::Tensor& input,
    at::Tensor& weight,
    int dilation) {
  GlobalPass _gp(BWD);
  if (input[0].dtype() == at::kFloat) {
    typedef float T;
#include "Conv1dOpti_backward_libxsmm_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "Conv1dOpti_backward_libxsmm_tmpl_bf16.h"
  }
}

std::tuple<at::Tensor, at::Tensor> relu_forward_bf16(at::Tensor& input) {
  /* RECORD_FUNCTION("ReLU_forward_bf16", std::vector<c10::IValue>({input})); //
   * For recording time */

  int64_t N_t = input.size(0); /* Batch */
  int64_t C_t = input.size(1); /* Channel */
  int64_t W_t = input.size(2); /* input width */

  libxsmm_bfloat16* input_a = (libxsmm_bfloat16*)input.data_ptr<at::BFloat16>();

  libxsmm_blasint tpp_m = W_t; /* columns */
  libxsmm_blasint tpp_n = C_t; /* rows */
  libxsmm_blasint ldi = W_t;

  libxsmm_blasint mask_ld = ((ldi + 15) - ((ldi + 15) % 16)) / 16;
  auto mask = input.new_empty({N_t, C_t, mask_ld});
  unsigned short* mask_a = (unsigned short*)mask.data_ptr<at::BFloat16>();

  libxsmm_meltw_unary_flags unary_flags =
      LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape(
      tpp_m,
      tpp_n,
      ldi,
      ldi,
      LIBXSMM_DATATYPE_BF16,
      LIBXSMM_DATATYPE_BF16,
      LIBXSMM_DATATYPE_BF16);
  libxsmm_meltwfunction_unary relu_fwd_kernel = libxsmm_dispatch_meltw_unary_v2(
      LIBXSMM_MELTW_TYPE_UNARY_RELU, unary_shape, unary_flags);

#pragma omp parallel for
  for (int n = 0; n < N_t; n++) {
    libxsmm_meltw_unary_param relu_params;
    relu_params.in.primary = &input_a[n * C_t * W_t];
    relu_params.out.primary = &input_a[n * C_t * W_t];
    relu_params.out.secondary = &mask_a[n * C_t * mask_ld];
    relu_fwd_kernel(&relu_params);
  }

  return {input, mask};
}

at::Tensor relu_backward_bf16(at::Tensor& grad, at::Tensor& mask) {
  /* RECORD_FUNCTION("ReLU_backward_bf16", std::vector<c10::IValue>({grad,
   * output}));        // For recording time */

  int64_t N_t = grad.size(0); /* Batch */
  int64_t C_t = grad.size(1); /* Channel */
  int64_t W_t = grad.size(2); /* input width */

  libxsmm_bfloat16* grad_a = (libxsmm_bfloat16*)grad.data_ptr<at::BFloat16>();

  libxsmm_blasint tpp_m = W_t; /* columns */
  libxsmm_blasint tpp_n = C_t; /* rows */
  libxsmm_blasint ldi = W_t;

  libxsmm_blasint mask_ld = ((ldi + 15) - ((ldi + 15) % 16)) / 16;
  unsigned short* mask_a = (unsigned short*)mask.data_ptr<at::BFloat16>();

  libxsmm_meltw_unary_flags unary_flags =
      LIBXSMM_MELTW_FLAG_UNARY_BITMASK_2BYTEMULT;
  libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape(
      tpp_m,
      tpp_n,
      ldi,
      ldi,
      LIBXSMM_DATATYPE_BF16,
      LIBXSMM_DATATYPE_BF16,
      LIBXSMM_DATATYPE_BF16);
  libxsmm_meltwfunction_unary relu_bwd_kernel = libxsmm_dispatch_meltw_unary_v2(
      LIBXSMM_MELTW_TYPE_UNARY_RELU_INV, unary_shape, unary_flags);

#pragma omp parallel for
  for (int n = 0; n < N_t; n++) {
    libxsmm_meltw_unary_param relu_params;
    relu_params.in.primary = &grad_a[n * C_t * W_t];
    relu_params.out.primary = &grad_a[n * C_t * W_t];
    relu_params.in.secondary = &mask_a[n * C_t * mask_ld];
    relu_bwd_kernel(&relu_params);
  }

  return grad;
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
REGISTER_SUBMODULE(_conv1dopti, m) {
  m.def("forward", &Conv1dOpti_forward_libxsmm, "Conv1dOpti lib forward");
  m.def("backward", &Conv1dOpti_backward_libxsmm, "Conv1dOpti lib backward");
  // m.def("forward_bf16", &Conv1dOpti_forward_bf16_libxsmm, "Conv1dOpti bf16
  // forward"); m.def("backward_bf16", &Conv1dOpti_backward_bf16_libxsmm,
  // "Conv1dOpti bf16 backward");
  m.def("relu_forward_bf16", &relu_forward_bf16, "ReLU bf16 forward");
  m.def("relu_backward_bf16", &relu_backward_bf16, "ReLU bf16 backward");
}

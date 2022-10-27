
#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdlib>

#include <sys/syscall.h>
#include <iostream>
#include <mutex>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();

REGISTER_SCOPE(rgo_gemm, "rgo_gemm");
REGISTER_SCOPE(rgo_mlp, "rgo_mlp");
REGISTER_SCOPE(rgewo_gemm, "rgewo_gemm");
REGISTER_SCOPE(rgo_eltw, "rgo_eltw");
REGISTER_SCOPE(rgo_norm, "rgo_norm");
REGISTER_SCOPE(rgdi_gemm, "rgdi_gemm");
REGISTER_SCOPE(rgdi_mlp, "rgdi_mlp");
REGISTER_SCOPE(rgdw_gemm, "rgdw_gemm");
REGISTER_SCOPE(rgdw_mlp, "rgdw_mlp");
REGISTER_SCOPE(rgdbias, "rgdbias");
REGISTER_SCOPE(rgewdi_gemm, "rgewdi_gemm");
REGISTER_SCOPE(rgewdw_gemm, "rgewdw_gemm");
REGISTER_SCOPE(rgewdbias, "rgewdbias");
REGISTER_SCOPE(rgew_dbias, "rgew_dbias");
REGISTER_SCOPE(rgdnorm, "rgdnorm");
REGISTER_SCOPE(rgo_dropout, "rgo_dropout");
REGISTER_SCOPE(rgdo_dropout, "rgdo_dropout");

std::vector<at::Tensor> fused_rgcn_norm_fwd(
    int align,
    std::string norm_type,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_rgcn_norm_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_rgcn_norm_fwd.h"
  }
}

std::vector<at::Tensor> fused_rgcn_mlp_fwd(
    int align,
    std::string norm_type,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[1].dtype() == at::kFloat) {
    typedef float T;
#include "fused_rgcn_mlp_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_rgcn_mlp_flat_fwd.h"
  }
}

at::Tensor fused_rgcn_gemm_fwd(int align, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_rgcn_gemm_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_rgcn_gemm_flat_fwd.h"
  }
}

std::vector<at::Tensor> fused_rgcn_eltw_fwd(
    bool self_loop,
    int align,
    bool apply_bias,
    float p,
    std::string act,
    bool training,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_rgcn_eltw_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_rgcn_eltw_fwd.h"
  }
}

at::Tensor fused_rgcn_norm_bwd(int align, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_rgcn_norm_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_rgcn_norm_bwd.h"
  }
}

std::vector<at::Tensor> fused_rgcn_mlp_bwd(
    int align,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_rgcn_mlp_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_rgcn_mlp_flat_bwd.h"
  }
}

std::vector<at::Tensor> fused_rgcn_gemm_bwd(
    int align,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_rgcn_gemm_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_rgcn_gemm_flat_bwd.h"
  }
}

std::vector<at::Tensor> fused_rgcn_eltw_bwd(
    bool self_loop,
    int align,
    bool apply_bias,
    float p,
    std::string act,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_rgcn_eltw_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_rgcn_eltw_bwd.h"
  }
}

std::vector<at::Tensor> rgcn_dropout_fwd(
    int align,
    float p,
    bool training,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "dropout_fwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_fwd.h"
  }
}

at::Tensor rgcn_dropout_bwd(
    int align,
    float p,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "dropout_bwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_bwd.h"
  }
}

REGISTER_SUBMODULE(_fused_rgcn, m) {
  m.def("fused_rgcn_norm_fwd", &fused_rgcn_norm_fwd, "Tpp RGCN Norm forward");
  m.def("fused_rgcn_mlp_fwd", &fused_rgcn_mlp_fwd, "Tpp RGCN MLP forward");
  m.def("fused_rgcn_gemm_fwd", &fused_rgcn_gemm_fwd, "Tpp RGCN MatMul forward");
  m.def(
      "fused_rgcn_eltw_fwd", &fused_rgcn_eltw_fwd, "Tpp RGCN Eltwise forward");
  m.def("fused_rgcn_norm_bwd", &fused_rgcn_norm_bwd, "Tpp RGCN Norm backward");
  m.def("fused_rgcn_mlp_bwd", &fused_rgcn_mlp_bwd, "Tpp RGCN MLP backward");
  m.def(
      "fused_rgcn_gemm_bwd", &fused_rgcn_gemm_bwd, "Tpp RGCN MatMul backward");
  m.def(
      "fused_rgcn_eltw_bwd", &fused_rgcn_eltw_bwd, "Tpp RGCN Eltwise backward");
  m.def("rgcn_dropout_fwd", &rgcn_dropout_fwd, "Tpp Optimized Dropout FWD");
  m.def("rgcn_dropout_bwd", &rgcn_dropout_bwd, "Tpp Optimized Dropout BWD");
}

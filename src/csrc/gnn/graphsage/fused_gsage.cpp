
#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdlib>

#include <iostream>
#include <mutex>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace pcl;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();

REGISTER_SCOPE(go_gemm, "go_gemm");
REGISTER_SCOPE(gdi_gemm, "gdi_gemm");
REGISTER_SCOPE(gdw_gemm, "gdw_gemm");
REGISTER_SCOPE(gdbias, "gdbias");
REGISTER_SCOPE(gdout, "gdout");
REGISTER_SCOPE(go_dropout, "go_dropout");
REGISTER_SCOPE(gdo_dropout, "gdo_dropout");

std::vector<at::Tensor> fused_gsage_mlp_fwd(
    int align,
    bool apply_bias,
    float p,
    std::string act,
    bool res,
    bool training,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_gsage_mlp_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_gsage_mlp_flat_fwd.h"
  }
}

std::vector<at::Tensor> fused_gsage_mlp_bwd(
    int align,
    bool apply_bias,
    float p,
    std::string act,
    bool res,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_gsage_mlp_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_gsage_mlp_flat_bwd.h"
  }
}

std::vector<at::Tensor> dropout_fwd(float p, at::Tensor inp, bool training) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "dropout_fwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_fwd.h"
  }
}

at::Tensor dropout_bwd(float p, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "dropout_bwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_bwd.h"
  }
}

REGISTER_SUBMODULE(_fused_gsage, m) {
  m.def(
      "fused_gsage_mlp_fwd", &fused_gsage_mlp_fwd, "Pcl GraphSAGE MLP forward");
  m.def(
      "fused_gsage_mlp_bwd",
      &fused_gsage_mlp_bwd,
      "Pcl GraphSAGE MLP backward");
  m.def("dropout_fwd", &dropout_fwd, "Pcl Optimized Dropout FWD");
  m.def("dropout_bwd", &dropout_bwd, "Pcl Optimized Dropout BWD");
}

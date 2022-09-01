
#include <ATen/record_function.h>
//#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace pcl;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();

#define THREADED_LOOPS

#ifdef THREADED_LOOPS
#   warning "Building batchnorm with threaded loops instead of OpenMP pragmas"
#   include "threaded_loops.h"
#endif

REGISTER_SCOPE(bn_fwd_reduce,   "bn_fwd_reduce");
REGISTER_SCOPE(bn_fwd_stats,    "bn_fwd_stats");
REGISTER_SCOPE(bn_fwd_scale,    "bn_fwd_scale");

REGISTER_SCOPE(bn_bwd_w_inpadd, "bn_bwd_w_inpadd");
REGISTER_SCOPE(bn_bwd_w_add,    "bn_bwd_w_add");
REGISTER_SCOPE(bn_bwd_d,        "bn_bwd_d");

#ifndef BITS_PER_CHAR
#   define BITS_PER_CHAR (8)
#endif

std::vector<at::Tensor> batchnorm_fwd_ext(
    bool  training,
    bool  relu,
    bool  eltwise,
    float eps,
    std::vector<long> padding,
    std::string tuning_string_ncp,
    std::string tuning_string_cp,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "batchnorm_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "batchnorm_fwd_tmpl.h"
  }
}

std::vector<at::Tensor> batchnorm_fwd(
    bool  training,
    bool  relu,
    bool  eltwise,
    float eps,
    std::vector<long> padding,
    std::vector<at::Tensor> inputs) {
  std::string default_string_ncp{"AB"};
  std::string default_string_cp {"A"};
  return batchnorm_fwd_ext(training, relu, eltwise, eps, padding, default_string_ncp, default_string_cp, inputs);
}

std::vector<at::Tensor> batchnorm_bwd_ext(
    bool  relu,
    bool  eltwise,
    float eps,
    std::vector<long> padding,
    std::string tuning_string_ncp,
    std::string tuning_string_cp,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "batchnorm_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "batchnorm_bwd_tmpl.h"
  }
}

std::vector<at::Tensor> batchnorm_bwd(
    bool  relu,
    bool  eltwise,
    float eps,
    std::vector<long> padding,
    std::vector<at::Tensor> inputs) {
  std::string default_string_ncp{"AB"};
  std::string default_string_cp {"A"};
  return batchnorm_bwd_ext(relu, eltwise, eps, padding, default_string_ncp, default_string_cp, inputs);
}

#define CHANNEL_BLOCK_SIZE 64

int batchnorm_get_c_block( int C /*, datatype as an int flag? */ )
{
  libxsmm_blasint bc = CHANNEL_BLOCK_SIZE; /* hardcoded for now */

  if (C % bc != 0)
    bc = C;

  return bc;
}

double batchnorm_fwd_get_gflop(int N, int C, int H, int W)
{
  double gflop = 0.0;
  /* gflop count for batchnorms */
  double coeff_batchnorm = 7.0; /* 3 for stats + 4 for scaling */
  gflop += coeff_batchnorm * (double)N * (double)C * (double)H * (double)W / (1000*1000*1000);

  return gflop;
}


REGISTER_SUBMODULE(_batchnorm, m) {
  m.def(
      "batchnorm_fwd",
      &batchnorm_fwd,
      "Pcl BN forward");
  m.def(
      "batchnorm_bwd",
      &batchnorm_bwd,
      "Pcl BN backward");
  m.def(
      "batchnorm_get_c_block",
      &batchnorm_get_c_block,
      "Pcl BN get_c_block");
  m.def("batchnorm_fwd_ext", &batchnorm_fwd_ext, "Pcl BN forward with tuning parameters (strings)");
  m.def("batchnorm_bwd_ext", &batchnorm_bwd_ext, "Pcl BN backward with tuning parameters (strings)");
  m.def("batchnorm_fwd_get_gflop", &batchnorm_fwd_get_gflop, "Pcl BN forward get gflop count (7NCHW)");
}


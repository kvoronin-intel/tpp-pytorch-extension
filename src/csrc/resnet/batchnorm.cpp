
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

std::vector<at::Tensor> batchnorm_fwd(
    bool  training,
    bool  relu,
    bool  eltwise,
    float eps,
    std::vector<long> padding,
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

std::vector<at::Tensor> batchnorm_bwd(
    bool  relu,
    bool  eltwise,
    float eps,
    std::vector<long> padding,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[1].dtype() == at::kFloat) {
    typedef float T;
#include "batchnorm_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "batchnorm_bwd_tmpl.h"
  }
}

#define CHANNEL_BLOCK_SIZE 64

int batchnorm_get_c_block( int C /*, datatype as an int flag? */ )
{
  libxsmm_blasint bc = CHANNEL_BLOCK_SIZE; /* hardcoded for now */

  if (C % bc != 0)
    bc = C;

  return bc;
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
}



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

REGISTER_SCOPE(bn_fwd_reduce, "bn_fwd_reduce");
REGISTER_SCOPE(bn_fwd_stats,  "bn_fwd_stats");
REGISTER_SCOPE(bn_fwd_scale,  "bn_fwd_scale");

REGISTER_SCOPE(bn_bwd_w,      "bn_bwd_w");

/*
REGISTER_SCOPE(b_emb, "b_emb");
REGISTER_SCOPE(q_gemm, "q_gemm");
REGISTER_SCOPE(k_gemm, "k_gemm");
REGISTER_SCOPE(v_gemm, "v_gemm");
REGISTER_SCOPE(a_gemm, "a_gemm");
REGISTER_SCOPE(c_gemm, "c_gemm");
REGISTER_SCOPE(o_gemm, "o_gemm");
REGISTER_SCOPE(i_gemm, "i_gemm");

REGISTER_SCOPE(db_emb, "db_emb");
REGISTER_SCOPE(diq_gemm, "diq_gemm");
REGISTER_SCOPE(dik_gemm, "dik_gemm");
REGISTER_SCOPE(div_gemm, "div_gemm");
REGISTER_SCOPE(dica_gemm, "dica_gemm");
REGISTER_SCOPE(dii_gemm, "dii_gemm");
REGISTER_SCOPE(dio_gemm, "dio_gemm");
REGISTER_SCOPE(dwqkv_gemm, "dwqkv_gemm");
REGISTER_SCOPE(dwq_gemm, "dwq_gemm");
REGISTER_SCOPE(dwk_gemm, "dwk_gemm");
REGISTER_SCOPE(dwv_gemm, "dwv_gemm");
REGISTER_SCOPE(dwa_gemm, "dwa_gemm");
REGISTER_SCOPE(dwc_gemm, "dwc_gemm");
REGISTER_SCOPE(dwi_gemm, "dwi_gemm");
REGISTER_SCOPE(dwo_gemm, "dwo_gemm");
REGISTER_SCOPE(dqkv_bias, "dqkv_bias");
REGISTER_SCOPE(di_bias, "di_bias");
REGISTER_SCOPE(do_bias, "do_bias");
*/

/*
template <typename T>
inline void omp_reduce_buf(
    int num_threads,
    int N,
    float** ptrs,
    T* buf,
    bool accumulate = false) {
  ScopedTimer _t(EW_RED);
#pragma omp for
  for (int i = 0; i < N; i++) {
    float sum = 0.0;
    for (int j = 0; j < num_threads; j++) {
      sum += ptrs[j][i];
    }
    if (accumulate) {
      buf[i] += sum;
    } else {
      buf[i] = sum;
    }
  }
}
*/

std::vector<at::Tensor> batchnorm_fwd(
    bool  training,
    bool  relu,
    bool  eltwise,
    float eps,
    std::vector<long> padding,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[1].dtype() == at::kFloat) {
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

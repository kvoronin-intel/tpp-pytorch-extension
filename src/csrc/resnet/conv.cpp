
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
#   warning "Building conv with threaded loops instead of OpenMP pragmas"
//#   error   "Building conv with threaded loops instead of OpenMP pragmas is not supported yet"
#   include "threaded_loops.h"
#endif

REGISTER_SCOPE(conv_fwd,     "conv_fwd");

REGISTER_SCOPE(conv_bwd_upd, "conv_bwd_upd");

/* Has the conv_config and all setters */
#include "conv_setup.h"

std::vector<at::Tensor> conv_fwd(
    conv_config cfg,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[1].dtype() == at::kFloat) {
    typedef float T;
#include "conv_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "conv_fwd_tmpl.h"
  }
}

std::vector<at::Tensor> conv_bwd(
    conv_config cfg,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[1].dtype() == at::kFloat) {
    typedef float T;
#include "conv_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "conv_bwd_tmpl.h"
  }
}

#define C_BLOCK_SIZE (64) /* hardcoded for now, used in conv_setup() */

#define K_BLOCK_SIZE (64) /* hardcoded for now, used in conv_setup() */

conv_config conv_setup(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint K, libxsmm_blasint R, libxsmm_blasint S,
                              libxsmm_blasint pad_h, libxsmm_blasint pad_w, libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                              libxsmm_blasint stride, int dtype_int )
{
  conv_config res;

  libxsmm_blasint bc, bk;
//  if (C % 64 == 0)
    bc = C_BLOCK_SIZE; /* hardcoded for now */
//  else
//    bc = C;
//  if (K % 64 == 0)
    bk = K_BLOCK_SIZE;
//  else
//    bk = K;
  //libxsmm_blasint bk      = K_BLOCK_SIZE;       /* hardcoded for now */
  libxsmm_blasint threads = (libxsmm_blasint)omp_get_max_threads();

  /*printf("debug: calling conv_setup_new with tensor N H W C K R S padding stride bc bk: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                                                                                          N, H, W, C, K, R, S,
                                                                                          pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out, stride, bc, bk);*/

  libxsmm_dnn_conv_eltwise_fuse fuse_type = LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE; /* FIXME: to be changed later? */

  libxsmm_datatype cnn_dtype_in  = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
  libxsmm_datatype cnn_dtype_out = cnn_dtype_in;

  //res.dtype = dtype_int;

  libxsmm_blasint overwrite_output    = 1; /* hardcoded for now */
  libxsmm_blasint avoid_bwd_wt_trans  = 0; /* hardcoded for now */
  libxsmm_blasint zero_fwd_output_rim = 0; /* hardcoded for now */

  /* memset( &res,  0, sizeof(res)); */

  res = setup_conv_config(cnn_dtype_in, cnn_dtype_out, N, H, W, C, K, R, S, stride, stride, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out, bc, bk, threads,
                          fuse_type, overwrite_output, avoid_bwd_wt_trans, zero_fwd_output_rim);

#if 0
  /* allocate and bind scratch */
  void *scratch = NULL;
  if ( (res.cnn_cfg.scratch_size) > 0 ) {
    size_t alloc_size = res.cnn_cfg.scratch_size;
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    //init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
    zero_buf((float*)scratch, (alloc_size)/4);
  }

  res.scratch = scratch;
#endif

  return res;
}

#define HARDCODED_BC (64)
#define HARDCODED_BK (64)

/* Returns a vector of size 3: {C_block, K_block, lp_block} */
std::vector<int> conv_get_feature_map_blocks( int C, int K, int dtype_int )
{
    std::vector<int> res;
    int C_block, K_block, fm_lp_block;

    libxsmm_datatype dtype_in  = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
    libxsmm_datatype dtype_out = dtype_in;

    libxsmm_dnn_conv_get_feature_map_blocks( C, K, &C_block, &K_block, &fm_lp_block, dtype_in, dtype_out, HARDCODED_BC, HARDCODED_BK );

    res.push_back(C_block);
    res.push_back(K_block);
    res.push_back(fm_lp_block);

    return res;
}

REGISTER_SUBMODULE(_conv, m) {
  m.def(
      "conv_fwd",
      &conv_fwd,
      "Pcl CONV forward");
  m.def(
      "conv_bwd",
      &conv_bwd,
      "Pcl CONV backward");
  m.def(
      "conv_get_feature_map_blocks",
      &conv_get_feature_map_blocks,
      "Pcl CONV get_feature_map_blocks");
  py::class_<conv_config>(m, "conv_config")
  .def(py::init<>());
  //.def_readwrite("initialized", &conv_config::initialized);
  m.def("conv_setup", &conv_setup, "Pcl CONV setup (params)");

/*
  .def_readwrite("pad_h",       &conv_params::pad_h)
  .def_readwrite("pad_w",       &conv_params::pad_w)
  .def_readwrite("pad_h_in",    &conv_params::pad_h_in)
  .def_readwrite("pad_w_in",    &conv_params::pad_w_in)
  .def_readwrite("pad_h_out",   &conv_params::pad_h_out)
  .def_readwrite("pad_w_out",   &conv_params::pad_w_out)
*/

}



#include <ATen/record_function.h>
//#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "tensor_helper.h"

static int my_rank = guess_mpi_rank();

// used for debugging
static int counter = 0;

#define THREADED_LOOPS

#ifdef THREADED_LOOPS
//#   warning "Building conv with threaded loops instead of OpenMP pragmas"
#   include "threaded_loops.h"
#endif

REGISTER_SCOPE(conv_fwd,     "conv_fwd");

REGISTER_SCOPE(conv_bwd_upd, "conv_bwd_upd");

REGISTER_SCOPE(conv_bwd_d,   "conv_bwd_d");

REGISTER_SCOPE(fusedbtlnk_conv_nobatchnorm_fwd, "fusedbtlnk_conv_nobatchnorm_fwd");

/* Has the conv_config and all setters */
#include "conv_setup_full.h"

at::Tensor conv_fwd_ext(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs,
    std::vector<int> tuning_params,
    std::string tuning_string,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(FWD);

  const int h_block = tuning_params[0];
  const int w_block = tuning_params[1];
  const int c_block = tuning_params[2];
  const int k_block = tuning_params[3];
  const int h_in_gemm = tuning_params[4];
        int pack_input = tuning_params[5];
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "conv_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "conv_fwd_tmpl.h"
  }
}

at::Tensor conv_fwd_preallocated_output_ext(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs,
    std::vector<int> tuning_params,
    std::string tuning_string,
    pybind11::array_t<float>& tuning_timings,
    at::Tensor t_O) {
  GlobalPass _gp(FWD);

  const int h_block = tuning_params[0];
  const int w_block = tuning_params[1];
  const int c_block = tuning_params[2];
  const int k_block = tuning_params[3];
  const int h_in_gemm = tuning_params[4];
        int pack_input = tuning_params[5];
#define TIMING
#define PREALLOCATED_OUTPUT
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "conv_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "conv_fwd_tmpl.h"
  }
#undef PREALLOCATED_OUTPUT
#undef TIMING
}

at::Tensor conv_fwd_as_fused_ext(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs,
    std::vector<int> tuning_params,
    std::string tuning_string,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(FWD);
  at::Tensor conv1_out;
//for (int i = 0; i < 1000; i++) {
#define NO_BATCHNORM
#define TIMING
  const int h_block = tuning_params[0];
  const int w_block = tuning_params[1];
  const int c_block = tuning_params[2];
  const int k_block = tuning_params[3];
  const int h_in_gemm = tuning_params[4];
        int pack_input = tuning_params[5];


  auto t_CI  = inputs[0];//input;
  auto t_CW  = inputs[1];//conv1_weight;

  auto conv_cfg = cfg;//cfg.conv1;
  #define CONV_OUT          conv1_out

  auto fuse_scaling = 0; /* fusion of scaling for the previous batchnorm into the conv */
  auto fuse_stats = 0;
  auto conv_loop_string = tuning_string;

  double t_start, t_conv_start, t_conv_end, /*t_bn_stats_end, t_bn_end,*/ t_end;

  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_conv_bn_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_conv_bn_fwd.h"
  }

  #undef CONV_OUT
//#ifdef TIMING
//  auto time_c1        = t_conv_end - t_conv_start;
//#endif

#ifdef TIMING
  auto buf = tuning_timings.request();
  float* ptr = (float*)buf.ptr;
  ptr[0] += t_end - t_conv_start;
  ptr[1] += t_end - t_start;
//  printf("updating timings here in conv fwd\n");
#endif

#undef TIMING
#undef NO_BATCHNORM
//}
  return conv1_out;
}



at::Tensor conv_fwd(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs) {
  int h_block = 1, w_block = 1, c_block = 1, k_block = 1;
  int h_in_gemm = 1;
  int pack_input_for_1x1_strided = 0;
  std::vector<int> default_tuning_params{h_block, w_block, c_block, k_block,
                                         h_in_gemm,
                                         pack_input_for_1x1_strided};
  //std::vector<float> default_tuning_timings(0);
  pybind11::array_t<float> default_tuning_timings(16);
  float *ptr = default_tuning_timings.mutable_data();
  for (int i = 0; i < 16; i++)
      ptr[i] = 0.0;
  //char conv_fwd_loop_specs_str[256] = "Abcdefg";
  std::string default_tuning_string{"Abcdefg"};
  return conv_fwd_ext(cfg, inputs, default_tuning_params, default_tuning_string, default_tuning_timings);
}

at::Tensor conv_bwd_d_ext(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs,
    std::vector<int> tuning_params,
    std::string tuning_string,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(BWD);

  const int h_block   = tuning_params[0];
  const int w_block   = tuning_params[1];
  const int c_block   = tuning_params[2];
  const int k_block   = tuning_params[3];
        int h_in_gemm = tuning_params[4];

  if (inputs[1].dtype() == at::kFloat) {
    typedef float T;
#include "conv_bwd_d_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "conv_bwd_d_tmpl.h"
  }
}

at::Tensor conv_bwd_d(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs) {
  int h_block = 1, w_block = 1, c_block = 1, k_block = 1;
  int h_in_gemm = 1;
  std::vector<int> default_tuning_params{h_block, w_block, c_block, k_block,
                                         h_in_gemm};
  pybind11::array_t<float> default_tuning_timings(16);
  float *ptr = default_tuning_timings.mutable_data();
  for (int i = 0; i < 16; i++)
      ptr[i] = 0.0;
  //char conv_fwd_loop_specs_str[256] = "Abcdefg";
  std::string default_tuning_string{"Abcdefg"};
  return conv_bwd_d_ext(cfg, inputs, default_tuning_params, default_tuning_string, default_tuning_timings);
}

at::Tensor conv_bwd_w_ext(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs,
    std::vector<int> tuning_params,
    std::string tuning_string,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(BWD);

  int p_block = tuning_params[0];

  int bf16_use_nchw_format                       = tuning_params[1];
  int pack_input_upfront                         = tuning_params[2];
  int fuse_upd_transposes                        = tuning_params[3];
  int use_f32_wt_reduction_and_external_wt_vnni  = tuning_params[4];
  int bf16_acc_nw                                = tuning_params[5];
  int par_over_h_pixels                          = tuning_params[6];
  int compute_full_wt_output_block               = tuning_params[7];
  int use_hybrid_imgfm_parallelization           = tuning_params[8];
  int n_img_teams                                = tuning_params[9];
  int n_ofm_teams                                = tuning_params[10];

  if (inputs[1].dtype() == at::kFloat) {
    typedef float T;
#include "conv_bwd_w_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "conv_bwd_w_tmpl.h"
  }
}

at::Tensor conv_bwd_w(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs) {

  int p_block = 1;
  int bf16_use_nchw_format                      = -1;
  int pack_input_upfront                        = -1;
  int fuse_upd_transposes                       = -1;
  int par_over_h_pixels                         = -1;
  int use_f32_wt_reduction_and_external_wt_vnni = -1;
  int bf16_acc_nw                               = -1;
  int compute_full_wt_output_block              = -1;
  int use_hybrid_imgfm_parallelization          = -1;
  int n_img_teams                               = -1;
  int n_ofm_teams                               = -1;
  std::vector<int> default_tuning_params{p_block,
                                         bf16_use_nchw_format,
                                         pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                         bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                         use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams};

  pybind11::array_t<float> default_tuning_timings(16);
  float *ptr = default_tuning_timings.mutable_data();
  for (int i = 0; i < 16; i++)
      ptr[i] = 0.0;
  std::string default_tuning_string;
  if (inputs[1].dtype() == at::kFloat)
    default_tuning_string = "Abcdefg";
  else
    default_tuning_string = "Abcdef";
  return conv_bwd_w_ext(cfg, inputs, default_tuning_params, default_tuning_string, default_tuning_timings);
}

std::vector<at::Tensor> conv_bwd_ext(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs,
    std::vector<int> tuning_params_d,
    std::string tuning_string_d,
    pybind11::array_t<float>& tuning_timings_d,
    std::vector<int> tuning_params_w,
    std::string tuning_string_w,
    pybind11::array_t<float>& tuning_timings_w) {
  GlobalPass _gp(BWD);
  if (inputs[1].dtype() == at::kFloat) {
    //typedef float T;
    auto t_grad_weight = conv_bwd_w_ext(cfg, inputs, tuning_params_w, tuning_string_w, tuning_timings_w);
    auto t_grad_input  = conv_bwd_d_ext(cfg, inputs, tuning_params_d, tuning_string_d, tuning_timings_d);
    return std::vector<at::Tensor> {t_grad_input, t_grad_weight};
  } else {
    //typedef bfloat16 T;
    auto t_grad_weight = conv_bwd_w_ext(cfg, inputs, tuning_params_w, tuning_string_w, tuning_timings_w);
    auto t_grad_input  = conv_bwd_d_ext(cfg, inputs, tuning_params_d, tuning_string_d, tuning_timings_d);
    return std::vector<at::Tensor> {t_grad_input, t_grad_weight};
  }
}

std::vector<at::Tensor> conv_bwd(
    conv_config cfg,
    const std::vector<at::Tensor>& inputs) {
  int h_block = 1, w_block = 1, c_block = 1, k_block = 1;
  int h_in_gemm = 1;
  std::vector<int> default_tuning_params_d{h_block, w_block, c_block, k_block,
                                         h_in_gemm};
  pybind11::array_t<float> default_tuning_timings_d(16);
  float *ptr_d = default_tuning_timings_d.mutable_data();
  for (int i = 0; i < 16; i++)
      ptr_d[i] = 0.0;
  std::string default_tuning_string_d{"Abcdefg"};

  int p_block = 1;
  int bf16_use_nchw_format                      = -1;
  int pack_input_upfront                        = -1;
  int fuse_upd_transposes                       = -1;
  int par_over_h_pixels                         = -1;
  int use_f32_wt_reduction_and_external_wt_vnni = -1;
  int bf16_acc_nw                               = -1;
  int compute_full_wt_output_block              = -1;
  int use_hybrid_imgfm_parallelization          = -1;
  int n_img_teams                               = -1;
  int n_ofm_teams                               = -1;
  std::vector<int> default_tuning_params_w{p_block,
                                           bf16_use_nchw_format,
                                           pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                           bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                           use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams};

  pybind11::array_t<float> default_tuning_timings_w(16);
  float *ptr_w = default_tuning_timings_w.mutable_data();
  for (int i = 0; i < 16; i++)
      ptr_w[i] = 0.0;
  std::string default_tuning_string_w;
  if (inputs[1].dtype() == at::kFloat)
    default_tuning_string_w = "Abcdefg";
  else
    default_tuning_string_w = "Abcdef";
  return conv_bwd_ext(cfg, inputs,
                      default_tuning_params_d, default_tuning_string_d, default_tuning_timings_d,
                      default_tuning_params_w, default_tuning_string_w, default_tuning_timings_w);
}

#define C_BLOCK_SIZE (32) /* hardcoded for now, used in conv_setup() */

#define K_BLOCK_SIZE (32) /* hardcoded for now, used in conv_setup() */

conv_config conv_setup(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint K, libxsmm_blasint R, libxsmm_blasint S,
                              libxsmm_blasint pad_h, libxsmm_blasint pad_w, libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                              libxsmm_blasint stride, int fuse_type_int, int dtype_int )
{
  conv_config res;

  libxsmm_blasint bc, bk;
  if (C % C_BLOCK_SIZE == 0)
    bc = C_BLOCK_SIZE; /* hardcoded for now, if not good, call conv_setup_preset instead */
  else
    bc = C;
  if (K % K_BLOCK_SIZE == 0)
    bk = K_BLOCK_SIZE; /* hardcoded for now, if not good, call conv_setup_preset instead */
  else
    bk = K;
  libxsmm_blasint threads = (libxsmm_blasint)omp_get_max_threads();

  /* printf("debug: calling conv_setup_new with tensor N H W C K R S padding stride bc bk: %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
                                                                                          N, H, W, C, K, R, S,
                                                                                          pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out, stride, bc, bk); */

  libxsmm_dnn_conv_eltwise_fuse fuse_type;
  if (fuse_type_int == 0)
    fuse_type = LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE;
  else if (fuse_type_int == 1)
    fuse_type = LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS;
  else {
    printf("Error: unsupported fuse_type_int = %d passed in conv_setup()\n", fuse_type_int);
    exit(-1);
  }

  libxsmm_datatype cnn_dtype_in  = (dtype_int == 0 ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_BF16);
  libxsmm_datatype cnn_dtype_out = cnn_dtype_in;

  //res.dtype = dtype_int;

  libxsmm_blasint overwrite_output    = 1; /* hardcoded for now */
  libxsmm_blasint avoid_bwd_wt_trans  = 0; /* hardcoded for now */
  libxsmm_blasint zero_fwd_output_rim = 0; /* hardcoded for now */

  /* Note: a caveat here is that arguments bc and bk are only used if main cases in libxsmm_dnn_conv_get_feature_map_blocks() are not used so in the majority of cases bc and bk will be ignored */
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

conv_config conv_setup_preset(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint K, libxsmm_blasint R, libxsmm_blasint S,
                              libxsmm_blasint pad_h, libxsmm_blasint pad_w, libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in, libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
                              libxsmm_blasint stride, int fuse_type_int, int dtype_int,
                              libxsmm_blasint bc, libxsmm_blasint bk) //, libxsmm_blasint avoid_fmas_in_rim )
{
  conv_config res = conv_setup(N, C, H, W, K, R, S, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out, stride, fuse_type_int, dtype_int );

  res.bc = bc;
  res.bk = bk;
  res.blocksifm = res.C / res.bc;
  res.blocksofm = res.K / res.bk;
  //res.avoid_fmas_in_rim = avoid_fmas_in_rim_int;

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

double conv_fwd_get_gflop(conv_config cfg)
{
  double gflop_conv = 0;
  //double gflop = (2.0*(double)n_iters*(double)N*(double)C*(double)K*(double)R*(double)S*(double)ofh*(double)ofw)/(1000*1000*1000);

  /* gflop count for convolution fwd */
  gflop_conv = (2.0*(double)cfg.N*(double)cfg.C*(double)cfg.K*(double)cfg.R*(double)cfg.S*(double)cfg.ofh*(double)cfg.ofw)/(1000*1000*1000);

  return gflop_conv;
}

/* Does not count weight transpose at the moment, hence same as fwd */
double conv_bwd_d_get_gflop(conv_config cfg)
{
  return conv_fwd_get_gflop(cfg);
}

/* Loosely uses the ballpark esitmate of fwd */
double conv_bwd_w_get_gflop(conv_config cfg)
{
  return conv_fwd_get_gflop(cfg);
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
  .def(py::init<>())
  .def_readwrite("pad_h",   &conv_config::pad_h);
  //.def_readwrite("initialized", &conv_config::initialized);
  m.def("conv_setup", &conv_setup, "Pcl CONV setup (with internally computed block sizes)");
  m.def("conv_setup_preset", &conv_setup_preset, "Pcl CONV setup (with reset block sizes)");
  m.def("conv_fwd_ext", &conv_fwd_ext, "Pcl CONV forward with extra (tuning) parameters");
  m.def("conv_fwd_get_gflop", &conv_fwd_get_gflop, "Pcl CONV get gflop count for fwd");
  m.def("conv_bwd_w", &conv_bwd_w, "Pcl CONV backward weight update");
  m.def("conv_bwd_w_ext", &conv_bwd_w_ext, "Pcl CONV backward weight update with extra (tuning) parameters");
  m.def("conv_bwd_w_get_gflop", &conv_bwd_w_get_gflop, "Pcl CONV get gflop count for bwd w");
  m.def("conv_bwd_d", &conv_bwd_d, "Pcl CONV backward input update");
  m.def("conv_bwd_d_ext", &conv_bwd_d_ext, "Pcl CONV backward input update with extra (tuning) parameters");
  m.def("conv_bwd_d_get_gflop", &conv_bwd_d_get_gflop, "Pcl CONV get gflop count for bwd d");
  m.def("conv_bwd_ext", &conv_bwd_ext, "Pcl CONV backward with extra (tuning) parameters");
  m.def("conv_fwd_as_fused_ext", &conv_fwd_as_fused_ext, "Pcl CONV forward with extra (tuning) parameters caled through fused conv bn code");
  m.def("conv_fwd_preallocated_output_ext", &conv_fwd_preallocated_output_ext, "Pcl CONV forward with extra (tuning) parameters and preallocated output");
}

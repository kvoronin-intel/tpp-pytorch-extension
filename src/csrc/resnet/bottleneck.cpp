
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

using namespace pcl;
#include "tensor_helper.h"

static int counter = 0;

//#define GLOBAL_SHARED_WEIGHTS


#ifdef GLOBAL_SHARED_WEIGHTS

//REGISTER_SCOPE(fusedbtlnk_global_shared_weights_nobatchnorm_fwd,    "fusedbtlnk_global_shared_weights_nobatchnorm_fwd");

#define DECL_VLA_PTR_PT_SPECIAL(type, name, dims, t) \
  type(*name) dims = (type(*) dims)(t)

#define MB_INT (1024*1024)

#define GLOBAL_SHARED_WEIGHT_NELEMENTS (16 * MB_INT / sizeof(libxsmm_bfloat16))
libxsmm_bfloat16 global_shared_weights[GLOBAL_SHARED_WEIGHT_NELEMENTS];

struct my_array_initializer {
  my_array_initializer() {
    // Initialize the global array here
    for (int i = 0; i < GLOBAL_SHARED_WEIGHT_NELEMENTS; i++) {
      float tmp = 0.0;
      libxsmm_convert_bf16_f32( &global_shared_weights[i], &tmp, 1);
    }
  }
};
my_array_initializer dummy_variable;
#endif

// Can be defined in the setup.py
#ifdef WITH_VTUNE
  #warning "Building with WITH_VTUNE enabled"
  #include "ittnotify.h"
#endif

#if 0
#define USE_UNCORE_PERF_COUNTERS
#if 1
#define USE_DRAM_COUNTERS
#endif
#endif

#if 0
#define USE_CORE_PERF_COUNTERS
#endif

#if defined(USE_UNCORE_PERF_COUNTERS) || defined(USE_CORE_PERF_COUNTERS)
#include "perf_counter_markers.h"
#endif


static int my_rank = guess_mpi_rank();

// not implemented (only conv created in the setup but not called)
// it should be in fact batchnorm, potentially with eltwise add (if residual connection is present in the previous bottleneck)
// #define WITH_CACHE_PREHEAT

#define FUSED_BOTTLENECK

#define TIMING

#ifdef FUSED_BOTTLENECK
  #warning "FUSED_BOTTLENECK is enabled"

  #define BITS_PER_CHAR 8

  //REGISTER_SCOPE(conv1_bn1_fwd,     "conv1_bn1_fwd");

  REGISTER_SCOPE(fusedbtlnk_conv_fwd,     "fusedbtlnk_conv_fwd");

  REGISTER_SCOPE(fusedbtlnk_bn_fwd_reduce,   "fusedbtlnk_bn_fwd_reduce");
  REGISTER_SCOPE(fusedbtlnk_bn_fwd_stats,    "fusedbtlnk_bn_fwd_stats");
  REGISTER_SCOPE(fusedbtlnk_bn_fwd_scale,    "fusedbtlnk_bn_fwd_scale");
#endif

#define THREADED_LOOPS

#ifdef THREADED_LOOPS
#   warning "Building bottleneck with threaded loops instead of OpenMP pragmas"
//#   error   "Building conv with threaded loops instead of OpenMP pragmas is not supported yet"
#   include "threaded_loops.h"
#endif

#include "conv_setup_external.h" /* for conv_config and conv_setup declaration */

extern std::vector<at::Tensor> batchnorm_fwd(bool training, bool relu, bool eltwise, float eps, std::vector<long> padding, std::vector<at::Tensor> inputs);
extern std::vector<at::Tensor> batchnorm_bwd(bool relu, bool eltwise, float eps, std::vector<long> padding, std::vector<at::Tensor> inputs);
extern std::vector<at::Tensor> batchnorm_bwd_ext(bool  relu, bool  eltwise, float eps, std::vector<long> padding,
                                                  std::string tuning_string_ncp, std::string tuning_string_cp, std::vector<at::Tensor> inputs);

extern at::Tensor conv_fwd(conv_config cfg, std::vector<at::Tensor> inputs);
extern std::vector<at::Tensor> conv_bwd(conv_config cfg, std::vector<at::Tensor> inputs);

extern std::vector<at::Tensor> conv_bwd_ext(conv_config cfg, std::vector<at::Tensor> inputs,
                                            std::vector<int> tuning_params_d, std::string tuning_string_d, pybind11::array_t<float>& tuning_timings_d,
                                            std::vector<int> tuning_params_w, std::string tuning_string_w, pybind11::array_t<float>& tuning_timings_w);
extern at::Tensor conv_bwd_d_ext(conv_config cfg, std::vector<at::Tensor> inputs,
                                 std::vector<int> tuning_params, std::string tuning_string, pybind11::array_t<float>& tuning_timings);
extern at::Tensor conv_bwd_w_ext(conv_config cfg, std::vector<at::Tensor> inputs,
                                 std::vector<int> tuning_params, std::string tuning_string, pybind11::array_t<float>& tuning_timings);

std::array<std::string, 2> parse_conv_loop_string_for_batchnorm(const char *conv_loop_specs, int conv_is_nckhwrs, int use_nchw_format);

typedef struct bottleneck_bn_config {
  libxsmm_blasint N;
  libxsmm_blasint inplanes;
  libxsmm_blasint H;
  libxsmm_blasint W;
  libxsmm_blasint planes;
  libxsmm_blasint stride;
  float bn_eps;
  float bn_momentum;
  libxsmm_blasint bn_track_running_stats_int;
  libxsmm_blasint expansion;
  libxsmm_blasint padding_3x3_type; /* 0 for logical, 1 for physical */
  libxsmm_blasint pad_size;         /* size of physical padding */
  libxsmm_blasint dtype_int;

  libxsmm_blasint has_residual_conv;

  libxsmm_blasint conv1_kernel_size;
  libxsmm_blasint conv1_stride     ;
  libxsmm_blasint conv1_padding    ;

  libxsmm_blasint conv2_kernel_size;
  libxsmm_blasint conv2_stride     ;
  libxsmm_blasint conv2_padding    ;

  libxsmm_blasint conv3_kernel_size;
  libxsmm_blasint conv3_stride     ;
  libxsmm_blasint conv3_padding    ;

  libxsmm_blasint conv4_kernel_size;
  libxsmm_blasint conv4_stride     ;
  libxsmm_blasint conv4_padding    ;

  libxsmm_blasint bn1_fuse_type;
  libxsmm_blasint bn2_fuse_type;
  libxsmm_blasint bn3_fuse_type;
  libxsmm_blasint bn4_fuse_type;

  conv_config     conv1;
  conv_config     conv2;
  conv_config     conv3;
  conv_config     conv4; /* (optionally used) */
#ifdef WITH_CACHE_PREHEAT
  conv_config     conv_preheat; /* (optionally used) */
#endif

  /* Note: batchnorms do not have configs in the extension code */
} bottleneck_bn_config;

std::vector<at::Tensor> bottleneck_bn_fwd_ext(
    bottleneck_bn_config cfg,
    bool training,
    std::vector<at::Tensor> inputs,
    std::vector<int> tuning_params,
    std::vector<std::string> tuning_strings,
    pybind11::array_t<float>& tuning_timings) {
//#ifdef GLOBAL_SHARED_WEIGHTS
//  #define NO_BATCHNORM
//#endif
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_fwd_tmpl.h"
#else
#   include "bottleneck_fwd_tmpl.h"
#endif
  } else {
    typedef bfloat16 T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_fwd_tmpl.h"
#else
#   include "bottleneck_fwd_tmpl.h"
#endif
  }

//#ifdef GLOBAL_SHARED_WEIGHTS
//  #undef NO_BATCHNORM
//#endif
}

std::vector<at::Tensor> bottleneck_bn_fwd_ext_study1(
    bottleneck_bn_config cfg,
    bool training,
    std::vector<at::Tensor> inputs,
    std::vector<int> tuning_params,
    std::vector<std::string> tuning_strings,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(FWD);
#define EXT_STUDY1
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_fwd_tmpl.h"
#else
#   include "bottleneck_fwd_tmpl.h"
#endif
  } else {
    typedef bfloat16 T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_fwd_tmpl.h"
#else
#   include "bottleneck_fwd_tmpl.h"
#endif
  }
#undef EXT_STUDY1
}

std::vector<at::Tensor> bottleneck_bn_fwd_ext_study2(
    bottleneck_bn_config cfg,
    bool training,
    std::vector<at::Tensor> inputs,
    std::vector<int> tuning_params,
    std::vector<std::string> tuning_strings,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(FWD);
#define EXT_STUDY2
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_fwd_tmpl.h"
#else
#   include "bottleneck_fwd_tmpl.h"
#endif
  } else {
    typedef bfloat16 T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_fwd_tmpl.h"
#else
#   include "bottleneck_fwd_tmpl.h"
#endif
  }
#undef EXT_STUDY2
}

std::vector<at::Tensor> bottleneck_bn_fwd(
    bottleneck_bn_config cfg,
    bool training,
    std::vector<at::Tensor> inputs) {
  int h1_block = 1, w1_block = 1, h2_block = 1, w2_block = 1, h3_block = 1, w3_block = 1, h4_block = 1, w4_block = 1;
  int c1_block = 1, k1_block = 1, c2_block = 1, k2_block = 1, c3_block = 1, k3_block = 1, c4_block = 1, k4_block = 1;
  int h1_in_gemm = 1, h2_in_gemm = 1, h3_in_gemm = 1, h4_in_gemm = 1;
  int pack_input_for_1x1_strided = 0;
  int fuse_stats        = 1;
  std::vector<int> default_tuning_params{h1_block, w1_block, h2_block, w2_block, h3_block, w3_block, h4_block, w4_block,
                                         c1_block, k1_block, c2_block, k2_block, c3_block, k3_block, c4_block, k4_block,
                                         h1_in_gemm, h2_in_gemm, h3_in_gemm, h4_in_gemm,
                                         pack_input_for_1x1_strided,
                                         fuse_stats};
  pybind11::array_t<float> default_tuning_timings(16);
  float *ptr = default_tuning_timings.mutable_data();
  for (int i = 0; i < 16; i++)
      ptr[i] = 0.0;
  //char conv_fwd_loop_specs_str[256] = "Abcdefg";
  std::vector<std::string> default_tuning_strings{"Abcdefg", "Abcdefg", "Abcdefg", "Abcdefg"};
  return bottleneck_bn_fwd_ext(cfg, training, inputs, default_tuning_params, default_tuning_strings, default_tuning_timings);
}

std::vector<at::Tensor> bottleneck_bn_bwd_w_ext(
    bottleneck_bn_config cfg,
    std::vector<at::Tensor> inputs,
    std::vector<int> tuning_params_w,
    std::vector<std::string> tuning_strings_w,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(BWD);

#define BWD_W_ONLY
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_bwd_tmpl.h"
#else
#   include "bottleneck_bwd_tmpl.h"
#endif
  } else {
    typedef bfloat16 T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_bwd_tmpl.h"
#else
#   include "bottleneck_bwd_tmpl.h"
#endif
  }
#undef BWD_W_ONLY
}


std::vector<at::Tensor> bottleneck_bn_bwd_d_ext(
    bottleneck_bn_config cfg,
    std::vector<at::Tensor> inputs,
    std::vector<int> tuning_params_d,
    std::vector<std::string> tuning_strings_d,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(BWD);

#define BWD_D_ONLY
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_bwd_tmpl.h"
#else
#   include "bottleneck_bwd_tmpl.h"
#endif
  } else {
    typedef bfloat16 T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_bwd_tmpl.h"
#else
#   include "bottleneck_bwd_tmpl.h"
#endif
  }
#undef BWD_D_ONLY
}

std::vector<at::Tensor> bottleneck_bn_bwd_ext(
    bottleneck_bn_config cfg,
    std::vector<at::Tensor> inputs,
    std::vector<int> tuning_params_d,
    std::vector<std::string> tuning_strings_d,
    std::vector<int> tuning_params_w,
    std::vector<std::string> tuning_strings_w,
    pybind11::array_t<float>& tuning_timings) {
  GlobalPass _gp(BWD);

  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_bwd_tmpl.h"
#else
#   include "bottleneck_bwd_tmpl.h"
#endif
  } else {
    typedef bfloat16 T;
#ifdef FUSED_BOTTLENECK
#   include "fused_bottleneck_bwd_tmpl.h"
#else
#   include "bottleneck_bwd_tmpl.h"
#endif
  }
}

std::vector<at::Tensor> bottleneck_bn_bwd_defaultd_ext(
    bottleneck_bn_config cfg,
    std::vector<at::Tensor> inputs,
    std::vector<int> tuning_params_w,
    std::vector<std::string> tuning_strings_w,
    pybind11::array_t<float>& tuning_timings) {
  int h1_block = 1, w1_block = 1, h2_block = 1, w2_block = 1, h3_block = 1, w3_block = 1, h4_block = 1, w4_block = 1;
  int c1_block = 1, k1_block = 1, c2_block = 1, k2_block = 1, c3_block = 1, k3_block = 1, c4_block = 1, k4_block = 1;
  int h1_in_gemm = 1, h2_in_gemm = 1, h3_in_gemm = 1, h4_in_gemm = 1;
  std::vector<int> default_tuning_params_d{h1_block, w1_block, h2_block, w2_block, h3_block, w3_block, h4_block, w4_block,
                                         c1_block, k1_block, c2_block, k2_block, c3_block, k3_block, c4_block, k4_block,
                                         h1_in_gemm, h2_in_gemm, h3_in_gemm, h4_in_gemm};
  //char conv_fwd_loop_specs_str[256] = "Abcdefg";
  std::vector<std::string> default_tuning_strings_d{"Abcdefg", "Abcdefg", "Abcdefg", "Abcdefg"};

  return bottleneck_bn_bwd_ext(cfg, inputs, default_tuning_params_d, default_tuning_strings_d, tuning_params_w, tuning_strings_w, tuning_timings);
}

std::vector<at::Tensor> bottleneck_bn_bwd_defaultw_ext(
    bottleneck_bn_config cfg,
    std::vector<at::Tensor> inputs,
    std::vector<int> tuning_params_d,
    std::vector<std::string> tuning_strings_d,
    pybind11::array_t<float>& tuning_timings) {
  int c1_pblock = 1, c2_pblock = 1, c3_pblock = 1, c4_pblock = 1;
  int c1_use_nchw_format = -1, c2_use_nchw_format = -1, c3_use_nchw_format = -1, c4_use_nchw_format = -1;
  int c1_pack_input_upfront                        = -1;
  int c1_fuse_upd_transposes                       = -1;
  int c1_par_over_h_pixels                         = -1;
  int c1_use_intermediate_f32_wt_tensor            = -1;
  int c1_use_f32_wt_reduction_and_external_wt_vnni = -1;
  int c1_bf16_acc_nw                               = -1;
  int c1_compute_full_wt_output_block              = -1;
  int c1_use_hybrid_imgfm_parallelization          = -1;
  int c1_n_img_teams                               = -1;
  int c1_n_ofm_teams                               = -1;
  int c2_pack_input_upfront                        = -1;
  int c2_fuse_upd_transposes                       = -1;
  int c2_par_over_h_pixels                         = -1;
  int c2_use_intermediate_f32_wt_tensor            = -1;
  int c2_use_f32_wt_reduction_and_external_wt_vnni = -1;
  int c2_bf16_acc_nw                               = -1;
  int c2_compute_full_wt_output_block              = -1;
  int c2_use_hybrid_imgfm_parallelization          = -1;
  int c2_n_img_teams                               = -1;
  int c2_n_ofm_teams                               = -1;
  int c3_pack_input_upfront                        = -1;
  int c3_fuse_upd_transposes                       = -1;
  int c3_par_over_h_pixels                         = -1;
  int c3_use_intermediate_f32_wt_tensor            = -1;
  int c3_use_f32_wt_reduction_and_external_wt_vnni = -1;
  int c3_bf16_acc_nw                               = -1;
  int c3_compute_full_wt_output_block              = -1;
  int c3_use_hybrid_imgfm_parallelization          = -1;
  int c3_n_img_teams                               = -1;
  int c3_n_ofm_teams                               = -1;
  int c4_pack_input_upfront                        = -1;
  int c4_fuse_upd_transposes                       = -1;
  int c4_par_over_h_pixels                         = -1;
  int c4_use_intermediate_f32_wt_tensor            = -1;
  int c4_use_f32_wt_reduction_and_external_wt_vnni = -1;
  int c4_bf16_acc_nw                               = -1;
  int c4_compute_full_wt_output_block              = -1;
  int c4_use_hybrid_imgfm_parallelization          = -1;
  int c4_n_img_teams                               = -1;
  int c4_n_ofm_teams                               = -1;
  std::vector<int> default_tuning_params_w{c1_use_nchw_format, c1_fuse_upd_transposes, c1_bf16_acc_nw, c1_par_over_h_pixels, c1_pack_input_upfront, c1_use_intermediate_f32_wt_tensor,
                                           c1_use_hybrid_imgfm_parallelization, c1_n_img_teams, c1_n_ofm_teams, c1_use_f32_wt_reduction_and_external_wt_vnni, c1_compute_full_wt_output_block, c1_pblock,
                                           c2_use_nchw_format, c2_fuse_upd_transposes, c2_bf16_acc_nw, c2_par_over_h_pixels, c2_pack_input_upfront, c2_use_intermediate_f32_wt_tensor,
                                           c2_use_hybrid_imgfm_parallelization, c2_n_img_teams, c2_n_ofm_teams, c2_use_f32_wt_reduction_and_external_wt_vnni, c2_compute_full_wt_output_block, c2_pblock,
                                           c3_use_nchw_format, c3_fuse_upd_transposes, c3_bf16_acc_nw, c3_par_over_h_pixels, c3_pack_input_upfront, c3_use_intermediate_f32_wt_tensor,
                                           c3_use_hybrid_imgfm_parallelization, c3_n_img_teams, c3_n_ofm_teams, c3_use_f32_wt_reduction_and_external_wt_vnni, c3_compute_full_wt_output_block, c3_pblock,
                                           c4_use_nchw_format, c4_fuse_upd_transposes, c4_bf16_acc_nw, c4_par_over_h_pixels, c4_pack_input_upfront, c4_use_intermediate_f32_wt_tensor,
                                           c4_use_hybrid_imgfm_parallelization, c4_n_img_teams, c4_n_ofm_teams, c4_use_f32_wt_reduction_and_external_wt_vnni, c4_compute_full_wt_output_block, c4_pblock};

  std::vector<std::string> default_tuning_strings_w{"Abcdef", "Abcdef", "Abcdef", "Abcdef"};

  return bottleneck_bn_bwd_ext(cfg, inputs, tuning_params_d, tuning_strings_d, default_tuning_params_w, default_tuning_strings_w, tuning_timings);
}

std::vector<at::Tensor> bottleneck_bn_bwd(
    bottleneck_bn_config cfg,
    std::vector<at::Tensor> inputs) {
  int h1_block = 1, w1_block = 1, h2_block = 1, w2_block = 1, h3_block = 1, w3_block = 1, h4_block = 1, w4_block = 1;
  int c1_block = 1, k1_block = 1, c2_block = 1, k2_block = 1, c3_block = 1, k3_block = 1, c4_block = 1, k4_block = 1;
  int h1_in_gemm = 1, h2_in_gemm = 1, h3_in_gemm = 1, h4_in_gemm = 1;
  std::vector<int> default_tuning_params_d{h1_block, w1_block, h2_block, w2_block, h3_block, w3_block, h4_block, w4_block,
                                         c1_block, k1_block, c2_block, k2_block, c3_block, k3_block, c4_block, k4_block,
                                         h1_in_gemm, h2_in_gemm, h3_in_gemm, h4_in_gemm};
  //char conv_fwd_loop_specs_str[256] = "Abcdefg";
  std::vector<std::string> default_tuning_strings_d{"Abcdefg", "Abcdefg", "Abcdefg", "Abcdefg"};

  int c1_pblock = 1, c2_pblock = 1, c3_pblock = 1, c4_pblock = 1;
  int c1_use_nchw_format = -1, c2_use_nchw_format = -1, c3_use_nchw_format = -1, c4_use_nchw_format = -1;
  int c1_pack_input_upfront                        = -1;
  int c1_fuse_upd_transposes                       = -1;
  int c1_par_over_h_pixels                         = -1;
  int c1_use_intermediate_f32_wt_tensor            = -1;
  int c1_use_f32_wt_reduction_and_external_wt_vnni = -1;
  int c1_bf16_acc_nw                               = -1;
  int c1_compute_full_wt_output_block              = -1;
  int c1_use_hybrid_imgfm_parallelization          = -1;
  int c1_n_img_teams                               = -1;
  int c1_n_ofm_teams                               = -1;
  int c2_pack_input_upfront                        = -1;
  int c2_fuse_upd_transposes                       = -1;
  int c2_par_over_h_pixels                         = -1;
  int c2_use_intermediate_f32_wt_tensor            = -1;
  int c2_use_f32_wt_reduction_and_external_wt_vnni = -1;
  int c2_bf16_acc_nw                               = -1;
  int c2_compute_full_wt_output_block              = -1;
  int c2_use_hybrid_imgfm_parallelization          = -1;
  int c2_n_img_teams                               = -1;
  int c2_n_ofm_teams                               = -1;
  int c3_pack_input_upfront                        = -1;
  int c3_fuse_upd_transposes                       = -1;
  int c3_par_over_h_pixels                         = -1;
  int c3_use_intermediate_f32_wt_tensor            = -1;
  int c3_use_f32_wt_reduction_and_external_wt_vnni = -1;
  int c3_bf16_acc_nw                               = -1;
  int c3_compute_full_wt_output_block              = -1;
  int c3_use_hybrid_imgfm_parallelization          = -1;
  int c3_n_img_teams                               = -1;
  int c3_n_ofm_teams                               = -1;
  int c4_pack_input_upfront                        = -1;
  int c4_fuse_upd_transposes                       = -1;
  int c4_par_over_h_pixels                         = -1;
  int c4_use_intermediate_f32_wt_tensor            = -1;
  int c4_use_f32_wt_reduction_and_external_wt_vnni = -1;
  int c4_bf16_acc_nw                               = -1;
  int c4_compute_full_wt_output_block              = -1;
  int c4_use_hybrid_imgfm_parallelization          = -1;
  int c4_n_img_teams                               = -1;
  int c4_n_ofm_teams                               = -1;
  std::vector<int> default_tuning_params_w{c1_use_nchw_format, c1_fuse_upd_transposes, c1_bf16_acc_nw, c1_par_over_h_pixels, c1_pack_input_upfront, c1_use_intermediate_f32_wt_tensor,
                                           c1_use_hybrid_imgfm_parallelization, c1_n_img_teams, c1_n_ofm_teams, c1_use_f32_wt_reduction_and_external_wt_vnni, c1_compute_full_wt_output_block, c1_pblock,
                                           c2_use_nchw_format, c2_fuse_upd_transposes, c2_bf16_acc_nw, c2_par_over_h_pixels, c2_pack_input_upfront, c2_use_intermediate_f32_wt_tensor,
                                           c2_use_hybrid_imgfm_parallelization, c2_n_img_teams, c2_n_ofm_teams, c2_use_f32_wt_reduction_and_external_wt_vnni, c2_compute_full_wt_output_block, c2_pblock,
                                           c3_use_nchw_format, c3_fuse_upd_transposes, c3_bf16_acc_nw, c3_par_over_h_pixels, c3_pack_input_upfront, c3_use_intermediate_f32_wt_tensor,
                                           c3_use_hybrid_imgfm_parallelization, c3_n_img_teams, c3_n_ofm_teams, c3_use_f32_wt_reduction_and_external_wt_vnni, c3_compute_full_wt_output_block, c3_pblock,
                                           c4_use_nchw_format, c4_fuse_upd_transposes, c4_bf16_acc_nw, c4_par_over_h_pixels, c4_pack_input_upfront, c4_use_intermediate_f32_wt_tensor,
                                           c4_use_hybrid_imgfm_parallelization, c4_n_img_teams, c4_n_ofm_teams, c4_use_f32_wt_reduction_and_external_wt_vnni, c4_compute_full_wt_output_block, c4_pblock};
  std::vector<std::string> default_tuning_strings_w{"Aefbcd", "Aefbcd", "Aefbcd", "Aefbcd"};

  pybind11::array_t<float> default_tuning_timings(16);
  float *ptr = default_tuning_timings.mutable_data();
  for (int i = 0; i < 16; i++)
      ptr[i] = 0.0;

  return bottleneck_bn_bwd_ext(cfg, inputs, default_tuning_params_d, default_tuning_strings_d, default_tuning_params_w, default_tuning_strings_w, default_tuning_timings);
}

bottleneck_bn_config bottleneck_bn_setup(libxsmm_blasint N, libxsmm_blasint inplanes, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint planes, libxsmm_blasint stride,
                                         float eps, float bn_momentum, libxsmm_blasint bn_track_running_stats_int, libxsmm_blasint expansion,
                                         libxsmm_blasint padding_3x3_type, libxsmm_blasint dtype_int )
{
  bottleneck_bn_config res;

  memset( &res, 0, sizeof(res));

  res.N = N;
  res.inplanes = inplanes;
  res.H = H;
  res.W = W;
  res.planes = planes;
  res.stride = stride;
  res.bn_eps = eps;
  res.bn_momentum = bn_momentum;
  res.bn_track_running_stats_int = bn_track_running_stats_int;
  res.expansion = expansion;
  res.padding_3x3_type = padding_3x3_type;
  res.dtype_int = dtype_int;

  res.has_residual_conv = ( (res.stride != 1 || res.inplanes != res.planes * expansion) ? 1 : 0);

  res.pad_size = 0;
  if (res.padding_3x3_type == 0)
    res.pad_size = 0;
  else  /* physical padding around 3x3 convolutions */
    res.pad_size = 1;

  res.conv1_kernel_size = 1;
  res.conv1_stride      = 1;
  res.conv1_padding     = 0;
  res.conv1 = conv_setup(res.N, res.inplanes, res.H, res.W, res.planes, res.conv1_kernel_size, res.conv1_kernel_size,
                             res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding, res.conv1_padding,
                             res.conv1_stride, res.dtype_int);

#ifdef WITH_CACHE_PREHEAT
  int res_conv_preheat_kernel_size = 1;
  int res_conv_preheat_stride      = 1;
  int res_conv_preheat_padding     = 0;
  res.conv_preheat = conv_setup(res.N, res.inplanes, res.H, res.W, res.inplanes, res_conv_preheat_kernel_size, res_conv_preheat_kernel_size,
                             res_conv_preheat_padding, res_conv_preheat_padding, res_conv_preheat_padding, res_conv_preheat_padding, res_conv_preheat_padding, res_conv_preheat_padding,
                             res_conv_preheat_stride, res.dtype_int);
#endif

  res.conv2_kernel_size = 3;
  res.conv2_stride      = res.stride;
  res.conv2_padding     = 1;
  if (res.padding_3x3_type == 0)
    res.conv2 = conv_setup(res.N, res.planes, res.H, res.W, res.planes, res.conv2_kernel_size, res.conv2_kernel_size,
                               res.conv2_padding, res.conv2_padding, 0, 0, 0, 0,
                               res.conv2_stride, res.dtype_int);
  else /* physical padding */
    res.conv2 = conv_setup(res.N, res.planes, res.H, res.W, res.planes, res.conv2_kernel_size, res.conv2_kernel_size,
                               res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding, res.conv2_padding,
                               res.conv2_stride, res.dtype_int);

  libxsmm_blasint downsampled_H, downsampled_W;
  if (res.stride != 1) {
    downsampled_H = res.H / res.stride;
    downsampled_W = res.W / res.stride;
  } else {
    downsampled_H = res.H;
    downsampled_W = res.W;
  }

  res.conv3_kernel_size = 1;
  res.conv3_stride      = 1;
  res.conv3_padding     = 0;
  res.conv3 = conv_setup(res.N, res.planes, downsampled_H, downsampled_W, res.planes * res.expansion, res.conv3_kernel_size, res.conv3_kernel_size,
                             res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding, res.conv3_padding,
                             res.conv3_stride, res.dtype_int);
#if 0
  /* optionally output-padded batchnorm before 3x3 conv */
  res.bn1_fuse_type = 4;
  res.bn1   = bnorm_setup_new(res.N, res.planes, res.H, res.W, 0, 0, res.pad_size, res.pad_size, res.bn_eps, res.bn1_fuse_type, res.dtype_int);

  /* optionally input-padded batchnorm after 3x3 conv */
  res.bn2_fuse_type = 4;
  res.bn2   = bnorm_setup_new(res.N, res.planes, downsampled_H, downsampled_W, res.pad_size, res.pad_size, 0, 0, res.bn_eps, res.bn2_fuse_type, res.dtype_int);

  res.bn3_fuse_type = 5;
  res.bn3   = bnorm_setup_new(res.N, res.planes * res.expansion, downsampled_H, downsampled_W, 0, 0, 0, 0, res.bn_eps, res.bn3_fuse_type, res.dtype_int);
#endif

  if (res.has_residual_conv) {
    res.conv4_kernel_size = 1;
    res.conv4_stride      = res.stride;
    res.conv4_padding     = 0;
    res.conv4 = conv_setup(res.N, res.inplanes, res.H, res.W, res.planes * res.expansion, res.conv4_kernel_size, res.conv4_kernel_size,
                               res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding, res.conv4_padding,
                               res.conv4_stride, res.dtype_int);
#if 0
    res.bn4_fuse_type = 0;
    res.bn4   = bnorm_setup_new(res.N, res.planes * res.expansion, downsampled_H, downsampled_W, 0, 0, 0, 0, res.bn_eps, res.bn4_fuse_type, res.dtype_int);
#endif
  }

  return res;
}

bottleneck_bn_config bottleneck_bn_setup_fused_fwd_tuner(libxsmm_blasint N, libxsmm_blasint inplanes, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint planes, libxsmm_blasint stride,
                                                         float eps, float bn_momentum, libxsmm_blasint bn_track_running_stats_int, libxsmm_blasint expansion,
                                                         libxsmm_blasint padding_3x3_type, libxsmm_blasint dtype_int,
                                                         libxsmm_blasint bc_conv1, libxsmm_blasint bc_conv2, libxsmm_blasint bc_conv3, libxsmm_blasint bk_conv3,
                                                         libxsmm_blasint avoid_fmas_in_rim_int  )
{
  bottleneck_bn_config res = bottleneck_bn_setup(N, inplanes, H, W, planes, stride, eps, bn_momentum, bn_track_running_stats_int, expansion,
                                                 padding_3x3_type, dtype_int);

  /* Manually overwriting fields which are needed for auto-tuning. This manipulation is only valid for the fused bottleneck */

  res.conv1.bc = bc_conv1;
  res.conv1.bk = bc_conv2;
  res.conv1.blocksifm = res.conv1.C / res.conv1.bc;
  res.conv1.blocksofm = res.conv1.K / res.conv1.bk;
  res.conv1.avoid_fmas_in_rim = avoid_fmas_in_rim_int;

  res.conv2.bc = bc_conv2;
  res.conv2.bk = bc_conv3;
  res.conv2.blocksifm = res.conv2.C / res.conv2.bc;
  res.conv2.blocksofm = res.conv2.K / res.conv2.bk;
  res.conv2.avoid_fmas_in_rim = avoid_fmas_in_rim_int;

  res.conv3.bc = bc_conv3;
  res.conv3.bk = bk_conv3;
  res.conv3.blocksifm = res.conv3.C / res.conv3.bc;
  res.conv3.blocksofm = res.conv3.K / res.conv3.bk;
  res.conv3.avoid_fmas_in_rim = avoid_fmas_in_rim_int;

  res.conv4.bc = bc_conv1;
  res.conv4.bk = bk_conv3;
  res.conv4.blocksifm = res.conv4.C / res.conv4.bc;
  res.conv4.blocksofm = res.conv4.K / res.conv4.bk;
  res.conv4.avoid_fmas_in_rim = avoid_fmas_in_rim_int;

  return res;
}

double bottleneck_bn_fwd_get_gflop(bottleneck_bn_config cfg)
{
  double gflop_convs = 0, gflop_bns = 0;
  //double gflop = (2.0*(double)n_iters*(double)N*(double)C*(double)K*(double)R*(double)S*(double)ofh*(double)ofw)/(1000*1000*1000);

  /* gflop count for convolutions */
  gflop_convs += (2.0*(double)cfg.conv1.N*(double)cfg.conv1.C*(double)cfg.conv1.K*(double)cfg.conv1.R*(double)cfg.conv1.S*(double)cfg.conv1.ofh*(double)cfg.conv1.ofw)/(1000*1000*1000);
  gflop_convs += (2.0*(double)cfg.conv2.N*(double)cfg.conv2.C*(double)cfg.conv2.K*(double)cfg.conv2.R*(double)cfg.conv2.S*(double)cfg.conv2.ofh*(double)cfg.conv2.ofw)/(1000*1000*1000);
  gflop_convs += (2.0*(double)cfg.conv3.N*(double)cfg.conv3.C*(double)cfg.conv3.K*(double)cfg.conv3.R*(double)cfg.conv3.S*(double)cfg.conv3.ofh*(double)cfg.conv3.ofw)/(1000*1000*1000);
  if (cfg.has_residual_conv)
    gflop_convs += (2.0*(double)cfg.conv4.N*(double)cfg.conv4.C*(double)cfg.conv4.K*(double)cfg.conv4.R*(double)cfg.conv4.S*(double)cfg.conv4.ofh*(double)cfg.conv4.ofw)/(1000*1000*1000);

  /* gflop count for batchnorms */
  double coeff_batchnorm = 7.0; /* 3 for stats + 4 for scaling */
  gflop_bns += coeff_batchnorm * (double)cfg.conv1.N * (double)cfg.conv1.K * (double)cfg.conv1.ofh * (double)cfg.conv1.ofw / (1000*1000*1000);
  gflop_bns += coeff_batchnorm * (double)cfg.conv2.N * (double)cfg.conv2.K * (double)cfg.conv2.ofh * (double)cfg.conv2.ofw / (1000*1000*1000);
  gflop_bns += coeff_batchnorm * (double)cfg.conv3.N * (double)cfg.conv3.K * (double)cfg.conv3.ofh * (double)cfg.conv3.ofw / (1000*1000*1000);
  if (cfg.has_residual_conv)
    gflop_bns += coeff_batchnorm * (double)cfg.conv4.N * (double)cfg.conv4.K * (double)cfg.conv4.ofh * (double)cfg.conv4.ofw / (1000*1000*1000);

  return gflop_convs + gflop_bns;
}

/* Ignores transpose for now */
double bottleneck_bn_bwd_d_get_gflop(bottleneck_bn_config cfg)
{
  std::cout << "Warning: bottleneck_bn_bwd_d_get_gflop ignores transpose for now\n";
  return bottleneck_bn_fwd_get_gflop(cfg);
}

std::vector<float> bottleneck_bn_fwd_get_gflop_details(bottleneck_bn_config cfg)
{
  std::vector<float> gflop_details(16);

  /* gflop counts for convolutions */
  gflop_details[0] = (2.0*(float)cfg.conv1.N*(float)cfg.conv1.C*(float)cfg.conv1.K*(float)cfg.conv1.R*(float)cfg.conv1.S*(float)cfg.conv1.ofh*(float)cfg.conv1.ofw)/(1000*1000*1000);
  gflop_details[1] = (2.0*(float)cfg.conv2.N*(float)cfg.conv2.C*(float)cfg.conv2.K*(float)cfg.conv2.R*(float)cfg.conv2.S*(float)cfg.conv2.ofh*(float)cfg.conv2.ofw)/(1000*1000*1000);
  gflop_details[2] = (2.0*(float)cfg.conv3.N*(float)cfg.conv3.C*(float)cfg.conv3.K*(float)cfg.conv3.R*(float)cfg.conv3.S*(float)cfg.conv3.ofh*(float)cfg.conv3.ofw)/(1000*1000*1000);
  if (cfg.has_residual_conv)
    gflop_details[3] = (2.0*(float)cfg.conv4.N*(float)cfg.conv4.C*(float)cfg.conv4.K*(float)cfg.conv4.R*(float)cfg.conv4.S*(float)cfg.conv4.ofh*(float)cfg.conv4.ofw)/(1000*1000*1000);
  else
    gflop_details[3] = 0.0;

  /* gflop count for batchnorms */
  float coeff_batchnorm = 7.0; /* 3 for stats + 4 for scaling */
  gflop_details[4] = coeff_batchnorm * (float)cfg.conv1.N * (float)cfg.conv1.K * (float)cfg.conv1.ofh * (float)cfg.conv1.ofw / (1000*1000*1000);
  gflop_details[5] = coeff_batchnorm * (float)cfg.conv2.N * (float)cfg.conv2.K * (float)cfg.conv2.ofh * (float)cfg.conv2.ofw / (1000*1000*1000);
  gflop_details[6] = coeff_batchnorm * (float)cfg.conv3.N * (float)cfg.conv3.K * (float)cfg.conv3.ofh * (float)cfg.conv3.ofw / (1000*1000*1000);
  if (cfg.has_residual_conv)
    gflop_details[7] = coeff_batchnorm * (float)cfg.conv4.N * (float)cfg.conv4.K * (float)cfg.conv4.ofh * (float)cfg.conv4.ofw / (1000*1000*1000);
  else
    gflop_details[7] = 0.0;

  return gflop_details;
}

std::vector<float> bottleneck_bn_bwd_d_get_gflop_details(bottleneck_bn_config cfg) {
  std::cout << "Warning: bottleneck_bn_bwd_d_get_gflop_details ignores transpose for now\n";
  return bottleneck_bn_fwd_get_gflop_details(cfg);
}

double bottleneck_bn_bwd_w_get_gflop(bottleneck_bn_config cfg)
{
  std::cout << "Warning: bottleneck_bn_bwd_d_get_gflop ignores transpose for now\n";
  return bottleneck_bn_fwd_get_gflop(cfg);
//  std::cout << "Warning: bottleneck_bn_bwd_w_get_gflop returns a dummy one for now\n";
//  return 1.0;
}

std::vector<float> bottleneck_bn_bwd_w_get_gflop_details(bottleneck_bn_config cfg) {
  std::cout << "Warning: bottleneck_bn_bwd_w_get_gflop_details ignores transpose for now\n";
  return bottleneck_bn_fwd_get_gflop_details(cfg);
/*
  std::cout << "Warning: bottleneck_bn_bwd_w_get_gflop_details returns a vector of dummy ones for now\n";
  std::vector<float> gflop_details(16);
  for (int i = 0; i < 16; i++)
    gflop_details[i] = 1.0;
  return gflop_details;
*/
}

/* conv_is_nckhwrs is 0 for bwd_w and fwd, or 1 for bwd_w or full bwd */
std::array<std::string, 2> parse_conv_loop_string_for_batchnorm(const char *conv_loop_specs, int conv_is_nckhwrs, int use_nchw_format) {
  int A_seen = 0, C_seen = 0;
  for (int i = 0; i < strlen(conv_loop_specs); i++) {
      if(conv_is_nckhwrs) {
        /* For nckhwrs the loop string is as for forward, with A and C for N and K respectively */
        if (conv_loop_specs[i] == 'A')
          A_seen++;
        else if (conv_loop_specs[i] == 'C')
          C_seen++;
      } else { /* nckpixrs (~nchw) or ckhwrs */
        /* For nchw format we check for A(N) and C(K) */
        if (use_nchw_format == 1) {
          if (conv_loop_specs[i] == 'A')
            A_seen++;
          else if (conv_loop_specs[i] == 'C')
            C_seen++;
        } else { /* for chwn format */
          /* For chwn format we check for B(K) */
          if (conv_loop_specs[i] == 'B')
            C_seen++;
        }
      } /* if-else over conv_is_bwd_d */
  }
  std::string nc_loop_stdstr;
  if (A_seen && C_seen)
    nc_loop_stdstr = "AB";
    //strcpy(nkb_loop_specs_str, "AB");
  else if (A_seen && !C_seen)
    nc_loop_stdstr = "Ab";
    //strcpy(nkb_loop_specs_str, "Ab");
  else if (!A_seen && C_seen)
    nc_loop_stdstr = "Ba";
    //strcpy(nkb_loop_specs_str, "Ba");
  else
    nc_loop_stdstr = "ab";
    //strcpy(nkb_loop_specs_str, "ab");

  std::string c_loop_stdstr;
  if (C_seen)
    c_loop_stdstr = "A";
    //strcpy(kb_loop_specs_str, "A");
  else
    c_loop_stdstr = "a";
    //strcpy(kb_loop_specs_str, "a");

  return {nc_loop_stdstr, c_loop_stdstr};
}

void bottleneck_pause_itt() {
#ifdef WITH_VTUNE
  __itt_pause();
#endif
}

void bottleneck_resume_itt() {
#ifdef WITH_VTUNE
  __itt_resume();
#endif
}

/*
void pause_itt() {
#ifdef WITH_VTUNE
  __itt_pause();
#endif
}
*/


REGISTER_SUBMODULE(_bottleneck, m) {
  m.def(
      "bottleneck_bn_fwd",
      &bottleneck_bn_fwd,
      "Pcl BOTTLENECK BN forward");
  m.def(
      "bottleneck_bn_bwd",
      &bottleneck_bn_bwd,
      "Pcl BOTTLENECK BN backward");
  py::class_<bottleneck_bn_config>(m, "bottleneck_bn_config")
  .def(py::init<>())
  .def_readwrite("has_residual_conv", &bottleneck_bn_config::has_residual_conv)
  .def_readwrite("N", &bottleneck_bn_config::N)
  .def_readwrite("inplanes", &bottleneck_bn_config::inplanes)
  .def_readwrite("planes", &bottleneck_bn_config::planes)
  .def_readwrite("H", &bottleneck_bn_config::H)
  .def_readwrite("W", &bottleneck_bn_config::W)
  .def_readwrite("stride", &bottleneck_bn_config::stride);
  m.def("bottleneck_bn_setup", &bottleneck_bn_setup, "Pcl BOTTLENECK BN setup (simple version)");
  m.def("bottleneck_bn_setup_fused_fwd_tuner", &bottleneck_bn_setup_fused_fwd_tuner, "Pcl BOTTLENECK BN setup (custom version for tuning fwd of the fused bottleneck)");
  m.def("bottleneck_bn_fwd_ext", &bottleneck_bn_fwd_ext, "Pcl BOTTLENECK BN forward with tuning params");
  m.def("bottleneck_bn_fwd_get_gflop", &bottleneck_bn_fwd_get_gflop, "Pcl BOTTLENECK BN forward gflop count");
  m.def("bottleneck_bn_fwd_get_gflop_details", &bottleneck_bn_fwd_get_gflop_details, "Pcl BOTTLENECK BN forward gflop counts for various components");
  m.def("bottleneck_bn_fwd_ext_study1", &bottleneck_bn_fwd_ext_study1, "Pcl BOTTLENECK BN forward with tuning params study (with some parts disabled)");
  m.def("bottleneck_bn_fwd_ext_study2", &bottleneck_bn_fwd_ext_study2, "Pcl BOTTLENECK BN forward with tuning params study (with some parts disabled)");
  m.def("bottleneck_bn_bwd_ext", &bottleneck_bn_bwd_ext, "Pcl BOTTLENECK BN backward with tuning params");
  m.def("bottleneck_bn_bwd_defaultd_ext", &bottleneck_bn_bwd_defaultd_ext, "Pcl BOTTLENECK BN backward with tuning params for w and default d");
  m.def("bottleneck_bn_bwd_defaultw_ext", &bottleneck_bn_bwd_defaultw_ext, "Pcl BOTTLENECK BN backward with tuning params for d and default w");
  m.def("bottleneck_bn_bwd_d_ext", &bottleneck_bn_bwd_d_ext, "Pcl BOTTLENECK BN backward over data with tuning params");
  m.def("bottleneck_bn_bwd_d_get_gflop", &bottleneck_bn_bwd_d_get_gflop, "Pcl BOTTLENECK BN bwd_d gflop count");
  m.def("bottleneck_bn_bwd_d_get_gflop_details", &bottleneck_bn_bwd_d_get_gflop_details, "Pcl BOTTLENECK BN bwd_d gflop counts for various components");
  m.def("bottleneck_bn_bwd_w_ext", &bottleneck_bn_bwd_w_ext, "Pcl BOTTLENECK BN backward over weights with tuning params");
  m.def("bottleneck_bn_bwd_w_get_gflop", &bottleneck_bn_bwd_w_get_gflop, "Pcl BOTTLENECK BN bwd_w gflop count");
  m.def("bottleneck_bn_bwd_w_get_gflop_details", &bottleneck_bn_bwd_w_get_gflop_details, "Pcl BOTTLENECK BN bwd_w gflop counts for various components");
  m.def("bottleneck_pause_itt", &bottleneck_pause_itt, "Pcl BOTTLENECK pause itt");
  m.def("bottleneck_resume_itt", &bottleneck_resume_itt, "Pcl BOTTLENECK pause itt");
}



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

#define FUSED_BOTTLENECK

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

extern at::Tensor conv_fwd(conv_config cfg, std::vector<at::Tensor> inputs);
extern std::vector<at::Tensor> conv_bwd(conv_config cfg, std::vector<at::Tensor> inputs);

extern std::vector<at::Tensor> batchnorm_fwd(bool  training, bool  relu, bool  eltwise, float eps, std::vector<long> padding, std::vector<at::Tensor> inputs);
extern std::vector<at::Tensor> batchnorm_bwd(bool  relu, bool  eltwise, float eps, std::vector<long> padding, std::vector<at::Tensor> inputs);

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

  /* Note: batchnorms do not have configs in the extension code */
} bottleneck_bn_config;

std::vector<at::Tensor> bottleneck_bn_fwd(
    bottleneck_bn_config cfg,
    bool training,
    std::vector<at::Tensor> inputs) {
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
}

std::vector<at::Tensor> bottleneck_bn_bwd(
    bottleneck_bn_config cfg,
    std::vector<at::Tensor> inputs) {
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
  .def_readwrite("has_residual_conv", &bottleneck_bn_config::has_residual_conv);
  m.def("bottleneck_bn_setup", &bottleneck_bn_setup, "Pcl BOTTLENECK BN setup (params)");
}


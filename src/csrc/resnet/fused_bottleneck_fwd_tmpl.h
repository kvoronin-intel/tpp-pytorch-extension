RECORD_FUNCTION("fused_bottleneck_bn_fwd", std::vector<c10::IValue>());

//#define VERBOSE

  auto input        = inputs[0];
  auto conv1_weight = inputs[1];
  auto conv2_weight = inputs[2];
  auto conv3_weight = inputs[3];
  auto conv4_weight = inputs[4];
  auto bn1_weight   = inputs[5];
  auto bn2_weight   = inputs[6];
  auto bn3_weight   = inputs[7];
  auto bn4_weight   = inputs[8];
  auto bn1_bias     = inputs[9];
  auto bn2_bias     = inputs[10];
  auto bn3_bias     = inputs[11];
  auto bn4_bias     = inputs[12];
  auto bn1_mean     = inputs[13];
  auto bn2_mean     = inputs[14];
  auto bn3_mean     = inputs[15];
  auto bn4_mean     = inputs[16];
  auto bn1_var      = inputs[17];
  auto bn2_var      = inputs[18];
  auto bn3_var      = inputs[19];
  auto bn4_var      = inputs[20];

  const long h1_block = tuning_params[0];
  const long w1_block = tuning_params[1];
  const long h2_block = tuning_params[2];
  const long w2_block = tuning_params[3];
  const long h3_block = tuning_params[4];
  const long w3_block = tuning_params[5];
  const long h4_block = tuning_params[6];
  const long w4_block = tuning_params[7];
  const long c1_block = tuning_params[8];
  const long k1_block = tuning_params[9];
  const long c2_block = tuning_params[10];
  const long k2_block = tuning_params[11];
  const long c3_block = tuning_params[12];
  const long k3_block = tuning_params[13];
  const long c4_block = tuning_params[14];
  const long k4_block = tuning_params[15];
  const long h1_in_gemm = tuning_params[16];
  const long h2_in_gemm = tuning_params[17];
  const long h3_in_gemm = tuning_params[18];
  const long h4_in_gemm = tuning_params[19];
  const long pack_input_for_1x1_strided = tuning_params[20];
  const long fuse_stats = tuning_params[21];

  const std::string c1_string = tuning_strings[0];
  const std::string c2_string = tuning_strings[1];
  const std::string c3_string = tuning_strings[2];
  const std::string c4_string = tuning_strings[3];

  std::vector<long> dummy_size{0};
  auto dummy_add    = at::zeros(dummy_size, input.options());
  auto dummy_return = at::zeros(dummy_size, input.options());
  auto dummy_stat   = at::zeros(dummy_size, bn1_weight.options());

  auto activation_dtype = input.dtype();

  //char conv_fwd_loop_specs_str[256] = "Abcdefg";

  auto global_fuse_scaling = 0; /* if set to 1, enables fusion of scaling for bn1 and bn2 in the subsequent convolutions conv2 and conv3 */

#ifdef TIMING
  double time_c1 = 0.0, time_b1 = 0.0, time_c2 = 0.0, time_b2 = 0.0, time_c3 = 0.0, time_b3 = 0.0, time_c4 = 0.0, time_b4 = 0.0;
  double time_c1b1 = 0.0, time_c2b2 = 0.0, time_c3b3 = 0.0, time_c4b4 = 0.0;
  double time_b1stats = 0.0, time_b2stats = 0.0, time_b3stats = 0.0, time_b4stats = 0.0;
  double time_c1b1extra = 0.0, time_c2b2extra = 0.0, time_c3b3extra = 0.0, time_c4b4extra = 0.0;
#endif

#ifdef VERBOSE
  printf("running conv1 + bn1\n");

  std::cout << "input dtype = " << input.dtype() << std::endl;
  std::cout << "bn1_weight dtype = " << bn1_weight.dtype() << std::endl;
  std::cout << "bn3_weight dtype = " << bn3_weight.dtype() << std::endl;
#endif

at::Tensor conv1_out, bn1_out, bn1_relu_out, bn1_scratch_out;

//RECORD_SCOPE("conv1_bn1_fwd", std::vector<c10::IValue>());
{
  auto t_CI  = input;
  auto t_CW  = conv1_weight;

  auto conv_cfg = cfg.conv1;
  #define CONV_OUT          conv1_out

  bool relu = true, eltwise = false;
  float eps = cfg.bn_eps;
  std::vector<long> bn_padding{conv_cfg.pad_h_out, conv_cfg.pad_w_out, cfg.pad_size, cfg.pad_size};

  #define BN_OUT            bn1_out
  #define BN_RELU_OUT       bn1_relu_out
  #define BN_SCRATCH_OUT    bn1_scratch_out
  #define BN_IN             CONV_OUT

  auto t_BIA = at::empty({0},  torch::TensorOptions().dtype(activation_dtype));
  auto t_BW  = bn1_weight;
  auto t_BB  = bn1_bias;
  auto t_BM  = bn1_mean;
  auto t_BV  = bn1_var;

  auto fuse_scaling = 0; /* fusion of scaling for the previous batchnorm into the conv */
  auto t_BM_prev  = dummy_stat;
  auto t_BV_prev  = dummy_stat;

  auto h_block = h1_block, w_block = w1_block;
  auto c_block = c1_block, k_block = k1_block;
  auto h_in_gemm = h1_in_gemm;
  auto conv_loop_string = c1_string;
  auto pack_input = 0; /* only could be non-zero for 1x1 strided to make any sense */

  double t_start, t_conv_start, t_conv_end, t_bn_stats_end, t_bn_end, t_end;
  #include "fused_conv_bn_fwd.h"
#ifdef TIMING
  time_c1        = t_conv_end - t_conv_start;
  time_b1        = t_bn_end - t_conv_end;
  time_c1b1      = t_end - t_start;
  time_b1stats   = t_bn_stats_end - t_conv_end;
  time_c1b1extra = (t_end - t_start) - (time_c1 + time_b1);
#endif
}

#ifdef VERBOSE
  printf("running conv2 + bn2\n");
#endif

at::Tensor conv2_out, bn2_out, bn2_relu_out, bn2_scratch_out;

//RECORD_SCOPE("conv2_bn2_fwd", std::vector<c10::IValue>());
{
  auto t_CI  = bn1_out;
  auto t_CW  = conv2_weight;

  auto conv_cfg = cfg.conv2;
  #define CONV_OUT          conv2_out

  bool relu = true, eltwise = false;
  float eps = cfg.bn_eps;
  std::vector<long> bn_padding{ cfg.pad_size, cfg.pad_size, 0, 0};

  #define BN_OUT            bn2_out
  #define BN_RELU_OUT       bn2_relu_out
  #define BN_SCRATCH_OUT    bn2_scratch_out
  #define BN_IN             CONV_OUT

  auto t_BIA = at::empty({0},  torch::TensorOptions().dtype(activation_dtype));
  auto t_BW  = bn2_weight;
  auto t_BB  = bn2_bias;
  auto t_BM  = bn2_mean;
  auto t_BV  = bn2_var;

  auto fuse_scaling = global_fuse_scaling;
  auto t_BM_prev  = bn1_mean;
  auto t_BV_prev  = bn1_var;

  auto h_block = h2_block, w_block = w2_block;
  auto c_block = c2_block, k_block = k2_block;
  auto h_in_gemm = h2_in_gemm;
  auto conv_loop_string = c2_string;
  auto pack_input = 0; /* only could be non-zero for 1x1 strided to make any sense */

  double t_start, t_conv_start, t_conv_end, t_bn_stats_end, t_bn_end, t_end;
  #include "fused_conv_bn_fwd.h"
#ifdef TIMING
  time_c2        = t_conv_end - t_conv_start;
  time_b2        = t_bn_end - t_conv_end;
  time_c2b2      = t_end - t_start;
  time_b2stats   = t_bn_stats_end - t_conv_end;
  time_c2b2extra = (t_end - t_start) - (time_c2 + time_b2);
#endif
}

  at::Tensor conv4_out, residual, bn4_relu_out, bn4_scratch_out;
  if (cfg.has_residual_conv) {
#ifdef VERBOSE
  printf("running conv4 + bn4\n");
#endif

//RECORD_SCOPE("conv4_bn4_fwd", std::vector<c10::IValue>());
{
  auto t_CI  = input;
  auto t_CW  = conv4_weight;

  auto conv_cfg = cfg.conv4;
  #define CONV_OUT          conv4_out

  bool relu = false, eltwise = false;
  float eps = cfg.bn_eps;
  std::vector<long> bn_padding{ 0, 0, 0, 0};

  #define BN_OUT            residual
  #define BN_RELU_OUT       bn4_relu_out
  #define BN_SCRATCH_OUT    bn4_scratch_out
  #define BN_IN             CONV_OUT

  auto t_BIA = at::empty({0},  torch::TensorOptions().dtype(activation_dtype));
  auto t_BW  = bn4_weight;
  auto t_BB  = bn4_bias;
  auto t_BM  = bn4_mean;
  auto t_BV  = bn4_var;

  auto fuse_scaling = 0;
  auto t_BM_prev  = dummy_stat;
  auto t_BV_prev  = dummy_stat;

  auto h_block = h4_block, w_block = w4_block;
  auto c_block = c4_block, k_block = k4_block;
  auto h_in_gemm = h4_in_gemm;
  auto conv_loop_string = c4_string;
  auto pack_input = ( (conv_cfg.u != 1 || conv_cfg.v != 1) ? pack_input_for_1x1_strided : 0); /* only could be non-zero for 1x1 strided to make any sense */

  double t_start, t_conv_start, t_conv_end, t_bn_stats_end, t_bn_end, t_end;
  #include "fused_conv_bn_fwd.h"
#ifdef TIMING
  time_c4        = t_conv_end - t_conv_start;
  time_b4        = t_bn_end - t_conv_end;
  time_c4b4      = t_end - t_start;
  time_b4stats   = t_bn_stats_end - t_conv_end;
  time_c4b4extra = (t_end - t_start) - (time_c4 + time_b4);
#endif
}
  } else {
    conv4_out    = dummy_return;
    residual     = input;
    bn4_relu_out = dummy_return;
    bn4_scratch_out = dummy_return;
  }

#ifdef VERBOSE
  printf("running conv3 + bn3\n");
#endif

at::Tensor conv3_out, bn3_out, bn3_relu_out, bn3_scratch_out;

//RECORD_SCOPE("conv3_bn3_fwd", std::vector<c10::IValue>());
{
  auto t_CI  = bn2_out;
  auto t_CW  = conv3_weight;

  auto conv_cfg = cfg.conv3;
  #define CONV_OUT          conv3_out

  bool relu = true, eltwise = true;
  float eps = cfg.bn_eps;
  std::vector<long> bn_padding{ 0, 0, 0, 0};

  #define BN_OUT            bn3_out
  #define BN_RELU_OUT       bn3_relu_out
  #define BN_SCRATCH_OUT    bn3_scratch_out
  #define BN_IN             CONV_OUT

  auto t_BIA = residual;//at::empty({0},  torch::TensorOptions().dtype(activation_dtype));
  auto t_BW  = bn3_weight;
  auto t_BB  = bn3_bias;
  auto t_BM  = bn3_mean;
  auto t_BV  = bn3_var;

  auto fuse_scaling = global_fuse_scaling; /* fusion of scaling for the previous batchnorm into the conv */
  auto t_BM_prev  = bn2_mean;
  auto t_BV_prev  = bn2_var;

  auto h_block = h3_block, w_block = w3_block;
  auto c_block = c3_block, k_block = k3_block;
  auto h_in_gemm = h3_in_gemm;
  auto conv_loop_string = c3_string;
  auto pack_input = 0; /* only could be non-zero for 1x1 strided to make any sense */

  double t_start, t_conv_start, t_conv_end, t_bn_stats_end, t_bn_end, t_end;
  #include "fused_conv_bn_fwd.h"
#ifdef TIMING
  time_c3        = t_conv_end - t_conv_start;
  time_b3        = t_bn_end - t_conv_end;
  time_c3b3      = t_end - t_start;
  time_b3stats   = t_bn_stats_end - t_conv_end;
  time_c3b3extra = (t_end - t_start) - (time_c3 + time_b3);
#endif
}

#ifdef TIMING
  auto buf = tuning_timings.request();
  float* ptr = (float*)buf.ptr;
  //if (tuning_timings.size())
  {
/*
    timings->time_c1 = time_c1;
    timings->time_c2 = time_c2;
    timings->time_c3 = time_c3;
    timings->time_c4 = time_c4;
    timings->time_b1 = time_b1;
    timings->time_b2 = time_b2;
    timings->time_b3 = time_b3;
    timings->time_b4 = time_b4;
    timings->time_c1b1 = time_c1b1;
    timings->time_c2b2 = time_c2b2;
    timings->time_c3b3 = time_c3b3;
    timings->time_c4b4 = time_c4b4;
    timings->time_c1b1extra = time_c1b1extra;
    timings->time_c2b2extra = time_c2b2extra;
    timings->time_c3b3extra = time_c3b3extra;
    timings->time_c4b4extra = time_c4b4extra;
    timings->time_b1stats = time_b1stats;
    timings->time_b2stats = time_b2stats;
    timings->time_b3stats = time_b3stats;
    timings->time_b4stats = time_b4stats;
*/
/*
    tuning_timings[0] += time_c1;
    tuning_timings[1] += time_c2;
    tuning_timings[2] += time_c3;
    tuning_timings[3] += time_c4;
    tuning_timings[4] += time_b1;
    tuning_timings[5] += time_b2;
    tuning_timings[6] += time_b3;
    tuning_timings[7] += time_b4;
    tuning_timings[8] += time_c1b1;
    tuning_timings[9] += time_c2b2;
    tuning_timings[10] += time_c3b3;
    tuning_timings[11] += time_c4b4;
    tuning_timings[12] += time_c1b1extra;
    tuning_timings[13] += time_c2b2extra;
    tuning_timings[14] += time_c3b3extra;
    tuning_timings[15] += time_c4b4extra;
*/
    ptr[0] += time_c1;
    ptr[1] += time_c2;
    ptr[2] += time_c3;
    ptr[3] += time_c4;
    ptr[4] += time_b1;
    ptr[5] += time_b2;
    ptr[6] += time_b3;
    ptr[7] += time_b4;
    ptr[8] += time_c1b1;
    ptr[9] += time_c2b2;
    ptr[10] += time_c3b3;
    ptr[11] += time_c4b4;
    ptr[12] += time_c1b1extra;
    ptr[13] += time_c2b2extra;
    ptr[14] += time_c3b3extra;
    ptr[15] += time_c4b4extra;
  }

  //printf("dbg: tuning_timings at the end of bf_fwd_ext = %f %f %f (time_c1 - c3 = %f %f %f)", ptr[0], ptr[1], ptr[2], time_c1, time_c2, time_c3);
#endif

  return {bn3_out, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, residual, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
            bn1_scratch_out, bn2_scratch_out, bn3_scratch_out, bn4_scratch_out };


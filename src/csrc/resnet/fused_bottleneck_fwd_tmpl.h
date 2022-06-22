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
  const long avoid_fmas_in_rim_from_params = tuning_params[16];

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
  auto avoid_fmas_in_rim = (avoid_fmas_in_rim_from_params == -1 ? conv_cfg.avoid_fmas_in_rim : avoid_fmas_in_rim_from_params);
  auto conv_loop_string = c1_string;
  #include "fused_conv_bn_fwd.h"
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
  auto avoid_fmas_in_rim = (avoid_fmas_in_rim_from_params == -1 ? conv_cfg.avoid_fmas_in_rim : avoid_fmas_in_rim_from_params);
  auto conv_loop_string = c2_string;
  #include "fused_conv_bn_fwd.h"
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
  auto avoid_fmas_in_rim = (avoid_fmas_in_rim_from_params == -1 ? conv_cfg.avoid_fmas_in_rim : avoid_fmas_in_rim_from_params);
  auto conv_loop_string = c4_string;
  #include "fused_conv_bn_fwd.h"
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
  auto avoid_fmas_in_rim = (avoid_fmas_in_rim_from_params == -1 ? conv_cfg.avoid_fmas_in_rim : avoid_fmas_in_rim_from_params);
  auto conv_loop_string = c3_string;
  #include "fused_conv_bn_fwd.h"
}

  return {bn3_out, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, residual, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
            bn1_scratch_out, bn2_scratch_out, bn3_scratch_out, bn4_scratch_out };


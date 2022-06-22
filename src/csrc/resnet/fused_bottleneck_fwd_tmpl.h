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

  std::vector<long> dummy_size{0};
  auto dummy_add    = at::zeros(dummy_size, input.options());
  auto dummy_return = at::zeros(dummy_size, input.options());
  auto dummy_stat   = at::zeros(dummy_size, bn1_weight.options());

  auto activation_dtype = input.dtype();

  /* FIXME: Fix this! */
  char conv_fwd_loop_specs_str[256] = "Abcdefg";

  auto global_fuse_scaling = 0; /* if set to 1, enables fusion of scaling for bn1 and bn2 in the subsequent convolutions conv2 and conv3 */

#ifdef VERBOSE
  printf("running conv1 + bn1\n");

  std::cout << "input dtype = " << input.dtype() << std::endl;
  std::cout << "bn1_weight dtype = " << bn1_weight.dtype() << std::endl;
  std::cout << "bn3_weight dtype = " << bn3_weight.dtype() << std::endl;
#endif

#if 1

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

  #include "fused_conv_bn_fwd.h"
}

#else
  //auto conv1_out = conv_forward_new(cfg.conv1, input, conv1_weight, conv1_output_size);
  auto conv1_out = conv_fwd(cfg.conv1, {input, conv1_weight});//conv1_inputs);

  //printf("running bn1\n");

  bool bn1_relu = true, bn1_eltwise = false;
  std::vector<long> bn1_padding{0, 0, cfg.pad_size, cfg.pad_size};
  auto bn1_ret = batchnorm_fwd(training, bn1_relu, bn1_eltwise, cfg.bn_eps, bn1_padding, std::vector<at::Tensor>{conv1_out, bn1_weight, bn1_bias, bn1_mean, bn1_var});
  auto bn1_out = bn1_ret[0];
  auto bn1_relu_out = bn1_ret[1];
  auto bn1_scratch_out = bn1_ret[2];
#endif

/*
auto dbg_conv1_out = conv1_out.data_ptr<T>();
printf("dbg_conv1_out data ptr = %p\n", dbg_conv1_out);
for (int i = 0; i < 20; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(dbg_conv1_out[i])), &tmp, 1);
                      printf("inp[%d] = %u = %f\n", i, *(unsigned short*)(&dbg_conv1_out[i]), tmp);
                    } else
                      printf("inp[%d] = %f\n", i, dbg_conv1_out[i]);
}

auto dbg_bn1_out = bn1_out.data_ptr<T>();
for (int i = 0; i < 500; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(dbg_bn1_out[i])), &tmp, 1);
                      printf("bn1 out[%d] = %u = %f\n", i, *(unsigned short*)(&dbg_bn1_out[i]), tmp);
                    } else
                      printf("bn1 out[%d] = %f\n", i, dbg_bn1_out[i]);
}
*/

#ifdef VERBOSE
  printf("running conv2 + bn2\n");
#endif

#if 1
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

  #include "fused_conv_bn_fwd.h"
}
#else

  //printf("running conv2\n");
  auto conv2_out = conv_fwd(cfg.conv2, {bn1_out, conv2_weight});//conv2_inputs);//bn1_out, conv2_weight, conv2_output_size);

  //printf("running bn2\n");

  bool bn2_relu = true, bn2_eltwise = false;
  std::vector<long> bn2_padding{cfg.pad_size, cfg.pad_size, 0, 0};
  auto bn2_ret = batchnorm_fwd(training, bn2_relu, bn2_eltwise, cfg.bn_eps, bn2_padding, std::vector<at::Tensor>{conv2_out, bn2_weight, bn2_bias, bn2_mean, bn2_var});
  auto bn2_out = bn2_ret[0];
  auto bn2_relu_out = bn2_ret[1];
  auto bn2_scratch_out = bn2_ret[2];
#endif

  at::Tensor conv4_out, residual, bn4_relu_out, bn4_scratch_out;
  if (cfg.has_residual_conv) {
#ifdef VERBOSE
  printf("running conv4 + bn4\n");
#endif

#if 1
//at::Tensor conv4_out, bn4_out, bn2_relu_out, bn2_scratch_out;

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

  #include "fused_conv_bn_fwd.h"
}
#else
    //printf("running conv4\n");
    conv4_out = conv_fwd(cfg.conv4, {input, conv4_weight});//conv4_inputs);

    //printf("running bn4\n");
    bool bn4_relu = false, bn4_eltwise = false;
    auto bn4_ret  = batchnorm_fwd(training, bn4_relu, bn4_eltwise, cfg.bn_eps, {0, 0, 0, 0}/*bn4_padding*/, std::vector<at::Tensor>{conv4_out, bn4_weight, bn4_bias, bn4_mean, bn4_var});
    residual = bn4_ret[0];
    bn4_relu_out = bn4_ret[1];
    bn4_scratch_out = bn4_ret[2];
#endif
  } else {
    conv4_out    = dummy_return;
    residual     = input;
    bn4_relu_out = dummy_return;
    bn4_scratch_out = dummy_return;
  }

#ifdef VERBOSE
  printf("running conv3 + bn3\n");
#endif

#if 1
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

  #include "fused_conv_bn_fwd.h"
}
#else

  //printf("running conv3\n");
  auto conv3_out = conv_fwd(cfg.conv3, {bn2_out, conv3_weight});//conv3_inputs);//bn2_out, conv3_weight, conv3_output_size);

  //printf("running bn3\n");
  bool bn3_relu = true, bn3_eltwise = true;
  auto bn3_ret = batchnorm_fwd(training, bn3_relu, bn3_eltwise, cfg.bn_eps, {0, 0, 0, 0}/*bn3_padding*/, std::vector<at::Tensor>{conv3_out, residual, bn3_weight, bn3_bias, bn3_mean, bn3_var});
  auto bn3_out = bn3_ret[0];
  auto bn3_relu_out = bn3_ret[1];
  auto bn3_scratch_out = bn3_ret[2];
#endif

  return {bn3_out, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, residual, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
            bn1_scratch_out, bn2_scratch_out, bn3_scratch_out, bn4_scratch_out };


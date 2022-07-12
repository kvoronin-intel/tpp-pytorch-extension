RECORD_FUNCTION("fused_bottleneck_bn_bwd", std::vector<c10::IValue>());

//auto t_dummy     = at::empty({0},  torch::TensorOptions().dtype(at::kFloat));
//return std::vector<at::Tensor>({t_dummy});

#define TIMING

#define VERBOSE

  auto grad_output  = inputs[0];
  auto conv1_input  = inputs[1];
  auto conv1_weight = inputs[2];
  auto conv2_weight = inputs[3];
  auto conv3_weight = inputs[4];
  auto conv4_weight = inputs[5];
  auto bn1_weight   = inputs[6];
  auto bn2_weight   = inputs[7];
  auto bn3_weight   = inputs[8];
  auto bn4_weight   = inputs[9];
  auto bn1_bias     = inputs[10];
  auto bn2_bias     = inputs[11];
  auto bn3_bias     = inputs[12];
  auto bn4_bias     = inputs[13];
  auto bn1_mean     = inputs[14];
  auto bn2_mean     = inputs[15];
  auto bn3_mean     = inputs[16];
  auto bn4_mean     = inputs[17];
  auto bn1_var      = inputs[18];
  auto bn2_var      = inputs[19];
  auto bn3_var      = inputs[20];
  auto bn4_var      = inputs[21];

  auto conv1_out    = inputs[22];
  auto bn1_out      = inputs[23];
  auto conv2_out    = inputs[24];
  auto bn2_out      = inputs[25];
  auto conv3_out    = inputs[26];
  auto bn3_out      = inputs[27];
  auto conv4_out    = inputs[28];
  auto bn4_out      = inputs[29];

  auto bn1_relu_out = inputs[30];
  auto bn2_relu_out = inputs[31];
  auto bn3_relu_out = inputs[32];
  auto bn4_relu_out = inputs[33];

  auto bn1_scratch   = inputs[34];
  auto bn2_scratch   = inputs[35];
  auto bn3_scratch   = inputs[36];
  auto bn4_scratch   = inputs[37];

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

  const std::string c1_string = tuning_strings[0];
  const std::string c2_string = tuning_strings[1];
  const std::string c3_string = tuning_strings[2];
  const std::string c4_string = tuning_strings[3];

#ifdef TIMING
  double t_start = 0.0;
  double time_b1 = 0.0, time_b2 = 0.0, time_b3 = 0.0, time_b4 = 0.0;
#endif

  std::vector<long> dummy_size{0};
  auto dummy_add    = at::zeros(dummy_size, conv1_input.options());
  auto dummy_return = at::zeros(dummy_size, conv1_input.options());

  pybind11::array_t<float> tuning_timings_d1{0, 0, 0}, tuning_timings_d2{0, 0, 0}, tuning_timings_d3{0, 0, 0}, tuning_timings_d4{0, 0, 0};

  at::Tensor  conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
              bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
              bn1_grad_beta,  bn2_grad_beta,  bn3_grad_beta,  bn4_grad_beta,
              conv1_grad_input, conv4_grad_input;

#ifdef VERBOSE
  printf("running bn3 bwd\n");
#endif

#ifdef TIMING
  t_start = getTime();
#endif

  bool bn3_relu = true, bn3_eltwise = true;
  auto residual           = bn4_out; // FIXME: Hopefully an alias and not a memory leak
  auto bn3_grad_ret       = batchnorm_bwd(bn3_relu, bn3_eltwise, cfg.bn_eps, {0, 0, 0, 0}, {grad_output, conv3_out, bn3_weight, bn3_mean, bn3_var, bn3_relu_out, bn3_scratch});
  //(cfg.bn3, grad_output /*grad_output*/, conv3_out, residual /*input_add*/, bn3_weight, dummy_output /*output*/, bn3_mean, bn3_var, dummy_invstd, bn3_relu_out);
  auto bn3_grad_input     = bn3_grad_ret[0];
  auto bn3_grad_input_add = bn3_grad_ret[1];
  bn3_grad_gamma          = bn3_grad_ret[2];
  bn3_grad_beta           = bn3_grad_ret[3];

#ifdef TIMING
  time_b3 = getTime() - t_start;
#endif

#ifdef VERBOSE
  printf("running conv3 bwd\n");
#endif

  bn2_out.requires_grad_(true);
#ifdef BWD_D_ONLY
  auto conv3_grad_input = conv_bwd_d_ext(cfg.conv3, {bn3_grad_input, bn2_out, conv3_weight}, {h3_block, w3_block, c3_block, k3_block, h3_in_gemm}, c3_string, tuning_timings_d3);
  conv3_grad_weight = dummy_return;
#else
  auto conv3_grad_ret   = conv_bwd_ext(cfg.conv3, {bn3_grad_input, bn2_out, conv3_weight}, {h3_block, w3_block, c3_block, k3_block, h3_in_gemm}, c3_string, tuning_timings_d3);
  //conv_backward_new(cfg.conv3, bn3_grad_input /*grad_output*/, bn2_out, conv3_weight);
  auto conv3_grad_input = conv3_grad_ret[0];
  conv3_grad_weight     = conv3_grad_ret[1];
#endif

#ifdef VERBOSE
  printf("running bn2 bwd\n");
#endif

#ifdef TIMING
  t_start = getTime();
#endif

  bool bn2_relu = true, bn2_eltwise = false;
  auto bn2_grad_ret   = batchnorm_bwd(bn2_relu, bn2_eltwise, cfg.bn_eps, {cfg.pad_size, cfg.pad_size, 0, 0}, {conv3_grad_input, conv2_out, bn2_weight, bn2_mean, bn2_var, bn2_relu_out, bn2_scratch});
  //bnorm_backward_new(cfg.bn2, conv3_grad_input /*grad_output*/, conv2_out, dummy_add, bn2_weight, dummy_output /*output*/, bn2_mean, bn2_var, dummy_invstd, bn2_relu_out);
  auto bn2_grad_input = bn2_grad_ret[0];
  bn2_grad_gamma      = bn2_grad_ret[1];
  bn2_grad_beta       = bn2_grad_ret[2];

#ifdef TIMING
  time_b2 = getTime() - t_start;
#endif

#ifdef VERBOSE
  printf("running conv2 bwd\n");
#endif

  bn1_out.requires_grad_(true);
#ifdef BWD_D_ONLY
  auto conv2_grad_input = conv_bwd_d_ext(cfg.conv2, {bn2_grad_input, bn1_out, conv2_weight}, {h2_block, w2_block, c2_block, k2_block, h2_in_gemm}, c2_string, tuning_timings_d2);
  conv2_grad_weight = dummy_return;
#else
  auto conv2_grad_ret   = conv_bwd_ext(cfg.conv2, {bn2_grad_input, bn1_out, conv2_weight}, {h2_block, w2_block, c2_block, k2_block, h2_in_gemm}, c2_string, tuning_timings_d2);
  //conv_backward_new(cfg.conv2, bn2_grad_input /*grad_output*/, bn1_out, conv2_weight);
  auto conv2_grad_input = conv2_grad_ret[0];
  conv2_grad_weight     = conv2_grad_ret[1];
#endif

#ifdef VERBOSE
  printf("running bn1 bwd\n");
#endif

#ifdef TIMING
  t_start = getTime();
#endif

  bool bn1_relu = true, bn1_eltwise = false;
  auto bn1_grad_ret   = batchnorm_bwd(bn1_relu, bn1_eltwise, cfg.bn_eps, {0, 0, cfg.pad_size, cfg.pad_size}, {conv2_grad_input, conv1_out, bn1_weight, bn1_mean, bn1_var, bn1_relu_out, bn1_scratch});
  //bnorm_backward_new(cfg.bn1, conv2_grad_input /*grad_output*/, conv1_out, dummy_add, bn1_weight, dummy_output /*output*/, bn1_mean, bn1_var, dummy_invstd, bn1_relu_out);
  auto bn1_grad_input = bn1_grad_ret[0];
  bn1_grad_gamma      = bn1_grad_ret[1];
  bn1_grad_beta       = bn1_grad_ret[2];

#ifdef TIMING
  time_b1 = getTime() - t_start;
#endif

#ifdef VERBOSE
  printf("running conv1 bwd\n");
#endif

  conv1_input.requires_grad_(true);
#ifdef BWD_D_ONLY
  conv1_grad_input = conv_bwd_d_ext(cfg.conv1,  {bn1_grad_input, conv1_input, conv1_weight}, {h1_block, w1_block, c1_block, k1_block, h1_in_gemm}, c1_string, tuning_timings_d1);
  conv1_grad_weight = dummy_return;
#else
  auto conv1_grad_ret = conv_bwd_ext(cfg.conv1,  {bn1_grad_input, conv1_input, conv1_weight}, {h1_block, w1_block, c1_block, k1_block, h1_in_gemm}, c1_string, tuning_timings_d1);
  //conv_backward_new(cfg.conv1, bn1_grad_input /*grad_output*/, conv1_input, conv1_weight);
  conv1_grad_input    = conv1_grad_ret[0];
  conv1_grad_weight   = conv1_grad_ret[1];
#endif

  if (cfg.has_residual_conv) {

#ifdef VERBOSE
    printf("running bn4 bwd\n");
#endif

#ifdef TIMING
    t_start = getTime();
#endif

    bool bn4_relu = false, bn4_eltwise = false;
    auto bn4_grad_ret   = batchnorm_bwd(bn4_relu, bn4_eltwise, cfg.bn_eps, {0, 0, 0, 0}, {bn3_grad_input_add, conv4_out, bn4_weight, bn4_mean, bn4_var, bn4_relu_out, bn4_scratch});
    //bnorm_backward_new(cfg.bn4, bn3_grad_input_add /*grad_output*/, conv4_out, dummy_add, bn4_weight, dummy_output /*output*/, bn4_mean, bn4_var, dummy_invstd, bn4_relu_out);
    auto bn4_grad_input = bn4_grad_ret[0];
    bn4_grad_gamma      = bn4_grad_ret[1];
    bn4_grad_beta       = bn4_grad_ret[2];

#ifdef TIMING
    time_b4 = getTime() - t_start;
#endif

#ifdef VERBOSE
    printf("running conv4 bwd\n");
#endif

    conv1_input.requires_grad_(true);
#ifdef BWD_D_ONLY
    auto conv4_grad_input = conv_bwd_d_ext(cfg.conv4, {bn4_grad_input, conv1_input, conv4_weight}, {h4_block, w4_block, c4_block, k4_block, h4_in_gemm}, c4_string, tuning_timings_d4);
    conv4_grad_weight = dummy_return;
#else
    auto conv4_grad_ret = conv_bwd_ext(cfg.conv4, {bn4_grad_input, conv1_input, conv4_weight}, {h4_block, w4_block, c4_block, k4_block, h4_in_gemm}, c4_string, tuning_timings_d4);
    //conv_backward_new(cfg.conv4, bn4_grad_input /*grad_output*/, conv1_input, conv4_weight);
    conv4_grad_input    = conv4_grad_ret[0];
    conv4_grad_weight   = conv4_grad_ret[1];
#endif

  } else {
    conv4_grad_weight = dummy_return;
    bn4_grad_gamma    = dummy_return;
    bn4_grad_beta     = dummy_return;
    conv4_grad_input  = bn3_grad_input_add;
    conv4_grad_weight = dummy_return;
  }

#ifdef TIMING
  auto buf = tuning_timings.request();
  float* ptr = (float*)buf.ptr;
  //if (tuning_timings.size())
  {
    auto buf_d1 = tuning_timings_d1.request();
    float* ptr_d1 = (float*)buf_d1.ptr;
    auto buf_d2 = tuning_timings_d2.request();
    float* ptr_d2 = (float*)buf_d2.ptr;
    auto buf_d3 = tuning_timings_d3.request();
    float* ptr_d3 = (float*)buf_d3.ptr;
    auto buf_d4 = tuning_timings_d4.request();
    float* ptr_d4 = (float*)buf_d4.ptr;
    ptr[0] += ptr_d1[0];
    ptr[1] += ptr_d2[0];
    ptr[2] += ptr_d3[0];
    ptr[3] += ptr_d4[0];
    ptr[4] += time_b1;
    ptr[5] += time_b2;
    ptr[6] += time_b3;
    ptr[7] += time_b4;
  }

  //printf("dbg: tuning_timings at the end of bf_fwd_ext = %f %f %f (time_c1 - c3 = %f %f %f)", ptr[0], ptr[1], ptr[2], time_c1, time_c2, time_c3);
#endif

  return {conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
          bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
          bn1_grad_beta, bn2_grad_beta, bn3_grad_beta, bn4_grad_beta,
          conv1_grad_input, conv4_grad_input};


RECORD_FUNCTION("bottleneck_bn_bwd", std::vector<c10::IValue>());

//auto t_dummy     = at::empty({0},  torch::TensorOptions().dtype(at::kFloat));
//return std::vector<at::Tensor>({t_dummy});

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

  std::vector<long> dummy_size{0};
  //auto dummy_add    = at::Tensor(dummy_size, input.options());
  //auto dummy_invstd = at::Tensor(dummy_size, input.options());
  //auto dummy_return = at::Tensor(dummy_size, input.options());
  //auto dummy_add = input;
  auto dummy_add = at::zeros(dummy_size, conv1_input.options());
  auto dummy_return = at::zeros(dummy_size, conv1_input.options());

  //auto dummy_output = at::Tensor();
  //auto dummy_add    = at::Tensor();
  //auto dummy_invstd = at::Tensor();
  //auto dummy_return = at::Tensor();

  at::Tensor  conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
      bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
          bn1_grad_beta, bn2_grad_beta, bn3_grad_beta, bn4_grad_beta,
              conv1_grad_input, conv4_grad_input;

//  printf("running bn3 bwd\n");

  bool bn3_relu = true, bn3_eltwise = true;
  auto residual           = bn4_out; // FIXME: Hopefully an alias and not a memory leak
  auto bn3_grad_ret       = batchnorm_bwd(bn3_relu, bn3_eltwise, cfg.bn_eps, {0, 0, 0, 0}, {grad_output, conv3_out, residual, bn3_weight, bn3_mean, bn3_var, bn3_relu_out});
  //(cfg.bn3, grad_output /*grad_output*/, conv3_out, residual /*input_add*/, bn3_weight, dummy_output /*output*/, bn3_mean, bn3_var, dummy_invstd, bn3_relu_out);
  auto bn3_grad_input     = bn3_grad_ret[0];
  auto bn3_grad_input_add = bn3_grad_ret[1];
  bn3_grad_gamma          = bn3_grad_ret[2];
  bn3_grad_beta           = bn3_grad_ret[3];

//  printf("running conv3 bwd\n");

  bn2_out.requires_grad_(true);
  auto conv3_grad_ret   = conv_bwd(cfg.conv3, {bn3_grad_input, bn2_out, conv3_weight});
  //conv_backward_new(cfg.conv3, bn3_grad_input /*grad_output*/, bn2_out, conv3_weight);
  auto conv3_grad_input = conv3_grad_ret[0];
  conv3_grad_weight     = conv3_grad_ret[1];

//  printf("running bn2 bwd\n");

  bool bn2_relu = true, bn2_eltwise = false;
  auto bn2_grad_ret   = batchnorm_bwd(bn2_relu, bn2_eltwise, cfg.bn_eps, {cfg.pad_size, cfg.pad_size, 0, 0}, {conv3_grad_input, conv2_out, dummy_add, bn2_weight, bn2_mean, bn2_var, bn2_relu_out});
  //bnorm_backward_new(cfg.bn2, conv3_grad_input /*grad_output*/, conv2_out, dummy_add, bn2_weight, dummy_output /*output*/, bn2_mean, bn2_var, dummy_invstd, bn2_relu_out);
  auto bn2_grad_input = bn2_grad_ret[0];
  bn2_grad_gamma      = bn2_grad_ret[2];
  bn2_grad_beta       = bn2_grad_ret[3];

//  float *bn2_grad_input_dbg = (float*)(bn2_grad_input.data_ptr<float>());
//  for (int i = 0; i < 10; i++)
//    printf("in btlnk bwd bn2_grad_input[%d] = %f \n", i, bn2_grad_input_dbg[i]);

//  printf("running conv2 bwd\n");

  bn1_out.requires_grad_(true);
  auto conv2_grad_ret   = conv_bwd(cfg.conv2, {bn2_grad_input, bn1_out, conv2_weight});
  //conv_backward_new(cfg.conv2, bn2_grad_input /*grad_output*/, bn1_out, conv2_weight);
  auto conv2_grad_input = conv2_grad_ret[0];
  conv2_grad_weight     = conv2_grad_ret[1];

//  printf("running bn1 bwd\n");

  bool bn1_relu = true, bn1_eltwise = false;
  auto bn1_grad_ret   = batchnorm_bwd(bn1_relu, bn1_eltwise, cfg.bn_eps, {0, 0, cfg.pad_size, cfg.pad_size}, {conv2_grad_input, conv1_out, dummy_add, bn1_weight, bn1_mean, bn1_var, bn1_relu_out});
  //bnorm_backward_new(cfg.bn1, conv2_grad_input /*grad_output*/, conv1_out, dummy_add, bn1_weight, dummy_output /*output*/, bn1_mean, bn1_var, dummy_invstd, bn1_relu_out);
  auto bn1_grad_input = bn1_grad_ret[0];
  bn1_grad_gamma      = bn1_grad_ret[2];
  bn1_grad_beta       = bn1_grad_ret[3];

//  printf("running conv1 bwd\n");

  conv1_input.requires_grad_(true);
  auto conv1_grad_ret = conv_bwd(cfg.conv1,  {bn1_grad_input, conv1_input, conv1_weight});
  //conv_backward_new(cfg.conv1, bn1_grad_input /*grad_output*/, conv1_input, conv1_weight);
  conv1_grad_input    = conv1_grad_ret[0];
  conv1_grad_weight   = conv1_grad_ret[1];

  if (cfg.has_residual_conv) {

//    printf("running bn4 bwd\n");

    bool bn4_relu = false, bn4_eltwise = false;
    auto bn4_grad_ret   = batchnorm_bwd(bn4_relu, bn4_eltwise, cfg.bn_eps, {0, 0, 0, 0}, {bn3_grad_input_add, conv4_out, dummy_add, bn4_weight, bn4_mean, bn4_var, bn4_relu_out});
    //bnorm_backward_new(cfg.bn4, bn3_grad_input_add /*grad_output*/, conv4_out, dummy_add, bn4_weight, dummy_output /*output*/, bn4_mean, bn4_var, dummy_invstd, bn4_relu_out);
    auto bn4_grad_input = bn4_grad_ret[0];
    bn4_grad_gamma      = bn4_grad_ret[2];
    bn4_grad_beta       = bn4_grad_ret[3];

//    printf("running conv4 bwd\n");

    conv1_input.requires_grad_(true);
    auto conv4_grad_ret = conv_bwd(cfg.conv4, {bn4_grad_input, conv1_input, conv4_weight});
    //conv_backward_new(cfg.conv4, bn4_grad_input /*grad_output*/, conv1_input, conv4_weight);
    conv4_grad_input    = conv4_grad_ret[0];
    conv4_grad_weight   = conv4_grad_ret[1];

  } else {
    conv4_grad_weight = dummy_return;
    bn4_grad_gamma    = dummy_return;
    bn4_grad_beta     = dummy_return;
    conv4_grad_input  = bn3_grad_input_add;
    conv4_grad_weight = dummy_return;
  }

  return {conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
          bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
          bn1_grad_beta, bn2_grad_beta, bn3_grad_beta, bn4_grad_beta,
          conv1_grad_input, conv4_grad_input};


RECORD_FUNCTION("bottleneck_bn_fwd", std::vector<c10::IValue>());

#define VERBOSE

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
  auto dummy_add = at::zeros(dummy_size, input.options());
  auto dummy_return = at::zeros(dummy_size, input.options());

#ifdef VERBOSE
  printf("running conv1\n");
#endif

  //auto conv1_out = conv_forward_new(cfg.conv1, input, conv1_weight, conv1_output_size);
  auto conv1_out = conv_fwd(cfg.conv1, {input, conv1_weight});//conv1_inputs);

#ifdef VERBOSE
  printf("running bn1\n");
#endif

  bool bn1_relu = true, bn1_eltwise = false;
  std::vector<long> bn1_padding{0, 0, cfg.pad_size, cfg.pad_size};
  auto bn1_ret = batchnorm_fwd(training, bn1_relu, bn1_eltwise, cfg.bn_eps, bn1_padding, std::vector<at::Tensor>{conv1_out, bn1_weight, bn1_bias, bn1_mean, bn1_var});
  auto bn1_out = bn1_ret[0];
  auto bn1_relu_out = bn1_ret[1];
  auto bn1_scratch_out = bn1_ret[2];

#ifdef VERBOSE
  printf("running conv2\n");
#endif
  auto conv2_out = conv_fwd(cfg.conv2, {bn1_out, conv2_weight});//conv2_inputs);//bn1_out, conv2_weight, conv2_output_size);

#ifdef VERBOSE
  printf("running bn2\n");
#endif

  bool bn2_relu = true, bn2_eltwise = false;
  std::vector<long> bn2_padding{cfg.pad_size, cfg.pad_size, 0, 0};
  auto bn2_ret = batchnorm_fwd(training, bn2_relu, bn2_eltwise, cfg.bn_eps, bn2_padding, std::vector<at::Tensor>{conv2_out, bn2_weight, bn2_bias, bn2_mean, bn2_var});
  auto bn2_out = bn2_ret[0];
  auto bn2_relu_out = bn2_ret[1];
  auto bn2_scratch_out = bn2_ret[2];

#ifdef VERBOSE
  printf("running conv3\n");
#endif
  auto conv3_out = conv_fwd(cfg.conv3, {bn2_out, conv3_weight});//conv3_inputs);//bn2_out, conv3_weight, conv3_output_size);

  at::Tensor conv4_out, residual, bn4_relu_out, bn4_scratch_out;
  if (cfg.has_residual_conv) {
#ifdef VERBOSE
    printf("running conv4\n");
#endif

    conv4_out = conv_fwd(cfg.conv4, {input, conv4_weight});//conv4_inputs);

#ifdef VERBOSE
    printf("running bn4\n");
#endif
    bool bn4_relu = false, bn4_eltwise = false;
    auto bn4_ret  = batchnorm_fwd(training, bn4_relu, bn4_eltwise, cfg.bn_eps, {0, 0, 0, 0}/*bn4_padding*/, std::vector<at::Tensor>{conv4_out, bn4_weight, bn4_bias, bn4_mean, bn4_var});
    residual = bn4_ret[0];
    bn4_relu_out = bn4_ret[1];
    bn4_scratch_out = bn4_ret[2];
  } else {
    conv4_out    = dummy_return;
    residual     = input;
    bn4_relu_out = dummy_return;
    bn4_scratch_out = dummy_return;
  }

  //printf("running bn3\n");
  bool bn3_relu = true, bn3_eltwise = true;
  auto bn3_ret = batchnorm_fwd(training, bn3_relu, bn3_eltwise, cfg.bn_eps, {0, 0, 0, 0}/*bn3_padding*/, std::vector<at::Tensor>{conv3_out, residual, bn3_weight, bn3_bias, bn3_mean, bn3_var});
  auto bn3_out = bn3_ret[0];
  auto bn3_relu_out = bn3_ret[1];
  auto bn3_scratch_out = bn3_ret[2];

  return {bn3_out, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, residual, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
            bn1_scratch_out, bn2_scratch_out, bn3_scratch_out, bn4_scratch_out };


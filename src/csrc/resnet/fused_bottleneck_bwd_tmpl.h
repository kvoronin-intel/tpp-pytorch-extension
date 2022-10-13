RECORD_FUNCTION("fused_bottleneck_bn_bwd", std::vector<c10::IValue>());

//auto t_dummy     = at::empty({0},  torch::TensorOptions().dtype(at::kFloat));
//return std::vector<at::Tensor>({t_dummy});

#define TIMING

//#define VERBOSE

//#define CHECK_FOR_NANS

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

#ifndef BWD_W_ONLY
  const long h1_block = tuning_params_d[0];
  const long w1_block = tuning_params_d[1];
  const long h2_block = tuning_params_d[2];
  const long w2_block = tuning_params_d[3];
  const long h3_block = tuning_params_d[4];
  const long w3_block = tuning_params_d[5];
  const long h4_block = tuning_params_d[6];
  const long w4_block = tuning_params_d[7];
  const long c1_block = tuning_params_d[8];
  const long k1_block = tuning_params_d[9];
  const long c2_block = tuning_params_d[10];
  const long k2_block = tuning_params_d[11];
  const long c3_block = tuning_params_d[12];
  const long k3_block = tuning_params_d[13];
  const long c4_block = tuning_params_d[14];
  const long k4_block = tuning_params_d[15];
  const long h1_in_gemm = tuning_params_d[16];
  const long h2_in_gemm = tuning_params_d[17];
  const long h3_in_gemm = tuning_params_d[18];
  const long h4_in_gemm = tuning_params_d[19];

  const std::string cd1_string = tuning_strings_d[0];
  const std::string cd2_string = tuning_strings_d[1];
  const std::string cd3_string = tuning_strings_d[2];
  const std::string cd4_string = tuning_strings_d[3];
#endif

#ifndef BWD_D_ONLY
  const long p1_block = tuning_params_w[0];
  const long p2_block = tuning_params_w[1];
  const long p3_block = tuning_params_w[2];
  const long p4_block = tuning_params_w[3];

  const long c1_use_nchw_format = tuning_params_w[4];
  const long c2_use_nchw_format = tuning_params_w[5];
  const long c3_use_nchw_format = tuning_params_w[6];
  const long c4_use_nchw_format = tuning_params_w[7];

  const long pack_input_upfront                         = tuning_params_w[8];
  const long fuse_upd_transposes                        = tuning_params_w[9];
  const long use_f32_wt_reduction_and_external_wt_vnni  = tuning_params_w[10];
  const long bf16_acc_nw                                = tuning_params_w[11];
  const long par_over_h_pixels                          = tuning_params_w[12];
  const long compute_full_wt_output_block               = tuning_params_w[13];

  const long use_hybrid_imgfm_parallelization           = tuning_params_w[14];
  const long n_img_teams                                = tuning_params_w[15];
  const long n_ofm_teams                                = tuning_params_w[16];

  const std::string cw1_string = tuning_strings_w[0];
  const std::string cw2_string = tuning_strings_w[1];
  const std::string cw3_string = tuning_strings_w[2];
  const std::string cw4_string = tuning_strings_w[3];
#endif

#ifdef TIMING
  double t_start = 0.0;
  double time_b1 = 0.0, time_b2 = 0.0, time_b3 = 0.0, time_b4 = 0.0, time_c1 = 0.0, time_c2 = 0.0, time_c3 = 0.0, time_c4 = 0.0;
#endif

  std::vector<long> dummy_size{0};
  auto dummy_add    = at::zeros(dummy_size, conv1_input.options());
  auto dummy_return = at::zeros(dummy_size, conv1_input.options());

  pybind11::array_t<float> tuning_timings_d1(3), tuning_timings_d2(3), tuning_timings_d3(3), tuning_timings_d4(3);
  pybind11::array_t<float> tuning_timings_w1(3), tuning_timings_w2(3), tuning_timings_w3(3), tuning_timings_w4(3);
  {
    float *ptr_d1 = tuning_timings_d1.mutable_data();
    for (int i = 0; i < 3; i++)
        ptr_d1[i] = 0.0;
    float *ptr_d2 = tuning_timings_d2.mutable_data();
    for (int i = 0; i < 3; i++)
        ptr_d2[i] = 0.0;
    float *ptr_d3 = tuning_timings_d3.mutable_data();
    for (int i = 0; i < 3; i++)
        ptr_d3[i] = 0.0;
    float *ptr_d4 = tuning_timings_d4.mutable_data();
    for (int i = 0; i < 3; i++)
        ptr_d4[i] = 0.0;
    float *ptr_w1 = tuning_timings_w1.mutable_data();
    for (int i = 0; i < 3; i++)
        ptr_w1[i] = 0.0;
    float *ptr_w2 = tuning_timings_w2.mutable_data();
    for (int i = 0; i < 3; i++)
        ptr_w2[i] = 0.0;
    float *ptr_w3 = tuning_timings_w3.mutable_data();
    for (int i = 0; i < 3; i++)
        ptr_w3[i] = 0.0;
    float *ptr_w4 = tuning_timings_w4.mutable_data();
    for (int i = 0; i < 3; i++)
        ptr_w4[i] = 0.0;
    //printf("dbg:in fused btlnk bwd initial values are %f %f %f %f (c1 to c4)\n", ptr_d1[0], ptr_d2[0], ptr_d3[0], ptr_d4[0]);
  }

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
  //auto bn3_grad_ret       = batchnorm_bwd(bn3_relu, bn3_eltwise, cfg.bn_eps, {0, 0, 0, 0}, {grad_output, conv3_out, bn3_weight, bn3_mean, bn3_var, bn3_relu_out, bn3_scratch});
#ifdef BWD_D_ONLY
  std::array<std::string, 2> matched_bwd_strings3 = parse_conv_loop_string_for_batchnorm(cd3_string.c_str(), 1 /* is_nckhwrs */, 1 /* is nchw */);
#elif defined(BWD_W_ONLY)
  std::array<std::string, 2> matched_bwd_strings3 = parse_conv_loop_string_for_batchnorm(cw3_string.c_str(), 0 /* is_nckhwrs */, c3_use_nchw_format /* is nchw */);
#else
  std::array<std::string, 2> matched_bwd_strings3 = parse_conv_loop_string_for_batchnorm(cd3_string.c_str(), 1 /* is_nkchwrs */, 1 /* is nchw */);
#endif
  auto bn3_grad_ret       = batchnorm_bwd_ext(bn3_relu, bn3_eltwise, cfg.bn_eps, {0, 0, 0, 0}, matched_bwd_strings3[0], matched_bwd_strings3[1],
                                              {grad_output, conv3_out, bn3_weight, bn3_mean, bn3_var, bn3_relu_out, bn3_scratch});
  //(cfg.bn3, grad_output /*grad_output*/, conv3_out, residual /*input_add*/, bn3_weight, dummy_output /*output*/, bn3_mean, bn3_var, dummy_invstd, bn3_relu_out);
  auto bn3_grad_input     = bn3_grad_ret[0];
  auto bn3_grad_input_add = bn3_grad_ret[1];
  bn3_grad_gamma          = bn3_grad_ret[2];
  bn3_grad_beta           = bn3_grad_ret[3];

#ifdef TIMING
  time_b3 = getTime() - t_start;
  t_start = time_b3 + t_start;
#endif

#ifdef VERBOSE
  printf("running conv3 bwd\n");
#endif

  bn2_out.requires_grad_(true);
#ifdef BWD_D_ONLY
  auto conv3_grad_input = conv_bwd_d_ext(cfg.conv3, {bn3_grad_input, bn2_out, conv3_weight}, {h3_block, w3_block, c3_block, k3_block, h3_in_gemm}, cd3_string, tuning_timings_d3);
  conv3_grad_weight = dummy_return;
#elif defined(BWD_W_ONLY)
  auto conv3_grad_input = at::empty(bn2_out.sizes(), torch::TensorOptions().dtype(bn2_out.dtype()));//dummy_return;
  conv3_grad_weight = conv_bwd_w_ext(cfg.conv3, {bn3_grad_input, bn2_out, conv3_weight},
                                      {p3_block, c3_use_nchw_format, pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                          bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                            use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams},
                                      cw3_string, tuning_timings_w3);
#else
  auto conv3_grad_ret   = conv_bwd_ext(cfg.conv3, {bn3_grad_input, bn2_out, conv3_weight},
                                       {h3_block, w3_block, c3_block, k3_block, h3_in_gemm}, cd3_string, tuning_timings_d3,
                                       {p3_block, c3_use_nchw_format, pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                          bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                            use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams},
                                       cw3_string, tuning_timings_w3);
  //conv_backward_new(cfg.conv3, bn3_grad_input /*grad_output*/, bn2_out, conv3_weight);
  auto conv3_grad_input = conv3_grad_ret[0];
  conv3_grad_weight     = conv3_grad_ret[1];
#endif

#ifdef TIMING
  time_c3 = getTime() - t_start;
  t_start = time_c3 + t_start;
#endif

#ifdef VERBOSE
  printf("running bn2 bwd\n");
#endif

  bool bn2_relu = true, bn2_eltwise = false;
  //auto bn2_grad_ret   = batchnorm_bwd(bn2_relu, bn2_eltwise, cfg.bn_eps, {cfg.pad_size, cfg.pad_size, 0, 0}, {conv3_grad_input, conv2_out, bn2_weight, bn2_mean, bn2_var, bn2_relu_out, bn2_scratch});
#ifdef BWD_D_ONLY
  std::array<std::string, 2> matched_bwd_strings2 = parse_conv_loop_string_for_batchnorm(cd2_string.c_str(), 1 /* is_nckhwrs */, 1 /* is nchw */);
#elif defined(BWD_W_ONLY)
  std::array<std::string, 2> matched_bwd_strings2 = parse_conv_loop_string_for_batchnorm(cw2_string.c_str(), 0 /* is_nckhwrs */, c2_use_nchw_format /* is nchw */);
#else
  std::array<std::string, 2> matched_bwd_strings2 = parse_conv_loop_string_for_batchnorm(cd2_string.c_str(), 1 /* is_nkchwrs */, 1 /* is nchw */);
#endif
  auto bn2_grad_ret   = batchnorm_bwd_ext(bn2_relu, bn2_eltwise, cfg.bn_eps, {cfg.pad_size, cfg.pad_size, 0, 0}, matched_bwd_strings2[0], matched_bwd_strings2[1],
                                          {conv3_grad_input, conv2_out, bn2_weight, bn2_mean, bn2_var, bn2_relu_out, bn2_scratch});
  //bnorm_backward_new(cfg.bn2, conv3_grad_input /*grad_output*/, conv2_out, dummy_add, bn2_weight, dummy_output /*output*/, bn2_mean, bn2_var, dummy_invstd, bn2_relu_out);
  auto bn2_grad_input = bn2_grad_ret[0];
  bn2_grad_gamma      = bn2_grad_ret[1];
  bn2_grad_beta       = bn2_grad_ret[2];

#ifdef CHECK_FOR_NANS
  {
  auto check_length = bn2_grad_input.numel();
  int nan_count = 0;
  for (int i = 0; i < check_length; i++)
    if (aten::isnan((bn2_grad_input.data_ptr<T>())[i])
      nan_count++;
  if (nan_count) {
    printf("nan_count is not zero in bn2_grad_input: %d \n", nan_count);
  }
  }
#endif

#ifdef TIMING
  time_b2 = getTime() - t_start;
  t_start = time_b2 + t_start;
#endif

#ifdef VERBOSE
  printf("running conv2 bwd\n");
#endif

  bn1_out.requires_grad_(true);
#ifdef BWD_D_ONLY
  auto conv2_grad_input = conv_bwd_d_ext(cfg.conv2, {bn2_grad_input, bn1_out, conv2_weight}, {h2_block, w2_block, c2_block, k2_block, h2_in_gemm}, cd2_string, tuning_timings_d2);
  conv2_grad_weight = dummy_return;
#elif defined(BWD_W_ONLY)
  auto conv2_grad_input = at::empty(bn1_out.sizes(), torch::TensorOptions().dtype(bn1_out.dtype()));//dummy_return;
  conv2_grad_weight = conv_bwd_w_ext(cfg.conv2, {bn2_grad_input, bn1_out, conv2_weight},
                                     {p2_block, c2_use_nchw_format, pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                          bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                            use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams},
                                     cw2_string, tuning_timings_w2);
#else
  auto conv2_grad_ret   = conv_bwd_ext(cfg.conv2, {bn2_grad_input, bn1_out, conv2_weight},
                                       {h2_block, w2_block, c2_block, k2_block, h2_in_gemm}, cd2_string, tuning_timings_d2,
                                       {p2_block, c2_use_nchw_format, pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                          bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                            use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams},
                                       cw2_string, tuning_timings_w2);
  //conv_backward_new(cfg.conv2, bn2_grad_input /*grad_output*/, bn1_out, conv2_weight);
  auto conv2_grad_input = conv2_grad_ret[0];
  conv2_grad_weight     = conv2_grad_ret[1];
#endif

#ifdef TIMING
  time_c2 = getTime() - t_start;
  t_start = time_c2 + t_start;
#endif

#ifdef VERBOSE
  printf("running bn1 bwd\n");
#endif

  bool bn1_relu = true, bn1_eltwise = false;
  //auto bn1_grad_ret   = batchnorm_bwd(bn1_relu, bn1_eltwise, cfg.bn_eps, {0, 0, cfg.pad_size, cfg.pad_size}, {conv2_grad_input, conv1_out, bn1_weight, bn1_mean, bn1_var, bn1_relu_out, bn1_scratch});
#ifdef BWD_D_ONLY
  std::array<std::string, 2> matched_bwd_strings1 = parse_conv_loop_string_for_batchnorm(cd1_string.c_str(), 1 /* is_nckhwrs */, 1 /* is nchw */);
#elif defined(BWD_W_ONLY)
  std::array<std::string, 2> matched_bwd_strings1 = parse_conv_loop_string_for_batchnorm(cw1_string.c_str(), 0 /* is_nckhwrs */, c1_use_nchw_format /* is nchw */);
#else
  std::array<std::string, 2> matched_bwd_strings1 = parse_conv_loop_string_for_batchnorm(cd1_string.c_str(), 1 /* is_nkchwrs */, 1 /* is nchw */);
#endif
  auto bn1_grad_ret   = batchnorm_bwd_ext(bn1_relu, bn1_eltwise, cfg.bn_eps, {0, 0, cfg.pad_size, cfg.pad_size}, matched_bwd_strings1[0], matched_bwd_strings1[1],
                        {conv2_grad_input, conv1_out, bn1_weight, bn1_mean, bn1_var, bn1_relu_out, bn1_scratch});
  //bnorm_backward_new(cfg.bn1, conv2_grad_input /*grad_output*/, conv1_out, dummy_add, bn1_weight, dummy_output /*output*/, bn1_mean, bn1_var, dummy_invstd, bn1_relu_out);
  auto bn1_grad_input = bn1_grad_ret[0];
  bn1_grad_gamma      = bn1_grad_ret[1];
  bn1_grad_beta       = bn1_grad_ret[2];

#ifdef TIMING
  time_b1 = getTime() - t_start;
  t_start = time_b1 + t_start;
#endif

#ifdef VERBOSE
  printf("running conv1 bwd\n");
#endif

  conv1_input.requires_grad_(true);
#ifdef BWD_D_ONLY
  conv1_grad_input = conv_bwd_d_ext(cfg.conv1,  {bn1_grad_input, conv1_input, conv1_weight}, {h1_block, w1_block, c1_block, k1_block, h1_in_gemm}, cd1_string, tuning_timings_d1);
  conv1_grad_weight = dummy_return;
#elif defined(BWD_W_ONLY)
  conv1_grad_input = at::empty(conv1_input.sizes(), torch::TensorOptions().dtype(conv1_input.dtype()));//dummy_return;
  conv1_grad_weight = conv_bwd_w_ext(cfg.conv1,  {bn1_grad_input, conv1_input, conv1_weight},
                                     {p1_block, c1_use_nchw_format, pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                          bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                            use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams},
                                     cw1_string, tuning_timings_w1);
#else
  auto conv1_grad_ret = conv_bwd_ext(cfg.conv1,  {bn1_grad_input, conv1_input, conv1_weight},
                                     {h1_block, w1_block, c1_block, k1_block, h1_in_gemm}, cd1_string, tuning_timings_d1,
                                     {p1_block, c1_use_nchw_format, pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                          bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                            use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams},
                                     cw1_string, tuning_timings_w1);
  //conv_backward_new(cfg.conv1, bn1_grad_input /*grad_output*/, conv1_input, conv1_weight);
  conv1_grad_input    = conv1_grad_ret[0];
  conv1_grad_weight   = conv1_grad_ret[1];
#endif

#ifdef TIMING
  time_c1 = getTime() - t_start;
  t_start = time_c1 + t_start;
#endif

  if (cfg.has_residual_conv) {

#ifdef VERBOSE
    printf("running bn4 bwd\n");
#endif

    bool bn4_relu = false, bn4_eltwise = false;
    //auto bn4_grad_ret   = batchnorm_bwd(bn4_relu, bn4_eltwise, cfg.bn_eps, {0, 0, 0, 0}, {bn3_grad_input_add, conv4_out, bn4_weight, bn4_mean, bn4_var, bn4_relu_out, bn4_scratch});
#ifdef BWD_D_ONLY
  std::array<std::string, 2> matched_bwd_strings4 = parse_conv_loop_string_for_batchnorm(cd4_string.c_str(), 1 /* is_nckhwrs */, 1 /* is nchw */);
#elif defined(BWD_W_ONLY)
  std::array<std::string, 2> matched_bwd_strings4 = parse_conv_loop_string_for_batchnorm(cw4_string.c_str(), 0 /* is_nckhwrs */, c4_use_nchw_format /* is nchw */);
#else
  std::array<std::string, 2> matched_bwd_strings4 = parse_conv_loop_string_for_batchnorm(cd4_string.c_str(), 1 /* is_nkchwrs */, 1 /* is nchw */);
#endif
    auto bn4_grad_ret   = batchnorm_bwd_ext(bn4_relu, bn4_eltwise, cfg.bn_eps, {0, 0, 0, 0}, matched_bwd_strings4[0], matched_bwd_strings4[1],
                                            {bn3_grad_input_add, conv4_out, bn4_weight, bn4_mean, bn4_var, bn4_relu_out, bn4_scratch});
    //bnorm_backward_new(cfg.bn4, bn3_grad_input_add /*grad_output*/, conv4_out, dummy_add, bn4_weight, dummy_output /*output*/, bn4_mean, bn4_var, dummy_invstd, bn4_relu_out);
    auto bn4_grad_input = bn4_grad_ret[0];
    bn4_grad_gamma      = bn4_grad_ret[1];
    bn4_grad_beta       = bn4_grad_ret[2];

#ifdef TIMING
    time_b4 = getTime() - t_start;
    t_start = time_b4 + t_start;
#endif

#ifdef VERBOSE
    printf("running conv4 bwd\n");
#endif

    conv1_input.requires_grad_(true);
#ifdef BWD_D_ONLY
    auto conv4_grad_input = conv_bwd_d_ext(cfg.conv4, {bn4_grad_input, conv1_input, conv4_weight}, {h4_block, w4_block, c4_block, k4_block, h4_in_gemm}, cd4_string, tuning_timings_d4);
    conv4_grad_weight = dummy_return;
#elif defined(BWD_W_ONLY)
    auto conv4_grad_input = at::empty(conv1_input.sizes(), torch::TensorOptions().dtype(conv1_input.dtype()));//dummy_return;
    conv4_grad_weight = conv_bwd_w_ext(cfg.conv4, {bn4_grad_input, conv1_input, conv4_weight},
                                       {p4_block, c4_use_nchw_format, pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                          bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                            use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams},
                                        cw4_string, tuning_timings_w4);
#else
    auto conv4_grad_ret = conv_bwd_ext(cfg.conv4, {bn4_grad_input, conv1_input, conv4_weight},
                                         {h4_block, w4_block, c4_block, k4_block, h4_in_gemm}, cd4_string, tuning_timings_d4,
                                         {p4_block, c4_use_nchw_format, pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                                            bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
                                              use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams},
                                         cw4_string, tuning_timings_w4);
    //conv_backward_new(cfg.conv4, bn4_grad_input /*grad_output*/, conv1_input, conv4_weight);
    conv4_grad_input    = conv4_grad_ret[0];
    conv4_grad_weight   = conv4_grad_ret[1];
#endif

#ifdef TIMING
    time_c4 = getTime() - t_start;
    t_start = time_c4 + t_start;
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
#ifndef BWD_W_ONLY
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
#endif
    //printf("in fused btlnk bwd adding %f %f %f %f (c1 to c4)\n", ptr_d1[0], ptr_d2[0], ptr_d3[0], ptr_d4[0]);
    ptr[4] += time_b1;
    ptr[5] += time_b2;
    ptr[6] += time_b3;
    ptr[7] += time_b4;
#ifndef BWD_D_ONLY
    auto buf_w1 = tuning_timings_w1.request();
    float* ptr_w1 = (float*)buf_w1.ptr;
    auto buf_w2 = tuning_timings_w2.request();
    float* ptr_w2 = (float*)buf_w2.ptr;
    auto buf_w3 = tuning_timings_w3.request();
    float* ptr_w3 = (float*)buf_w3.ptr;
    auto buf_w4 = tuning_timings_w4.request();
    float* ptr_w4 = (float*)buf_w4.ptr;
    ptr[8] += ptr_w1[0];
    ptr[9] += ptr_w2[0];
    ptr[10] += ptr_w3[0];
    ptr[11] += ptr_w4[0];
#endif
  }

  //printf("dbg: tuning_timings at the end of bf_fwd_ext = %f %f %f (time_c1 - c3 = %f %f %f)", ptr[0], ptr[1], ptr[2], time_c1, time_c2, time_c3);
#endif

#define MB (1024.0*1024.0)
#define GB (1024.0*1024.0*1024.0)

#if 1
        int training = 1;
        printf("perfdebug: checking for bottleneck in bwd with cfg C K H W stride: %d %d %d %d %d\n", cfg.inplanes, cfg.planes, cfg.H, cfg.W, cfg.stride);
/*
        printf("activation size (in Mb, per core): (inp = c4_in -> c1 out = c2_in (stride) -> c2_out = c3_in -> c3_out = c4_out %f %f %f %f \n",
                                                                   (cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) / MB,
                                                                   (cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) / MB,
                                                                   (cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) / MB,
                                                                   (4*cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) / MB );
        double c1_ab_size = ((cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(cfg.planes)*1*1*sizeof(T)) / MB;
        double c2_ab_size = ((cfg.planes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.planes)*(cfg.planes)*3*3*sizeof(T)) / MB;
        double c3_ab_size = ((cfg.planes)*(cfg.H / cfg.stride)*(cfg.W / cfg.stride)*sizeof(T) + (cfg.planes)*(4*cfg.planes)*1*1*sizeof(T)) / MB;
        double c4_ab_size = ((cfg.inplanes)*(cfg.H)*(cfg.W)*sizeof(T) + (cfg.inplanes)*(4*cfg.planes)*1*1*sizeof(T)) / MB;
        printf("conv input footprint (inp + weights) (in Mb, per core): %f %f %f %f (c1, c2, c3, c4)\n",
                                                                   c1_ab_size,
                                                                   c2_ab_size,
                                                                   c3_ab_size,
                                                                   c4_ab_size );
*/

        //(2.0*(double)cfg.N*(double)cfg.C*(double)cfg.K*(double)cfg.R*(double)cfg.S*(double)cfg.ofh*(double)cfg.ofw)/(1000*1000*1000)
        double c1_gflop = 2.0 * (2.0*(double)cfg.N*(double)cfg.inplanes*(double)cfg.planes*(double)1*(double)1*(double)cfg.H*(double)cfg.W)/(1000*1000*1000);
        double c2_gflop = 2.0 * (2.0*(double)cfg.N*(double)cfg.planes*(double)cfg.planes*(double)3*(double)3*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        double c3_gflop = 2.0 * (2.0*(double)cfg.N*(double)cfg.planes*(double)4*cfg.planes*(double)1*(double)1*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        double c4_gflop = 2.0 * (2.0*(double)cfg.N*(double)cfg.inplanes*(double)4*cfg.planes*(double)1*(double)1*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
        //printf("theoretical total conv flop: %f %f %f %f (c1, c2, c3, c4)\n", c1_gflop, c2_gflop, c3_gflop, c4_gflop);

        printf("PERFDUMP,BP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.inplanes), (cfg.planes)  , (cfg.H), (cfg.W), time_c1, c1_gflop / time_c1);
        printf("PERFDUMP,BP,resnetconv,%d,%d,%d,%d,%d,%d,3,3,%d,1,1,%f,%f\n", (cfg.N), (cfg.N), (cfg.planes),   (cfg.planes)  , (cfg.H), (cfg.W), cfg.stride, time_c2, c2_gflop / time_c2);
        printf("PERFDUMP,BP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.planes),   (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), time_c3, c3_gflop / time_c3);
        if (cfg.has_residual_conv)
            printf("PERFDUMP,BP,resnetconv,%d,%d,%d,%d,%d,%d,1,1,%d,0,0,%f,%f\n", (cfg.N), (cfg.N), (cfg.inplanes), (4*cfg.planes), (cfg.H), (cfg.W), (cfg.stride), time_c4, c4_gflop / time_c4);

        printf("PERFDUMP,BP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (cfg.planes)  , (cfg.planes)  , (cfg.H)             , (cfg.W)             , "na", "na", "na", (0), (1), time_b1, 1.0, (1), (0), (training));
        printf("PERFDUMP,BP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (cfg.planes)  , (cfg.planes)  , (cfg.H / cfg.stride), (cfg.W / cfg.stride), "na", "na", "na", (1), (0), time_b2, 1.0, (1), (0), (training));
        printf("PERFDUMP,BP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,%f,%d,%d,%d\n", (cfg.N), (cfg.N), (4*cfg.planes), (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), "na", "na", "na", (0), (0), time_b3, 1.0, (1), (1), (training));
        if (cfg.has_residual_conv)
            printf("PERFDUMP,BP,resnetbn,%d,%d,%d,%d,%d,%d,%s,%s,%s,%d,%d,%f,1.0,%d,%d,%d\n", (cfg.N), (cfg.N), (4*cfg.planes), (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride)                 , "na", "na", "na", (0), (0), time_b4, 1.0, (0), (0), (training));

#ifdef TIMING
    {
    auto buf_d1 = tuning_timings_d1.request();
    float* ptr_d1 = (float*)buf_d1.ptr;
    auto buf_d2 = tuning_timings_d2.request();
    float* ptr_d2 = (float*)buf_d2.ptr;
    auto buf_d3 = tuning_timings_d3.request();
    float* ptr_d3 = (float*)buf_d3.ptr;
    auto buf_d4 = tuning_timings_d4.request();
    float* ptr_d4 = (float*)buf_d4.ptr;

        printf("PERFDUMP,BP,resnetconv_d,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.inplanes), (cfg.planes)  , (cfg.H), (cfg.W), ptr_d1[0], c1_gflop / ptr_d1[0]);
        printf("PERFDUMP,BP,resnetconv_d,%d,%d,%d,%d,%d,%d,3,3,%d,1,1,%f,%f\n", (cfg.N), (cfg.N), (cfg.planes),   (cfg.planes)  , (cfg.H), (cfg.W), cfg.stride, ptr_d2[0], c2_gflop / ptr_d2[0]);
        printf("PERFDUMP,BP,resnetconv_d,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.planes),   (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), ptr_d3[0], c3_gflop / ptr_d3[0]);
        if (cfg.has_residual_conv)
            printf("PERFDUMP,BP,resnetconv_d,%d,%d,%d,%d,%d,%d,1,1,%d,0,0,%f,%f\n", (cfg.N), (cfg.N), (cfg.inplanes), (4*cfg.planes), (cfg.H), (cfg.W), (cfg.stride), ptr_d4[0], c4_gflop / ptr_d4[0]);
    auto buf_w1 = tuning_timings_w1.request();
    float* ptr_w1 = (float*)buf_w1.ptr;
    auto buf_w2 = tuning_timings_w2.request();
    float* ptr_w2 = (float*)buf_w2.ptr;
    auto buf_w3 = tuning_timings_w3.request();
    float* ptr_w3 = (float*)buf_w3.ptr;
    auto buf_w4 = tuning_timings_w4.request();
    float* ptr_w4 = (float*)buf_w4.ptr;

        printf("PERFDUMP,BP,resnetconv_w,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.inplanes), (cfg.planes)  , (cfg.H), (cfg.W), ptr_w1[0], c1_gflop / ptr_w1[0]);
        printf("PERFDUMP,BP,resnetconv_w,%d,%d,%d,%d,%d,%d,3,3,%d,1,1,%f,%f\n", (cfg.N), (cfg.N), (cfg.planes),   (cfg.planes)  , (cfg.H), (cfg.W), cfg.stride, ptr_w2[0], c2_gflop / ptr_w2[0]);
        printf("PERFDUMP,BP,resnetconv_w,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.planes),   (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), ptr_w3[0], c3_gflop / ptr_w3[0]);
        if (cfg.has_residual_conv)
            printf("PERFDUMP,BP,resnetconv_w,%d,%d,%d,%d,%d,%d,1,1,%d,0,0,%f,%f\n", (cfg.N), (cfg.N), (cfg.inplanes), (4*cfg.planes), (cfg.H), (cfg.W), (cfg.stride), ptr_w4[0], c4_gflop / ptr_w4[0]);

    }
#endif

#endif


  return {conv1_grad_weight, conv2_grad_weight, conv3_grad_weight, conv4_grad_weight,
          bn1_grad_gamma, bn2_grad_gamma, bn3_grad_gamma, bn4_grad_gamma,
          bn1_grad_beta, bn2_grad_beta, bn3_grad_beta, bn4_grad_beta,
          conv1_grad_input, conv4_grad_input};


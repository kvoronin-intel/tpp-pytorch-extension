RECORD_FUNCTION("fused_bottleneck_bn_fwd", std::vector<c10::IValue>());

//#define VERBOSE

  auto input        = inputs[0];
  auto conv1_weight = inputs[1];
  auto conv2_weight = inputs[2];
  auto conv3_weight = inputs[3];
  auto conv4_weight = inputs[4];
  auto conv1_bias   = inputs[5];
  auto conv2_bias   = inputs[6];
  auto conv3_bias   = inputs[7];
  auto conv4_bias   = inputs[8];

  const int h1_block = tuning_params[0];
  const int w1_block = tuning_params[1];
  const int h2_block = tuning_params[2];
  const int w2_block = tuning_params[3];
  const int h3_block = tuning_params[4];
  const int w3_block = tuning_params[5];
  const int h4_block = tuning_params[6];
  const int w4_block = tuning_params[7];
  const int c1_block = tuning_params[8];
  const int k1_block = tuning_params[9];
  const int c2_block = tuning_params[10];
  const int k2_block = tuning_params[11];
  const int c3_block = tuning_params[12];
  const int k3_block = tuning_params[13];
  const int c4_block = tuning_params[14];
  const int k4_block = tuning_params[15];
  const int h1_in_gemm = tuning_params[16];
  const int h2_in_gemm = tuning_params[17];
  const int h3_in_gemm = tuning_params[18];
  const int h4_in_gemm = tuning_params[19];
  const int pack_input_for_1x1_strided = tuning_params[20];
  //const int fuse_stats = tuning_params[21];
  //int fuse_stats = tuning_params[21];

  const std::string c1_string = tuning_strings[0];
  const std::string c2_string = tuning_strings[1];
  const std::string c3_string = tuning_strings[2];
  const std::string c4_string = tuning_strings[3];

  std::vector<long> dummy_size{0};
  auto dummy_return = at::zeros(dummy_size, input.options());

#ifdef TIMING
  double t_start_all = 0.0;
  t_start_all = getTime();
  double time_c1 = 0.0, time_c2 = 0.0, time_c3 = 0.0, time_c4 = 0.0;

  const int n_used_timers = 6;

  pybind11::array_t<float> c1_tuning_timings(n_used_timers), c2_tuning_timings(n_used_timers), c3_tuning_timings(n_used_timers), c4_tuning_timings(n_used_timers);

  {
    float *ptr_c1 = c1_tuning_timings.mutable_data();
    for (int i = 0; i < n_used_timers; i++)
        ptr_c1[i] = 0.0;
    float *ptr_c2 = c2_tuning_timings.mutable_data();
    for (int i = 0; i < n_used_timers; i++)
        ptr_c2[i] = 0.0;
    float *ptr_c3 = c3_tuning_timings.mutable_data();
    for (int i = 0; i < n_used_timers; i++)
        ptr_c3[i] = 0.0;
    float *ptr_c4 = c4_tuning_timings.mutable_data();
    for (int i = 0; i < n_used_timers; i++)
        ptr_c4[i] = 0.0;
  }

#endif

#ifdef VERBOSE
  printf("running conv1 \n");
  std::cout << "input dtype = " << input.dtype() << std::endl;
#endif

  auto conv1_out = conv_fwd_ext(cfg.conv1, {input, conv1_weight, conv1_bias}, {h1_block, w1_block, c1_block, k1_block, h1_in_gemm, pack_input_for_1x1_strided}, c1_string, c1_tuning_timings);

#ifdef VERBOSE
  printf("running conv2 \n");
#endif

  auto conv2_out = conv_fwd_ext(cfg.conv2, {conv1_out, conv2_weight, conv2_bias}, {h2_block, w2_block, c2_block, k2_block, h2_in_gemm, pack_input_for_1x1_strided}, c2_string, c2_tuning_timings);

#ifdef VERBOSE
  printf("running conv3 \n");
#endif

  at::Tensor residual;
  if (cfg.has_residual_conv) {
#ifdef VERBOSE
    printf("running conv4 \n");
#endif
    residual = conv_fwd_ext(cfg.conv4, {input, conv4_weight, conv4_bias}, {h4_block, w4_block, c4_block, k4_block, h4_in_gemm, pack_input_for_1x1_strided}, c4_string, c4_tuning_timings);
  } else {
    residual = input;
  }

  auto conv3_out = conv_fwd_ext(cfg.conv3, {conv2_out, conv3_weight, conv3_bias, residual}, {h3_block, w3_block, c3_block, k3_block, h3_in_gemm, pack_input_for_1x1_strided}, c3_string, c3_tuning_timings);

#ifdef TIMING
  {
    double c1_gflop = 2.0 * (2.0*(double)cfg.N*(double)cfg.inplanes*(double)cfg.planes*(double)1*(double)1*(double)cfg.H*(double)cfg.W)/(1000*1000*1000);
    double c2_gflop = 2.0 * (2.0*(double)cfg.N*(double)cfg.planes*(double)cfg.planes*(double)3*(double)3*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
    double c3_gflop = 2.0 * (2.0*(double)cfg.N*(double)cfg.planes*(double)4*cfg.planes*(double)1*(double)1*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);
    double c4_gflop = 2.0 * (2.0*(double)cfg.N*(double)cfg.inplanes*(double)4*cfg.planes*(double)1*(double)1*(double)(cfg.H/cfg.stride)*(double)(cfg.W/cfg.stride))/(1000*1000*1000);

    auto buf_c1 = c1_tuning_timings.request();
    float* ptr_c1 = (float*)buf_c1.ptr;
    auto buf_c2 = c2_tuning_timings.request();
    float* ptr_c2 = (float*)buf_c2.ptr;
    auto buf_c3 = c3_tuning_timings.request();
    float* ptr_c3 = (float*)buf_c3.ptr;
    auto buf_c4 = c4_tuning_timings.request();
    float* ptr_c4 = (float*)buf_c4.ptr;

    printf("PERFDUMP,INF,resnetconv,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.inplanes), (cfg.planes)  , (cfg.H), (cfg.W), ptr_c1[0], c1_gflop / ptr_c1[0]);
    printf("PERFDUMP,INF,resnetconv,%d,%d,%d,%d,%d,%d,3,3,%d,1,1,%f,%f\n", (cfg.N), (cfg.N), (cfg.planes),   (cfg.planes)  , (cfg.H), (cfg.W), cfg.stride, ptr_c2[0], c2_gflop / ptr_c2[0]);
    printf("PERFDUMP,INF,resnetconv,%d,%d,%d,%d,%d,%d,1,1,1,0,0,%f,%f\n",  (cfg.N), (cfg.N), (cfg.planes),   (4*cfg.planes), (cfg.H / cfg.stride), (cfg.W / cfg.stride), ptr_c3[0], c3_gflop / ptr_c3[0]);
    if (cfg.has_residual_conv)
      printf("PERFDUMP,INF,resnetconv,%d,%d,%d,%d,%d,%d,1,1,%d,0,0,%f,%f\n", (cfg.N), (cfg.N), (cfg.inplanes), (4*cfg.planes), (cfg.H), (cfg.W), (cfg.stride), ptr_c4[0], c4_gflop / ptr_c4[0]);
  }
#endif

  return conv3_out;


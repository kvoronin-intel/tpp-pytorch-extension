{

#ifdef TIMING
t_start = getTime();
#endif

//#define VERBOSE

// t_CI and t_CW should be defined outside
// t_BW, t_BB, t_BM, t_BV, t_BIA should be defined outside
// t_BW_prev, t_BB_prev, t_BM_prev, t_BV_prev, t_relu_mask_prev, eltwise_prev must be defined outside if fuse_scaling = 1
// h_block, w_block, c_block, k_block, h_in_gemm, pack_input and fuse_stats must be defined outside

auto sizes = t_CI.sizes();

//const int fuse_stats = ?
const int separate_stats_reduction = 1; /* only value currently supported is 1 */
//const int fuse_scaling = 0; /* must be defined in the calling code */

char conv_fwd_loop_specs_str[256];
std::strcpy(conv_fwd_loop_specs_str, conv_loop_string.c_str());

#ifdef NO_BATCHNORM
fuse_stats = 0;
#endif

#ifdef VERBOSE
std::cout << "CONV+BN meta setup info"           << std::endl;
std::cout << "fuse_stats    = " << fuse_stats    << std::endl;
std::cout << "fuse_scaling  = " << fuse_scaling  << std::endl;
std::cout << "tuning_params = " << tuning_params << std::endl;
std::cout << "pack_input (on entry) = " << pack_input  << std::endl;
std::cout << "conv_fwd_loop_specs_str = " << conv_fwd_loop_specs_str << std::endl;
#endif

int R = conv_cfg.R;
int S = conv_cfg.S;
int ofh = conv_cfg.ofh;
int ofw = conv_cfg.ofw;
int ifhp = conv_cfg.ifhp;
int ifwp = conv_cfg.ifwp;
int ofhp = conv_cfg.ofhp;
int ofwp = conv_cfg.ofwp;
int bk = conv_cfg.bk;
int bc = conv_cfg.bc;
int stride_h = conv_cfg.u;
int stride_w = conv_cfg.v;
int Cb = conv_cfg.blocksifm;
int Kb = conv_cfg.blocksofm;

int conv_pad_h_out = conv_cfg.pad_h_out;
int conv_pad_w_out = conv_cfg.pad_w_out;
int conv_pad_h = conv_cfg.pad_h;
int conv_pad_w = conv_cfg.pad_w;

const long N  = sizes[0];

std::vector<long> conv_output_size{N, Kb, ofhp, ofwp, bk};

/* in T */
const long conv_fwd_scratch_size = (pack_input == 0 ? 0 : N*ofh*ofw*conv_cfg.C);
auto t_scratch_conv = at::empty({conv_fwd_scratch_size}, torch::TensorOptions().dtype(t_CI.dtype()));

#ifdef VERBOSE
std::cout << "CONV setup info" << std::endl;
std::cout << "t_CI sizes = " << t_CI.sizes() << std::endl;
std::cout << "size of T = " << sizeof(T) << std::endl;
std::cout << "conv_output_size = " << conv_output_size << std::endl;
std::cout << "Cb bc Kb bk = " << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;
std::cout << "stride_h stride_w = " << stride_h << " " << stride_w << std::endl;
std::cout << "scratch size = " << conv_fwd_scratch_size << std::endl;
std::cout << "conv_pad_h_out conv_pad_w_out = " << conv_pad_h_out << " " << conv_pad_w_out << std::endl;
#endif

CONV_OUT = at::empty(conv_output_size, torch::TensorOptions().dtype(t_CI.dtype()));
auto t_CO = CONV_OUT;

#ifndef NO_BATCHNORM
const long bn_pad_h_in  = bn_padding[0];
const long bn_pad_w_in  = bn_padding[1];
const long bn_pad_h_out = bn_padding[2];
const long bn_pad_w_out = bn_padding[3];

auto t_BI = BN_IN;

auto bn_sizes = t_BI.sizes();
//const long N  = sizes[0];
//const long Kb = sizes[1];
const long H  = bn_sizes[2] - 2 * bn_pad_h_in;
const long W  = bn_sizes[3] - 2 * bn_pad_w_in;
//const long bk = sizes[4];

const long hi_start      = bn_pad_h_in;
const long wi_start      = bn_pad_w_in;
const long bn_ifhp       = H + 2 * bn_pad_h_in;
const long bn_ifwp       = W + 2 * bn_pad_w_in;

const long ho_start      = bn_pad_h_out;
const long ho_end        = ho_start + H;
const long wo_start      = bn_pad_w_out;
const long wo_end        = wo_start + W;
const long bn_ofhp       = H + 2 * bn_pad_h_out;
const long bn_ofwp       = W + 2 * bn_pad_w_out;

const float scale = 1.0f /((float)N * H * W);

std::vector<long> bn_output_size  {N, Kb, bn_ofhp, bn_ofwp, bk};
std::vector<long> bn_relumask_size{N, Kb, bn_ofhp, bn_ofwp, bk/BITS_PER_CHAR};

BN_OUT = at::empty(bn_output_size, torch::TensorOptions().dtype(t_BI.dtype()));
auto t_BO = BN_OUT;

BN_RELU_OUT = at::empty(bn_relumask_size, torch::TensorOptions().dtype(at::kByte));
auto t_relu_mask = BN_RELU_OUT;

const long sum_N_offset          = LIBXSMM_UP2(Kb * 2 * bk, 64);
const long sumsq_N_offset        = LIBXSMM_UP2(sum_N_offset + Kb * N * bk, 64);

const long dbeta_N_offset        = LIBXSMM_UP2(Kb * N * bk, 64);

/* in floats */
const long full_fwd_scratch_size_in_floats = sumsq_N_offset + LIBXSMM_UP2((size_t)Kb * (size_t)N * (size_t)bk, 64);
const long full_bwd_scratch_size_in_floats = dbeta_N_offset + LIBXSMM_UP2(Kb * N * bk, 64);
const long full_scratch_size_in_floats     = std::max(full_fwd_scratch_size_in_floats, full_bwd_scratch_size_in_floats);
std::vector<long> scratch_size_in_floats{full_scratch_size_in_floats};
BN_SCRATCH_OUT = at::empty(scratch_size_in_floats, torch::TensorOptions().dtype(at::kFloat));
auto t_scratch_bn = BN_SCRATCH_OUT;

bool use_hw_blocking_in_fusion      = true;
bool use_hw_blocking_outside_fusion = true;

//const long num_HW_blocks = (H > W ? H : W);
//const long num_W_blocks  = (W % 64 == 0 ? W / 64 : 1);
long num_HW_blocks_outside_fusion = 0;
long num_W_blocks_outside_fusion = 0;

long spatial_block_size_in_fusion = 0, spatial_block_size_outside_fusion = 0;

if (fuse_stats == 1) {
  /* In fusion, hw- or w-blocking is chosen based on w_block value */
  if (w_block != 1) {
    use_hw_blocking_in_fusion    = false; /* w blocking ([w, bc] blocks) */
    spatial_block_size_in_fusion = ofw / w_block;
  } else {
    use_hw_blocking_in_fusion    = true; /* hw-blocking using [hw, bc] blocks */
    auto w_gemm_pixels = ofw/w_block; // == ofw
    spatial_block_size_in_fusion = (w_gemm_pixels +  2 * conv_pad_w) * (h_in_gemm - 2) + 2 * (w_gemm_pixels + conv_pad_w);
    //spatial_block_size_in_fusion = h_in_gemm * W;
  }
}

if (bn_pad_h_in != 0 || bn_pad_w_in != 0 || bn_pad_h_out != 0 || bn_pad_w_out != 0) {
  /* If padding is non-zero, batchnorm cannot do hw-blocking */
  use_hw_blocking_outside_fusion    = false; /* alternative is w blocking ([w, bc] blocks) */
  spatial_block_size_outside_fusion = ofw / w_block;
  num_W_blocks_outside_fusion       = W / spatial_block_size_outside_fusion;
} else {
  /* If there is no padding, one can choose */
  if (w_block != 1) {
    use_hw_blocking_outside_fusion    = false; /* w blocking ([w, bc] blocks) */
    spatial_block_size_outside_fusion = ofw / w_block;
    num_W_blocks_outside_fusion       = W / spatial_block_size_outside_fusion;
  } else {
    use_hw_blocking_outside_fusion    = true; /* hw-blocking using [hw, bc] blocks */
    spatial_block_size_outside_fusion = h_in_gemm * W; //or set it always to 1 * W ???
    num_HW_blocks_outside_fusion      = H * W / spatial_block_size_outside_fusion;
  }
}

#ifdef VERBOSE
std::cout << "BN setup info" << std::endl;
std::cout << "bn_padding = " << bn_padding << std::endl;
std::cout << "size of T = " << sizeof(T) << std::endl;
std::cout << "bn_output_size = " << bn_output_size << std::endl;
std::cout << "t_BI sizes = " << t_BI.sizes() << std::endl;
std::cout << "use_hw_blocking_in_fusion      = " << use_hw_blocking_in_fusion << " and spatial_block_size_in_fusion = " << spatial_block_size_in_fusion  << std::endl;
std::cout << "use_hw_blocking_outside_fusion = " << use_hw_blocking_outside_fusion << " and spatial_block_size_outside_fusion = " << spatial_block_size_outside_fusion
          << " and num blocks = " << (use_hw_blocking_outside_fusion ? num_HW_blocks_outside_fusion : num_W_blocks_outside_fusion) << std::endl;
#endif
#endif /* for #ifndef NO_BATCHNORM */

#ifdef VERBOSE
std::cout << "Setting up the conv in conv/bn fusion" << std::endl;
#endif
//return std::vector<at::Tensor>({t_CO});

  long avoid_fmas_in_rim = 0;
  if (ofh <= 7 && ofw <=7 && R == 3 && S == 3 && stride_w == 1 && stride_h == 1 && h_in_gemm == 1) {
    avoid_fmas_in_rim = 1;
  }

  if (R != 1 || S != 1) {
#ifdef VERBOSE
    std::cout << "Setting pack_input to zero for non 1x1 convs" << std::endl;
#endif
    pack_input = 0;
  }

  //auto gemm_n = ofw / w_block;
  auto w_gemm_pixels = ofw/w_block;
  auto gemm_n = (w_gemm_pixels +  2 * conv_pad_w) * (h_in_gemm - 2) + 2 * (w_gemm_pixels + conv_pad_w);
  auto gemm_m = bk;
  auto gemm_k = bc;

  long Cb_step = Cb / c_block;

  long n_step = 1;
  long c_step = Cb_step;
  long k_step = 1;
  long h_step = h_in_gemm;
  long w_step = ofw / w_block;
  long r_step = R;
  long s_step = S;

  if (avoid_fmas_in_rim == 1) {
    r_step = 1;
    s_step = 1;
  }

  //std::cout << "gemm_n gemm_m gemm_k = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;

  std::unique_ptr<unsigned long long[]> A_offsets, B_offsets;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>) brgemm_tpp, brgemm2_tpp;
  SCOPEIT_DECL(CpyTPP<T>)     input_pack_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>) zero_tpp;

  zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);
  /* n,m,k, stride_b, stride_a, ldb, lda, ldc, beta, a_trans, unroll_hint because of the row-major */
  float beta;
  if (Cb_step == Cb && r_step == R && s_step == S)
    beta = 0.0;
  else
    beta = 1.0;

  if ((R == 1 && S == 1) || (avoid_fmas_in_rim == 1)) {
    //brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n  , gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);
    if (pack_input == 0) {
      //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n  , gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, beta, 0, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);
    } else {
      if (avoid_fmas_in_rim) {
        printf("Error: avoid_fmas_in_rim = %d is incompatible with pack_input = %d\n", avoid_fmas_in_rim, pack_input);
        exit(-1);
      }
      if (R != 1 || S != 1) {
        printf("Error: R = %d and S = %d are incompatible with pack_input = %d\n", R, S, pack_input);
        exit(-1);
      }
      //auto l_pack_shape = libxsmm_create_meltw_unary_shape(bc, gemm_n, bc*stride_w, bc, dtype, dtype, dtype);
      //input_pack_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_pack_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
      //printf("input_pack_tpp\n");
      input_pack_tpp = SCOPEIT(CpyTPP<T>(w_gemm_pixels, bc, bc*stride_w, bc), EW_COPY); /* gemm_n, bc because of the row-major for unary */
      //l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc, bk, dtype, dtype, dtype, dtype );
      //l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bc*ofh*ofw*sizeof(DType), Cb_step );
      //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
      //printf("brgemm_tpp\n");
      brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, bc*ofh*ofw, R*S*bc*bk, bc, bk, bk, beta, 0, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);
    }

    //printf("brgemm2_tpp\n");
    brgemm2_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-1, gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, beta, 0, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);

  } else {
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* no strides due to reduce_offset */ bc*stride_w, bk, bk, beta, 0, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);

    A_offsets = std::make_unique<unsigned long long[]>(Cb * R * S);
    B_offsets = std::make_unique<unsigned long long[]>(Cb * R * S);

    // Prepare offset array
    unsigned long long i = 0;
    for (long ifm = 0; ifm < Cb_step; ifm++) {
      for (long kj = 0; kj < R; kj++) {
        for (long ki = 0; ki < S; ki++) {
          A_offsets[i] = (ifm * R * S * bc * bk +
              kj * S * bc * bk +
              ki * bc * bk) * sizeof(T);
          B_offsets[i] = (ifm * ifhp * ifwp * bc +
              kj * ifwp * bc +
              ki * bc) * sizeof(T);
          i++;
        }
      }
    } /* outer loop for filling the offsets */
  }


  if ( (h_in_gemm > 1) && (w_block != 1) ) {
    printf("Invalid input GEMM config: When multiple H pixels are handled in the gemm, then the full ofw should be also used as gemm_n...\n");
    exit(-1);
  }

#ifdef VERBOSE
  std::cout << "debug: N n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " " << n_step << " " << Cb << " " << c_step << " "
                                                                                                << Kb << " " << k_step << " " << ofh << " " << h_step << " "
                                                                                                << ofw << " " << w_step << " " << R << " " << r_step << " "
                                                                                                << S << " " << s_step << " " << std::endl;
  std::cout << "h_block w_block c_block k_block = " << h_block << " " << w_block << " " << c_block << " " << k_block << std::endl;
  std::cout << "h_in_gemm = " << h_in_gemm << std::endl;
  std::cout << "avoid fmas in rim = " <<  avoid_fmas_in_rim << std::endl;
  std::cout << "pack_input = " << pack_input << std::endl;
#endif

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, false},//, true},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, Kb, k_step, {k_block}},
      LoopSpecs{0, ofh, h_step, {h_block}},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      conv_fwd_loop_specs_str);


#ifdef VERBOSE
  printf("parlooper fwd string: OMP_NUM_THREADS=%d USE_BF16=%d ./run_conv_fwd.sh %s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d 1000\n", N, (sizeof(T) == 2 ? 1 : 0), conv_fwd_loop_specs_str,
                                        N, ifhp - 2 * conv_pad_h_out, ifwp - 2 * conv_pad_w_out, conv_cfg.C, conv_cfg.K, R, S, stride_h, stride_w, conv_pad_h_out, conv_pad_w_out,
                                        bc, bk, h_block, w_block, c_block, k_block, h_in_gemm, pack_input);

//200~--with-bwd --bc 64 --bk 64 --basic-sizes 28 7 7 2048 512 1 1 --tuning-params  1  1  0 0 0 0 0 0 0 7 4 --tuning-string Aefbcd --perf-bwd-w201~
  printf("conv_ext fwd string: python -u test_conv_ext.py --test-module ext_tpp %s --perf-fwd --bc %d --bk %d --basic-sizes %d %d %d %d %d %d %d --tuning-params %d %d %d %d %d %d --tuning-string %s --niters 1000 --niters-warmup 100 \n",
                                        (sizeof(T) == 2 ? "--use-bf16-opt" : ""), bc, bk,
                                        N, ifhp - 2 * conv_pad_h_out, ifwp - 2 * conv_pad_w_out, conv_cfg.C, conv_cfg.K, stride_h, R,
                                        h_block, w_block, c_block, k_block, h_in_gemm, pack_input,
                                        conv_fwd_loop_specs_str);
#endif

#ifndef NO_BATCHNORM
#ifdef VERBOSE
std::cout << "Setting up the bn in conv/bn fusion" << std::endl;
#endif

  auto zero_bk_tpp = SCOPEIT(SetZeroTPP<float>(bk), EW_ZERO);

  auto zero_padbk_T_tpp = SCOPEIT(SetZeroTPP<T>(2*conv_pad_w_out*bk), EW_ZERO);

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bk, bk, bk), EW_ADD); /* 1, bc because of row-major for unary */

  auto reduce_beta0_fusion_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size_in_fusion, bk, bk, bk, 1)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto reduce_beta1_fusion_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size_in_fusion, bk, bk, bk, 0)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto reduce_beta0_nonfusion_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size_outside_fusion, bk, bk, bk, 1)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto reduce_beta1_nonfusion_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size_outside_fusion, bk, bk, bk, 0)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto mean_var_tpp = SCOPEIT(MeanVarTPP<float>(bk, scale), EW_MEAN_VAR);

  auto coeffs_tpp = SCOPEIT(BatchNormStatCoeffsTPP<float>(bk, eps), NORMALIZE);

  auto zero_hp_tpp = SCOPEIT(SetZeroTPP<T>((bn_pad_h_out * bn_ofwp), bk, bk), EW_ZERO); /* (pad_h_out * ofwp), bc because of row-major for unary */

  auto zero_wp_tpp = SCOPEIT(SetZeroTPP<T>(bn_pad_w_out, bk, bk), EW_ZERO);          /* pad_w_out, bc because of row-major for unary */

  auto normalize_tpp = SCOPEIT((BatchNormFwdScaleTPP<T,T>(bk, spatial_block_size_outside_fusion, relu, eltwise)), NORMALIZE);

  char kb_loop_specs_str[256];
  char nkb_loop_specs_str[256];// = "AB";
  int A_seen = 0, C_seen = 0;
  for (int i = 0; i < strlen(conv_fwd_loop_specs_str); i++) {
      if (conv_fwd_loop_specs_str[i] == 'A')
        A_seen++;
      else if (conv_fwd_loop_specs_str[i] == 'C')
        C_seen++;
  }
  if (A_seen && C_seen)
    strcpy(nkb_loop_specs_str, "AB");
  else if (A_seen && !C_seen)
    strcpy(nkb_loop_specs_str, "Ab");
  else if (!A_seen && C_seen)
    strcpy(nkb_loop_specs_str, "Ba");
  else
    strcpy(nkb_loop_specs_str, "ab");

  if (C_seen)
    strcpy(kb_loop_specs_str, "A");
  else
    strcpy(kb_loop_specs_str, "a");

#ifdef VERBOSE
  std::cout << "nkb_loop_specs_str = " << nkb_loop_specs_str << std::endl;
  std::cout << "kb_loop_specs_str  = " << kb_loop_specs_str  << std::endl;
#endif

  const long bn_n_step = 1, bn_kb_step = 1;
  auto nkb_loop = ThreadedLoop<2>({
      LoopSpecs{0, N,  bn_n_step,  {/*l1_n_step, l0_n_step*/}},   // Logical N  loop specs
      LoopSpecs{0, Kb, bn_kb_step, {/*l1_k_step, l0_k_step*/}}},  // Logical Kb loop specs
      nkb_loop_specs_str);

  //char kb_loop_specs_str[256] = "A";
  auto kb_loop = ThreadedLoop<1>({
      LoopSpecs{0, Kb, bn_kb_step, {/*l1_k_step, l0_k_step*/}}},  // Logical Kb loop specs
      kb_loop_specs_str);
#endif /* #ifndef NO_BATCHNORM */

#ifdef VERBOSE
std::cout << "Running conv part in conv/bn fusion" << std::endl;
#endif

#ifdef TIMING
  t_conv_start = getTime();
#endif

//for (int i = 0; i < 1000; i++) {


  {
#ifdef NO_BATCHNORM
    RECORD_SCOPE(fusedbtlnk_conv_nobatchnorm_fwd, {});
#else
    RECORD_SCOPE(fusedbtlnk_conv_fwd, {});
#endif
    {
      conv_loop(
        [&](int* ind) {
          int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

          DECL_VLA_PTR_PT_EXT(T,     output_off, [Kb][ofhp][ofwp][bk],   t_CO, (conv_pad_h_out * ofwp * bk + conv_pad_w_out * bk));
          DECL_VLA_PTR_PT    (T,     inp,        [Cb][ifhp][ifwp][bc],   t_CI);
          DECL_VLA_PTR_PT    (T,     weight,     [Cb][R][S][bc][bk],     t_CW);

          if (avoid_fmas_in_rim == 0) {

            if (Cb_step != Cb || r_step != R || s_step != S) {
              if (i_c == 0 && i_r == 0 && i_s == 0) {
                zero_tpp(output_off[i_n][i_k][i_h][i_w]);
              }
            }

            if (fuse_scaling && i_k == 0 && i_r == 0 && i_s == 0) {
              printf("fuse_scaling = 1 has not been implemented yet\n");
              exit(-1);
            } /* if fuse_scaling + extra conditions */

            if (pack_input > 0 && i_r == 0 && i_s == 0 && i_k == 0 && i_c == 0) {
              //DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);
              DECL_VLA_PTR_PT(T, packed_inp, [Cb][ofh][ofw][bc], t_scratch_conv);
              //libxsmm_blasint _br, _h;
              for (int _br = 0; _br < Cb; _br++) {
                for (int _h = 0; _h < h_step; _h++) {
                  //libxsmm_meltw_unary_param pack_param;
                  //pack_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, _br, (i_h+_h) * stride_h, i_w * stride_w, 0, Cb, ifhp, ifwp, bc);
                  //pack_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), packed_input_libxsmm, i_n, _br, i_h+_h, i_w, 0, Cb, ofh, ofw, bc);
                  //input_pack_kernel( &pack_param );
                  input_pack_tpp(inp[i_n][_br][(i_h+_h)*stride_h][i_w * stride_w], packed_inp[i_n][_br][i_h+_h][i_w]);
                }
              }
            }

            if (pack_input == 0) {
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
              brgemm_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         weight    [i_k][i_c][i_r][i_s][0],
                         output_off[i_n][i_k][i_h]                 [i_w],
                         B_offsets.get(), A_offsets.get(),
                         Cb_step * r_step * s_step,
                         true);
            } else {
              //DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);
              DECL_VLA_PTR_PT(T, packed_inp, [Cb][ofh][ofw][bc], t_scratch_conv);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), packed_input_libxsmm, i_n, i_c, i_h, i_w, 0, Cb, ofh, ofw, bc);
              brgemm_tpp(packed_inp[i_n][i_c][i_h][i_w],
                         weight    [i_k][i_c][i_r][i_s][0],
                         output_off[i_n][i_k][i_h]                 [i_w],
                         B_offsets.get(), A_offsets.get(),
                         Cb_step * r_step * s_step,
                         true);
            }

#ifndef NO_BATCHNORM
            /* Computing local stats */
            if (training && fuse_stats && i_c == Cb - c_step && i_r == R - r_step && i_s == S - s_step) {
              DECL_VLA_PTR_PT_EXT(float, sums_N,    [N][2*bk],             t_scratch_bn, sum_N_offset);

              /* Zeroing out the rims which are glued together due to the h_in_gemm != 1 for non 1x1 convs */
              if (R > 1 && h_in_gemm > 1) {
                for (int _h = 0; _h < h_in_gemm - 1; _h++) {
                  zero_padbk_T_tpp(output_off[i_n][i_k][i_h + _h][ofw + 0]);
                  //zero_bk_T_tpp(output_off[i_n][i_k][i_h + _h][ofw + conv_pad_w_out]);
                }
              }

              if (i_h == 0 && i_w == 0)
                reduce_beta0_fusion_tpp(output_off[i_n][i_k][i_h][i_w], sums_N[i_k][i_n]);
              else
                reduce_beta1_fusion_tpp(output_off[i_n][i_k][i_h][i_w], sums_N[i_k][i_n]);

            } /* for computing local stats */
#endif /*#ifndef NO_BATCHNORM */
          } else { /* for if avoid_fmas_in_rim == 0 */

            if (fuse_scaling && i_k == 0 && i_r == 0 && i_s == 0) {
              printf("fuse_scaling = 1 has not been implemented yet for avoid_fmas_in_rim != 0\n");
              exit(-1);
            } /* if fuse_scaling + extra conditions */

            if (Cb_step != Cb || r_step != R || s_step != S) {
              if (i_c == 0 && i_r == 0 && i_s == 0) {
                zero_tpp(output_off[i_n][i_k][i_h][i_w]);
              }
            }

            if (i_r == 0 && i_h == 0) {
              /* Do no FLOPS  */
            } else if (i_r == R - r_step && i_h == ofh - h_step ) {
              /* Do no FLOPS  */
            } else if ( i_w == 0 && i_s == 0 ) {
              brgemm2_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s + 1],
                          weight    [i_k][i_c][i_r][i_s][0],
                          output_off[i_n][i_k][i_h]                 [i_w + 1],
                          Cb_step,
                          true);
            } else if ( i_w + w_step == ofw  && i_s == S - s_step) {
              brgemm2_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                          weight    [i_k][i_c][i_r][i_s][0],
                          output_off[i_n][i_k][i_h]                 [i_w],
                          Cb_step,
                          true);
            } else {
              brgemm_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         weight    [i_k][i_c][i_r][i_s][0],
                         output_off[i_n][i_k][i_h]                 [i_w],
                         Cb_step,
                         true);
            }

#ifndef NO_BATCHNORM
            /* Computing local stats */
            if (training && fuse_stats && i_c == Cb - c_step && i_r == R - r_step && i_s == S - s_step) {
              DECL_VLA_PTR_PT_EXT(float, sums_N,    [N][2*bk],             t_scratch_bn, sum_N_offset);

              /* Zeroing out the rims which are glued together due to the h_in_gemm != 1 for non 1x1 convs */
              if (R > 1 && h_in_gemm > 1) {
                for (int _h = 0; _h < h_in_gemm - 1; _h++) {
                  zero_padbk_T_tpp(output_off[i_n][i_k][i_h + _h][ofw + 0]);
                  //zero_bk_T_tpp(output_off[i_n][i_k][i_h + _h][ofw + conv_pad_w_out]);
                }
              }

              if (i_h == 0 && i_w == 0)
                reduce_beta0_fusion_tpp(output_off[i_n][i_k][i_h][i_w], sums_N[i_k][i_n]);
              else
                reduce_beta1_fusion_tpp(output_off[i_n][i_k][i_h][i_w], sums_N[i_k][i_n]);
            } /* for computing local stats */
#endif /* #ifndef NO_BATCHNORM */
          } /* for if-else avoid_fmas_in_rim == 0 */
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});
    } /* end of the fusedbtlnk_conv_fwd scope with recorded parallel for */
  } /* end of the dummy scope */

#ifdef TIMING
  t_conv_end = getTime();
#endif

#ifndef NO_BATCHNORM

#ifdef VERBOSE
std::cout << "Running bn part in conv/bn fusion" << std::endl;
#endif

  if (training) {
    if (!fuse_stats) {
#ifdef VERBOSE
      std::cout << "Running the standalone partial stats reduce loops" << std::endl;
#endif
      RECORD_SCOPE(fusedbtlnk_bn_fwd_reduce, {});
      {
        nkb_loop(
          [&](int *ind) {
            const int n = ind[0], kb = ind[1];

            DECL_VLA_PTR_PT_EXT(T,     inp,      [Kb][bn_ifhp][bn_ifwp][bk], t_BI, (hi_start * bn_ifwp + wi_start) * bk);
            DECL_VLA_PTR_PT_EXT(float, sums_N,   [N][2*bk],              t_scratch_bn, sum_N_offset);

            if (!use_hw_blocking_outside_fusion) {
              for (int hi = 0; hi < H; hi++) {
                for (int w = 0; w < W; w += spatial_block_size_outside_fusion) {
                  if (hi == 0 && w == 0)
                    reduce_beta0_nonfusion_tpp(inp[n][kb][hi][w], sums_N[kb][n]);
                  else
                    reduce_beta1_nonfusion_tpp(inp[n][kb][hi][w], sums_N[kb][n]);
                }
              }
            } else {
              for(int hwb=0; hwb < num_HW_blocks_outside_fusion; hwb++){
                int hi = (hwb*(H*W/num_HW_blocks_outside_fusion))/W;
                int w  = (hwb*(H*W/num_HW_blocks_outside_fusion))%W;
                if (hwb == 0)
                  reduce_beta0_nonfusion_tpp(inp[n][kb][hi][w], sums_N[kb][n]);
                else
                  reduce_beta1_nonfusion_tpp(inp[n][kb][hi][w], sums_N[kb][n]);
              }
            }
          },
          [&]() {},
          [&]() {});
      } /* end of the fusedbtlnk_bn_fwd_reduce scope with recorded parallel for */
    } /* for if (!fuse_stats) */

    if (separate_stats_reduction) {
#ifdef VERBOSE
      std::cout << "Running the separate stats reduction loop" << std::endl;
#endif
      RECORD_SCOPE(fusedbtlnk_bn_fwd_stats, {});
      {
        kb_loop(
          [&](int *ind) {
            const int kb = ind[0];

            DECL_VLA_PTR_PT    (float, sum_X_X2, [Kb][bk],  t_scratch_bn);
            DECL_VLA_PTR_PT_EXT(float, sums_N,   [N][2*bk], t_scratch_bn, sum_N_offset);
            DECL_VLA_PTR_PT    (float, mean,     [bk],      t_BM);
            DECL_VLA_PTR_PT    (float, var,      [bk],      t_BV);

            zero_bk_tpp(sum_X_X2[0][kb]);
            zero_bk_tpp(sum_X_X2[1][kb]);

            for(int ni = 0; ni < N; ni++){
              helper_add_tpp(sum_X_X2[0][kb], &sums_N[kb][ni][0],  sum_X_X2[0][kb]);
              helper_add_tpp(sum_X_X2[1][kb], &sums_N[kb][ni][bk], sum_X_X2[1][kb]);
            }

            mean_var_tpp( sum_X_X2[0][kb], sum_X_X2[1][kb], mean[kb], var[kb]);
          },
          [&]() {},
          [&]() {});
      } /* end of the fusedbtlnk_bn_fwd_stats scope with recorded parallel for */
    } /* for if (separate_stats_reduction) */
  } /* end of if (training) for computing the stats */

#ifdef TIMING
  t_bn_stats_end = getTime();
#endif

  if (!fuse_scaling)
  {
    RECORD_SCOPE(fusedbtlnk_bn_fwd_scale, {});
    {
      nkb_loop(
        [&](int *ind) {
          const int n = ind[0], kb = ind[1];

          DECL_VLA_PTR_PT_EXT(T,             inp,      [Kb][bn_ifhp][bn_ifwp][bk], t_BI, (hi_start * bn_ifwp + wi_start) * bk);
          DECL_VLA_PTR_PT_EXT(T,             inp_add,  [Kb][bn_ifhp][bn_ifwp][bk], t_BIA, (hi_start * bn_ifwp + wi_start) * bk);
          DECL_VLA_PTR_PT    (T,             out,      [Kb][bn_ofhp][bn_ofwp][bk], t_BO);
          DECL_VLA_PTR_PT    (unsigned char, relumask, [Kb][bn_ofhp][bn_ofwp][bk/BITS_PER_CHAR], t_relu_mask);
          DECL_VLA_PTR_PT    (float,         gamma,    [bk],                 t_BW);
          DECL_VLA_PTR_PT    (float,         beta,     [bk],                 t_BB);
          DECL_VLA_PTR_PT    (float,         mean,     [bk],     t_BM);
          DECL_VLA_PTR_PT    (float,         var,      [bk],     t_BV);

          LIBXSMM_ALIGNED(float s[bk], 64);
          LIBXSMM_ALIGNED(float b[bk], 64);

          coeffs_tpp(mean[kb], var[kb], &s[0], &b[0]);

          if (!use_hw_blocking_outside_fusion) {

            if (bn_pad_h_out != 0) {
              zero_hp_tpp(out[n][kb][0][0]);
            }

            for (int hi = 0, ho = ho_start; hi < H; hi++, ho++) {
              /* zeroing out starting [0, wo_start) x bk and [wo_end, bn_ofwp] x bk blocks for fixed ho */
              if (bn_pad_w_out != 0) {
                zero_wp_tpp(out[n][kb][ho][0]);
              }

              for (int wb = 0; wb < num_W_blocks_outside_fusion; wb++) {
                normalize_tpp(inp[n][kb][hi][wb*(W/num_W_blocks_outside_fusion)], &s[0], &b[0], gamma[kb], beta[kb],
                                eltwise ? inp_add[n][kb][hi][wb*(W/num_W_blocks_outside_fusion)] : NULL,
                                out[n][kb][ho][wo_start + wb*(W/num_W_blocks_outside_fusion)],
                                relu ? relumask[n][kb][ho][wo_start + wb*(W/num_W_blocks_outside_fusion)] : NULL);
              }
              /* zeroing out ending [wo_end, bn_ofwp] x bk block for fixed ho */
              if (bn_pad_w_out != 0) {
                zero_wp_tpp(out[n][kb][ho][wo_end]);
              }
            }
            /* zeroing out strip [ho_end, bn_ofhp) x bn_ofwp x bk */
            if (bn_pad_h_out != 0) {
              zero_hp_tpp(out[n][kb][ho_end][0]);
            }

          } else {
            for(int hwb = 0; hwb < num_HW_blocks_outside_fusion; hwb++){
              int hi = (hwb*(H*W/num_HW_blocks_outside_fusion))/W;
              int ho = hi;
              int w  = (hwb*(H*W/num_HW_blocks_outside_fusion))%W;

              /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
              normalize_tpp(inp[n][kb][hi][w], &s[0], &b[0], gamma[kb], beta[kb],
                              eltwise ? inp_add[n][kb][hi][w] : NULL,
                              out[n][kb][ho][w],
                              relu ? relumask[n][kb][ho][w] : NULL);
            }
          } /* if-else for the presence of padding */
        },
        [&]() {},
        [&]() {});
    } /* end of the fusedbtlnk_bn_fwd_scale scope with recorded parallel for */
  } /* if (!fuse_scaling) */


#ifdef TIMING
  t_bn_end = getTime();
  t_end = t_bn_end;
#endif

#else /* #ifndef NO_BATCHNORM */

#ifdef TIMING
  t_end = getTime();
#endif

#endif /* #ifndef NO_BATCHNORM */
} /* end of the scope for conv1 + bn1 */

//} // for the dummy 1000 iter loop while perf debugging

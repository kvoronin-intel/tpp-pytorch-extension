RECORD_FUNCTION("conv_bwd_d", std::vector<c10::IValue>());

#define TIMING

#ifdef TIMING
  double t_start = 0.0, t_end = 0.0, t_conv_start = 0.0, t_wt_trans_end = 0.0;
#endif

#ifdef TIMING
t_start = getTime();
#endif

// ( grad_output, input, weight) = inputs

#define VERBOSE

auto t_GO = inputs[0]; // [N][Kb][H][W][bk]
auto t_I  = inputs[1]; // [N][Cb][H][W][bc]
auto t_W  = inputs[2];

auto sizes = t_I.sizes();

int R = cfg.R;
int S = cfg.S;
int ifh = cfg.H;
int ifw = cfg.W;
int ofh = cfg.ofh;
int ofw = cfg.ofw;
int ifhp = cfg.ifhp;
int ifwp = cfg.ifwp;
int ofhp = cfg.ofhp;
int ofwp = cfg.ofwp;
int bk = cfg.bk;
int bc = cfg.bc;
int bn = cfg.N;
int stride_h = cfg.u;
int stride_w = cfg.v;
int Cb = cfg.blocksifm;
int Kb = cfg.blocksofm;

int conv_pad_h = cfg.pad_h;
int conv_pad_w = cfg.pad_w;

int pad_h_in = cfg.pad_h_in;
int pad_w_in = cfg.pad_w_in;
int pad_h_out = cfg.pad_h_out;
int pad_w_out = cfg.pad_w_out;

int nThreads = cfg.threads;

int C = Cb * bc;
int K = Kb * bk;

const long N  = sizes[0];

std::vector<long> output_size{N, Kb, ofhp, ofwp, bk};

std::vector<long> weight_tr_size{Cb, Kb, R, S, bk, bc};

#ifdef VERBOSE
std::cout << "t_I sizes = " << t_I.sizes() << std::endl;
std::cout << "output_size = " << output_size << std::endl;
std::cout << "R = " << R << " S = " << S << std::endl;
std::cout << "stride_h = " << stride_h << " stride_w = " << stride_w << std::endl;
std::cout << "pad_h_in = " << pad_h_in << " pad_w_in = " << pad_w_in << std::endl;
std::cout << "Cb Kb bc Kb bk = " << Cb << " " << Kb << " " << bc << " " << Kb << " " << bk << std::endl;
//std::cout << "weight_tr_size = " << weight_tr_size << std::endl;
#endif

auto t_grad_input  = at::empty(t_I.sizes(), torch::TensorOptions().dtype(t_I.dtype()));
auto t_WT          = at::empty(weight_tr_size, torch::TensorOptions().dtype(t_W.dtype()));

{ /* main dummy scope */

//------------------------------------

  long  pad_h = pad_h_out;
  long  pad_w = pad_w_out;

  long Kb_step = Kb / k_block;

  long avoid_rim_fmas = 0;
  long non_1x1_with_strides = 0;
  if (ofh <= 7 && ofw <=7 && R == 3 && S == 3 && stride_w == 1 && stride_h == 1) {
    avoid_rim_fmas = 1;
  }

  if (avoid_rim_fmas == 1) {
#ifdef VERBOSE
    printf("Tweaking setup for avoid_rim_fmas = 1 in bwd_d: h_in_gemm = 1 (must be!)\n");
#endif
    h_in_gemm = 1;
  }

  if ((R != 1 && stride_h != 1) ||
      (S != 1 && stride_w != 1)) {
    non_1x1_with_strides = 1;
  }

  //auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bc, dtype, dtype, dtype);
  //if (dtype == LIBXSMM_DATATYPE_F32) {
  //  wt_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  //} else {
  //  wt_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  //}

  SCOPEIT_DECL(XformExtTPP<T>) wt_trans_tpp;

  if (sizeof(T) == 4)
    wt_trans_tpp = SCOPEIT(XformExtTPP<T>(bc, bk, bk, bc, XformTPP::XFORM_XPOSE_TPP, false), XPOSE); /* assuming row-major-ness */
  else
    wt_trans_tpp = SCOPEIT(XformExtTPP<T>(bc, bk, bk, bc, XformTPP::XFORM_XPOSE_V2V_TPP, false), XPOSE); /* assuming row-major-ness */

  long n_step = 1;
  long c_step = 1;
  long k_step = Kb_step;
  long h_step = h_in_gemm;
  long w_step = ofw / w_block;
  long r_step = R;
  long s_step = S;

  if ((avoid_rim_fmas == 1) || (non_1x1_with_strides == 1)) {
    r_step = 1;
    s_step = 1;
  }

  if ( (h_in_gemm > 1) && (w_block != 1) ) {
    printf("Invalid input GEMM config: When multiple H pixels are handled in the gemm, then the full ofw should be also used as gemm_n...\n");
    exit(-1);
  }

  auto w_gemm_pixels = ofw / w_block;
  auto gemm_n = (w_gemm_pixels +  2 * conv_pad_w) * (h_in_gemm - 2) + 2 * (w_gemm_pixels + conv_pad_w);
  auto w_zero_pixels = ifw / w_block;
  auto zero_n = (w_zero_pixels +  2 * pad_w) * (h_in_gemm - 2) + 2 * (w_zero_pixels + pad_w);

  //long gemm_n = ofw;
  long gemm_m = bc;
  long gemm_k = bk;

#ifdef VERBOSE
  std::cout << "gemm_n gemm_m gemm_k = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;
#endif

  std::unique_ptr<unsigned long long[]> A_offsets, B_offsets;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>)         brgemm_tpp, brgemm2_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>)               zero_initial_pixels_tpp, zero_all_pixels_tpp, zero_bc_tpp;

  //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bc*ifwp, 1, bc*ifwp, bc*ifwp, dtype, dtype, dtype);
  //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  //zero_initial_pixels_tpp = SCOPEIT(SetZeroTPP<T>(bc*ifwp/w_block), EW_ZERO);
  zero_initial_pixels_tpp = SCOPEIT(SetZeroTPP<T>(bc*zero_n), EW_ZERO);
  //l_unary_shape = libxsmm_create_meltw_unary_shape(bc*ifwp*ifhp, 1, bc*ifwp*ifhp, bc*ifwp*ifhp, dtype, dtype, dtype);
  //zero_kernel_all_pixels = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  zero_all_pixels_tpp = SCOPEIT(SetZeroTPP<T>(bc*ifwp*ifhp), EW_ZERO);

  if ((R == 1 && S == 1) ||
      (avoid_rim_fmas == 1) ||
      (non_1x1_with_strides == 1)) {

    //l_unary_shape = libxsmm_create_meltw_unary_shape(bc, 1, bc, bc, dtype, dtype, dtype);
    //zero_kernel_bc = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
    zero_bc_tpp = SCOPEIT(SetZeroTPP<T>(bc), EW_ZERO);

    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bk*ofhp*ofwp*sizeof(DType), Kb_step );
    //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, bk*ofhp*ofwp, R*S*bc*bk, bk, bc, bc*stride_w, 1.0, 0, Kb_step * r_step * s_step /*brcount*/)));//, BRGEMM);
    //l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n-1, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    //brgemm_kernel2.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm2_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-1, gemm_m, gemm_k, bk*ofhp*ofwp, R*S*bc*bk, bk, bc, bc*stride_w, 1.0, 0, Kb_step * r_step * s_step /*brcount*/)));//, BRGEMM);

  } else {

    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_OFFSET, 0, 0, 0 );
    //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* no strides due to reduce_offset */ bk, bc, bc*stride_w, 1.0, 0, Kb_step * r_step * s_step /*brcount*/)));//, BRGEMM);

    A_offsets = std::make_unique<unsigned long long[]>(Kb * R * S);
    B_offsets = std::make_unique<unsigned long long[]>(Kb * R * S);

    // Prepare offset array
    unsigned long long i = 0;
    for (long ifm = 0; ifm < Kb_step; ifm++) {
      for (long kj = 0; kj < R; kj++) {
        for (long ki = 0; ki < S; ki++) {
          A_offsets[i] = (ifm * R * S * bc * bk +
              kj * S * bc * bk +
              ki * bc * bk) * sizeof(T);
          B_offsets[i] = (ifm * ofhp * ofwp * bk +
              kj * ofwp * bk +
              ki * bk) * sizeof(T);
          i++;
        }
      }
    } /* outer loop for filling the offsets */
  } /* if-else over the datatype T */

#ifdef VERBOSE
  std::cout << "debug: N n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " " << n_step << " " << Cb << " " << c_step << " "
                                                                                                << Kb << " " << k_step << " " << ofh << " " << h_step << " "
                                                                                                << ofw << " " << w_step << " " << R << " " << r_step << " "
                                                                                                << S << " " << s_step << " " << std::endl;

  std::cout << "pad_h_out pad_w_out = " << pad_h_out << " " << pad_w_out << std::endl;
  std::cout << "h_block w_block c_block k_block = " << h_block << " " << w_block << " " << c_block << " " << k_block << std::endl;
  std::cout << "h_in_gemm = " << h_in_gemm << std::endl;
  std::cout << "avoid_rim_fmas = " <<  avoid_rim_fmas << std::endl;
  std::cout << "non_1x1_with_strides = " << non_1x1_with_strides << std::endl;
#endif


  /* FIXME: Fix this! */
  char wt_trans_loop_specs_str[256] = "ABCD";

  auto wt_trans_loop = ThreadedLoop<4>({
      LoopSpecs{0, Kb, 1, false}, // true},
      LoopSpecs{0, Cb, 1, false},//, true},
      LoopSpecs{0, R, 1, false},//, true},
      LoopSpecs{0, S, 1, false}},//, true}},
      "ABCD");

  /* FIXME: Fix this! */
  //char loop_specs_str[256] = "Abcdefg";

  char loop_specs_str[256];
  std::strcpy(loop_specs_str, tuning_string.c_str());

#ifdef VERBOSE
  std::cout << "tuning_params = " << tuning_params << std::endl;
  std::cout << "wt_trans_loop spec_str = " << wt_trans_loop_specs_str << std::endl;
  std::cout << "loop_specs_str = " << loop_specs_str << std::endl;
#endif

  auto conv_bwd_d_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, false},// true},
      LoopSpecs{0, Cb, c_step, {c_block}},
      LoopSpecs{0, Kb, k_step},//, true},
      LoopSpecs{0, ofh, h_step, {h_block}},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      loop_specs_str);

#ifdef TIMING
  t_conv_start = getTime();
#endif

  {
    RECORD_SCOPE(conv_bwd_d, {});
    {

      wt_trans_loop(
        [&](int* ind) {
          int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];

          DECL_VLA_PTR_PT    (T,    weight,    [Cb][R][S][bc][bk], t_W);
          DECL_VLA_PTR_PT    (T,    weight_tr, [Kb][R][S][bk][bc], t_WT);

          wt_trans_tpp(weight[i_k][i_c][i_r][i_s][0], weight_tr[i_c][i_k][R-1-i_r][S-1-i_s][0]);
        },
        [&]() {},
        [&]() {});

#ifdef TIMING
      t_wt_trans_end = getTime();
#endif

      conv_bwd_d_loop(
        [&](int* ind) {
          int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

          DECL_VLA_PTR_PT    (T,    gradout,     [Kb][ofhp][ofwp][bk],   t_GO);
          DECL_VLA_PTR_PT_EXT(T,    gradout_off, [Kb][ofhp][ofwp][bk],   t_GO, (pad_h_out * ofwp * bk + pad_w_out * bk));
          DECL_VLA_PTR_PT    (T,    dinp,        [Cb][ifhp][ifwp][bc],   t_grad_input);
          DECL_VLA_PTR_PT_EXT(T,    dinp_off,    [Cb][ifhp][ifwp][bc],   t_grad_input, (pad_h_in * ifwp * bc + pad_w_in * bc));
          DECL_VLA_PTR_PT    (T,    weight_tr,   [Kb][R][S][bk][bc],     t_WT);

          if (avoid_rim_fmas == 0) {
            if (non_1x1_with_strides == 0) {
              if (i_k == 0 && i_r == 0 && i_s == 0) {
                if (stride_h != 1) {
                  if (i_w == 0 && i_h == 0) {
                    zero_all_pixels_tpp(dinp[i_n][i_c][0][0]);
                  }
                } else {
                  zero_initial_pixels_tpp(dinp_off[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
                }
              }

              brgemm_tpp(gradout  [i_n][i_k][i_h][i_w],
                         weight_tr[i_c][i_k][i_r][i_s][0],
                         dinp_off [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         B_offsets.get(), A_offsets.get(),
                         Kb_step * r_step * s_step,
                         true);
            } else { /* for non_1x1_with_strides == 0 */
              if (i_k == 0 && i_r == 0 && i_s == 0) {
                if (stride_h != 1) {
                  if (i_w == 0 && i_h == 0) {
                    zero_all_pixels_tpp(dinp[i_n][i_c][0][0]);
                  }
                } else {
                  zero_initial_pixels_tpp(dinp[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
                }
              }

              brgemm_tpp(gradout_off[i_n][i_k][i_h][i_w],
                         weight_tr  [i_c][i_k][R-1-i_r][S-1-i_s][0],
                         dinp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         Kb_step * r_step * s_step,
                         true);

              /* Zero Rim..  */
              if (i_r == R-1 && i_s == S-1 && i_h == ofh-h_step && i_w == ofw-w_step && i_k == Kb - Kb_step) {
                for (int ij = 0; ij < ifhp; ij++) {
                  for (int ii = 0; ii < ifwp; ii++) {
                    if ((ij < pad_h_in || ij >= ifh + pad_h_in) ||
                        (ii < pad_w_in || ii >= ifw + pad_w_in)) {
                      zero_bc_tpp(dinp[i_n][i_c][ij][ii]);
                    }
                  }
                }
              }
            } /* else-if for non_1x1_with_strides == 0 */
          } else { /* else for if avoid_rim_fmas == 0 */
            if (i_k == 0 && i_r == 0 && i_s == 0) {
              zero_initial_pixels_tpp(dinp_off[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
            }
            if (i_r == 0 && i_h == 0) {
              /* Do no FLOPS  */
            } else if (i_r == R-1 && i_h == ofh-1 ) {
              /* Do no FLOPS  */
            } else if ( i_w == 0 && i_s == 0 ) {
              brgemm2_tpp(gradout  [i_n][i_k][i_h + i_r][i_w + i_s + 1],
                          weight_tr[i_c][i_k][i_r][i_s][0],
                          dinp_off [i_n][i_c][i_h][i_w + 1],
                          Kb_step,
                          true);
            } else if ( i_w + w_step == ofw  && i_s == S-1) {
              brgemm2_tpp(gradout  [i_n][i_k][i_h + i_r][i_w + i_s],
                          weight_tr[i_c][i_k][i_r][i_s][0],
                          dinp_off [i_n][i_c][i_h][i_w],
                          Kb_step,
                          true);
            } else {
              brgemm_tpp(gradout  [i_n][i_k][i_h + i_r][i_w + i_s],
                         weight_tr[i_c][i_k][i_r][i_s][0],
                         dinp_off [i_n][i_c][i_h][i_w],
                         Kb_step,
                         true);
            }
          } /* else-if for avoid_rim_fmas */
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});

    } /* end of the scope with recorded parallel for */
  } /* end of the conv_bwd_d scope */

} /* end of the dummy scope */

#ifdef TIMING
  t_end = getTime();
#endif

#ifdef TIMING
  auto buf = tuning_timings.request();
  float* ptr = (float*)buf.ptr;
  //printf("dbg: in conv_bwd_d_tmpl.h adding %f to current %f timer \n", (t_end - t_conv_start), ptr[0]);
  ptr[0] += t_end - t_conv_start;
  ptr[1] += t_wt_trans_end - t_conv_start;
  ptr[2] += t_end - t_start;
#endif

#ifdef VERBOSE
  #undef VERBOSE
#endif

#ifdef TIMING
  #undef TIMING
#endif

//auto t_dummy     = at::empty({0},  torch::TensorOptions().dtype(at::kFloat));
return t_grad_input;

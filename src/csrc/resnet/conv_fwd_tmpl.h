
RECORD_FUNCTION("conv_fwd", std::vector<c10::IValue>());

// ( input, weight) = inputs

auto t_I  = inputs[0]; // [N][CP][H][W][bc]
auto t_W  = inputs[1];

auto sizes = t_I.sizes();

int R = cfg.R;
int S = cfg.S;
//int ifm = cfg.C;
int ofh = cfg.ofh;
int ofw = cfg.ofw;
int ifhp = cfg.ifhp;
int ifwp = cfg.ifwp;
int ofhp = cfg.ofhp;
int ofwp = cfg.ofwp;
int bk = cfg.bk;
int bc = cfg.bc;
int stride_h = cfg.u;
int stride_w = cfg.v;
int Cb = cfg.blocksifm;
int Kb = cfg.blocksofm;

int pad_h_out = cfg.pad_h_out;
int pad_w_out = cfg.pad_w_out;

const long N  = sizes[0];

std::vector<long> output_size{N, Kb, ofhp, ofwp, bk};

//std::cout << "t_I sizes = " << t_I.sizes() << std::endl;
//std::cout << "size of T = " << sizeof(T) << std::endl;
//std::cout << "output_size = " << output_size << std::endl;
//std::cout << "CP Cb bc Kb bk = " << CP << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;

auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));
//return std::vector<at::Tensor>({t_O});

{

  auto gemm_n = ofw;
  auto gemm_m = bk;
  auto gemm_k = bc;

  long Cb_step = Cb;

  //std::cout << "gemm_n gemm_m gemm_k = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;

  std::unique_ptr<unsigned long long[]> A_offsets, B_offsets;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>) brgemm_tpp, brgemm2_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>) zero_tpp;

  //cfg.avoid_fmas_in_rim = 1;

  /* n,m,k, stride_b, stride_a, ldb, lda, ldc, beta, a_trans, unroll_hint because of the row-major */
  if ((R == 1 && S == 1) || (cfg.avoid_fmas_in_rim == 1)) {
    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bc*ifhp*ifwp*sizeof(DType), Cb_step );
    //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    //l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n-1, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    //brgemm_kernel2.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n  , gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);
    brgemm2_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-1, gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);

    //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, dtype, dtype, dtype);
    //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);
  } else {
    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_OFFSET, 0, 0, 0 );
    //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* no strides due to reduce_offset */ bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);

    //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, dtype, dtype, dtype);
    //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);

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
          /* printf("A_offsets[%d] = %llu B_offsets[%d] = %llu \n", i, A_offsets[i], i, B_offsets[i]); */
          i++;
        }
      }
    } /* outer loop for filling the offsets */
  }

  long n_step = 1;
  long c_step = Cb_step;
  long k_step = 1;
  long h_step = 1;
  long w_step = ofw;
  long r_step = R;
  long s_step = S;

  if (cfg.avoid_fmas_in_rim == 1) {
    r_step = 1;
    s_step = 1;
  }

  //std::cout << "debug: N n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " " << n_step << " " << Cb << " " << c_step << " "
  //                                                                                              << Kb << " " << k_step << " " << ofh << " " << h_step << " "
  //                                                                                              << ofw << " " << w_step << " " << R << " " << r_step << " "
  //                                                                                              << S << " " << s_step << " " << std::endl;

  //std::cout << "pad_h_out pad_w_out = " << pad_h_out << " " << pad_w_out << std::endl;
  //std::cout << "avoid fmas in rim = " <<  cfg.avoid_fmas_in_rim << std::endl;

  /* FIXME: Fix this! */
  char loop_specs_str[256] = "Abcdefg";
  //char loop_specs_str[256] = "ABc";

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, false},//, true},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      loop_specs_str);

  {
    RECORD_SCOPE(conv_fwd, {});
    {
      conv_loop(
        [&](int* ind) {
          int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

          DECL_VLA_PTR_PT_EXT(T,     output_off, [Kb][ofhp][ofwp][bk],   t_O, (pad_h_out * ofwp * bk + pad_w_out * bk));
          DECL_VLA_PTR_PT    (T,     inp,        [Cb][ifhp][ifwp][bc],   t_I);
          DECL_VLA_PTR_PT    (T,     weight,     [Cb][R][S][bc][bk],     t_W);

          if (cfg.avoid_fmas_in_rim == 0) {

            if (i_c == 0 && i_r == 0 && i_s == 0) {
              zero_tpp(output_off[i_n][i_k][i_h][i_w]);
            }

            brgemm_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                       weight    [i_k][i_c][i_r][i_s][0],
                       output_off[i_n][i_k][i_h]                 [i_w],
                       B_offsets.get(), A_offsets.get(),
                       Cb_step * r_step * s_step,
                       true);
          } else { /* for if cfg.avoid_fmas_in_rim == 0 */
            if (i_c == 0 && i_r == 0 && i_s == 0) {
              zero_tpp(output_off[i_n][i_k][i_h][i_w]);
            }
            if (i_r == 0 && i_h == 0) {
              /* Do no FLOPS  */
            } else if (i_r == R-1 && i_h == ofh-1 ) {
              /* Do no FLOPS  */
            } else if ( i_w == 0 && i_s == 0 ) {
              brgemm2_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s + 1],
                          weight    [i_k][i_c][i_r][i_s][0],
                          output_off[i_n][i_k][i_h]                 [i_w + 1],
                          Cb_step,
                          true);
            } else if ( i_w + w_step == ofw  && i_s == S-1) {
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
          } /* for if-else cfg.avoid_fmas_in_rim == 0 */
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});
    } /* end of the scope with recorded parallel for */
  } /* end of the conv_fwd_scale scope */

} /* end of the dummy scope */

return t_O;


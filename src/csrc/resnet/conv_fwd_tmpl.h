
RECORD_FUNCTION("conv_fwd", std::vector<c10::IValue>());

// ( input, weight) = inputs

//#define VERBOSE

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

auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));
//return std::vector<at::Tensor>({t_O});

/* hardcoded here unlike the fused bottleneck where it is an external parameter */
long pack_input = 0;

/* in T */
const long conv_fwd_scratch_size = (pack_input == 0 ? 0 : N*ofh*ofw*cfg.C);
auto t_scratch_conv = at::empty(conv_fwd_scratch_size, torch::TensorOptions().dtype(t_I.dtype()));

#ifdef VERBOSE
std::cout << "t_I sizes = " << t_I.sizes() << std::endl;
std::cout << "size of T = " << sizeof(T) << std::endl;
std::cout << "output_size = " << output_size << std::endl;
std::cout << "Cb bc Kb bk = " << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;
std::cout << "stride_h stride_w = " << cfg.u << " " << cfg.v << std::endl;
std::cout << "scratch size = " << conv_fwd_scratch_size << std::endl;
#endif

{

  auto gemm_n = ofw;
  auto gemm_m = bk;
  auto gemm_k = bc;

  long Cb_step = Cb;

#ifdef VERBOSE
  std::cout << "gemm_n gemm_m gemm_k = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;
#endif

  std::unique_ptr<unsigned long long[]> A_offsets, B_offsets;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>) brgemm_tpp, brgemm2_tpp;
  SCOPEIT_DECL(CpyTPP<T>)     input_pack_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>) zero_tpp;

  //cfg.avoid_fmas_in_rim = 1;

  if (R != 1 || S != 1) {
    pack_input = 0;
  }

  /* n,m,k, stride_b, stride_a, ldb, lda, ldc, beta, a_trans, unroll_hint because of the row-major */
  if ((R == 1 && S == 1) || (cfg.avoid_fmas_in_rim == 1)) {
    if (pack_input == 0) {
      brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n  , gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);
    } else {
      if (cfg.avoid_fmas_in_rim) {
        printf("Error: cfg.avoid_fmas_in_rim = %d is incompatible with pack_input = %d\n", cfg.avoid_fmas_in_rim, pack_input);
        exit(-1);
      }
      if (R != 1 || S != 1) {
        printf("Error: R = %d and S = %d are incompatible with pack_input = %d\n", R, S, pack_input);
        exit(-1);
      }
      input_pack_tpp = SCOPEIT(CpyTPP<T>(gemm_n, bc, bc*stride_w, bc), EW_COPY); /* gemm_n, bc because of the row-major for unary */

      brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, bc*ofh*ofw, R*S*bc*bk, bc, bk, bk, 1.0, 0, 0)));//, BRGEMM);
    }

    brgemm2_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-1, gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);

    zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);
  } else {
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* no strides due to reduce_offset */ bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);

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

#ifdef VERBOSE
  std::cout << "debug: N n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " " << n_step << " " << Cb << " " << c_step << " "
                                                                                                << Kb << " " << k_step << " " << ofh << " " << h_step << " "
                                                                                                << ofw << " " << w_step << " " << R << " " << r_step << " "
                                                                                                << S << " " << s_step << " " << std::endl;

  std::cout << "pad_h_out pad_w_out = " << pad_h_out << " " << pad_w_out << std::endl;
  std::cout << "pack_input = " << pack_input << std::endl;
  std::cout << "avoid fmas in rim = " <<  cfg.avoid_fmas_in_rim << std::endl;
#endif

  /* FIXME: Fix this! */
  char loop_specs_str[256] = "Abcdefg";
  //char loop_specs_str[256] = "ABc";
  //char loop_specs_str[256] = "aBC";

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

            if (pack_input > 0 && i_r == 0 && i_s == 0 && i_k == 0 && i_c == 0) {
              DECL_VLA_PTR_PT(T, packed_inp, [Cb][ofh][ofw][bc], t_scratch_conv);
              for (int _br = 0; _br < Cb; _br++) {
                for (int _h = 0; _h < h_step; _h++) {
                  input_pack_tpp(inp[i_n][_br][(i_h+_h)*stride_h][i_w * stride_w], packed_inp[i_n][_br][i_h+_h][i_w]);
                }
              }
            }

            if (pack_input == 0) {
              brgemm_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         weight    [i_k][i_c][i_r][i_s][0],
                         output_off[i_n][i_k][i_h]                 [i_w],
                         B_offsets.get(), A_offsets.get(),
                         Cb_step * r_step * s_step,
                         true);
            } else {
              DECL_VLA_PTR_PT(T, packed_inp, [Cb][ofh][ofw][bc], t_scratch_conv);
              brgemm_tpp(packed_inp[i_n][i_c][i_h][i_w],
                         weight    [i_k][i_c][i_r][i_s][0],
                         output_off[i_n][i_k][i_h]                 [i_w],
                         B_offsets.get(), A_offsets.get(),
                         Cb_step * r_step * s_step,
                         true);
            }
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


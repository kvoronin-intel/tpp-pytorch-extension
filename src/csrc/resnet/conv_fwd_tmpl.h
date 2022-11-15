
RECORD_FUNCTION("conv_fwd", std::vector<c10::IValue>());

#define TIMING

#ifdef TIMING
  double t_start = 0.0, t_end = 0.0, t_conv_start = 0.0;
#endif

#ifdef TIMING
t_start = getTime();
#endif

// ( input, weight) = inputs

#define VERBOSE

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

const int pad_h_in = cfg.pad_h_in;
const int pad_w_in = cfg.pad_w_in;
const int pad_h_out = cfg.pad_h_out;
const int pad_w_out = cfg.pad_w_out;
const int conv_pad_h = cfg.pad_h;
const int conv_pad_w = cfg.pad_w;

const int ifh = ifhp - 2 * pad_h_in;
const int ifw = ifwp - 2 * pad_w_in;

const int logical_padding = (pad_h_in == 0 && pad_w_in == 0 && pad_h_out == 0 && pad_w_out == 0 ? 1 : 0);

const long N  = sizes[0];

std::vector<long> output_size{N, Kb, ofhp, ofwp, bk};

#ifndef PREALLOCATED_OUTPUT
auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));
#else
//printf("PREALLOCATED_OUTPUT is enabled\n");
#endif
//return std::vector<at::Tensor>({t_O});

/* hardcoded here unlike the fused bottleneck where it is an external parameter */
//long pack_input = 0;

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

  //auto gemm_n = ofw;
  auto w_gemm_pixels = ofw / w_block;
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

  //cfg.avoid_fmas_in_rim = 1;
  long avoid_fmas_in_rim = 0;
  if (ofh <= 7 && ofw <=7 && R == 3 && S == 3 && stride_w == 1 && stride_h == 1 && h_in_gemm == 1) {
    avoid_fmas_in_rim = 1;
    cfg.avoid_fmas_in_rim = 1; //??? FIXME
  }

  if (logical_padding && (conv_pad_h != 0 || conv_pad_w != 0)) {
    avoid_fmas_in_rim = 1;
    cfg.avoid_fmas_in_rim = 1;
  }

  if (cfg.avoid_fmas_in_rim == 1 && (R == 1 || S == 1)) {
    printf("Error: cfg.avoid_fmas_in_rim does not work (and does not make sense) for 1x1 filters\n");
    exit(-1);
  }

  if (cfg.avoid_fmas_in_rim == 1 && ((R%2) == 0 || (S%2) == 0)) {
    printf("Error: cfg.avoid_fmas_in_rim does not work for even-sized filters\n");
    exit(-1);
  }

  if (cfg.avoid_fmas_in_rim == 1 && w_block != 1) {
    printf("Warning: w_block != 1 is not thoroughly tested with cfg.avoid_fmas_in_rim\n");
    //return -1;
  }

  if (cfg.avoid_fmas_in_rim == 1) {
    r_step = 1;
    s_step = 1;
  }

#ifdef VERBOSE
  std::cout << "gemm_n gemm_m gemm_k = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;
#endif

  std::unique_ptr<unsigned long long[]> A_offsets, B_offsets;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>) brgemm_tpp, brgemm_1less_tpp, brgemm_2less_tpp; // 2less is added to enable 7x7 filters
  SCOPEIT_DECL(CpyTPP<T>)     input_pack_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>) zero_tpp;

  if (R != 1 || S != 1) {
#ifdef VERBOSE
    std::cout << "Setting pack_input to zero for non 1x1 convs" << std::endl;
#endif
    pack_input = 0;
  }

  /* n,m,k, stride_b, stride_a, ldb, lda, ldc, beta, a_trans, unroll_hint because of the row-major */
  float beta;
  if (Cb_step == Cb && r_step == R && s_step == S)
    beta = 0.0;
  else
    beta = 1.0;

  if ((R == 1 && S == 1) || (cfg.avoid_fmas_in_rim == 1)) {
    if (pack_input == 0) {
      brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n  , gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, beta, 0, 0 /* c_vnni*/, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);
    } else {
      if (cfg.avoid_fmas_in_rim) {
        printf("Error: cfg.avoid_fmas_in_rim = %d is incompatible with pack_input = %d\n", cfg.avoid_fmas_in_rim, pack_input);
        exit(-1);
      }
      if (R != 1 || S != 1) {
        printf("Error: R = %d and S = %d are incompatible with pack_input = %d\n", R, S, pack_input);
        exit(-1);
      }
      input_pack_tpp = SCOPEIT(CpyTPP<T>(w_gemm_pixels, bc, bc*stride_w, bc), EW_COPY); /* gemm_n, bc because of the row-major for unary */

      brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, bc*ofh*ofw, R*S*bc*bk, bc, bk, bk, beta, 0, 0 /* c_vnni*/, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);
    }

    brgemm_1less_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-1, gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, beta, 0, 0 /* c_vnni*/, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);
    brgemm_2less_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-2, gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, beta, 0, 0 /* c_vnni*/, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);

    zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);
  } else {
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* no strides due to reduce_offset */ bc*stride_w, bk, bk, beta, 0, 0 /* c_vnni*/, Cb_step * r_step * s_step /*brcount*/)));//, BRGEMM);

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

  if ( (h_in_gemm > 1) && (w_block != 1) ) {
    printf("Invalid input GEMM config: When multiple H pixels are handled in the gemm, then the full ofw should be also used as gemm_n...\n");
    exit(-1);
  }

  if (logical_padding && h_in_gemm > 1 ) {
    printf("Error: logical padding in conv fwd is only supported for h_in_gemm = 1\n");
    exit(-1);
  }

#ifdef VERBOSE
  std::cout << "debug: N n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " " << n_step << " " << Cb << " " << c_step << " "
                                                                                                << Kb << " " << k_step << " " << ofh << " " << h_step << " "
                                                                                                << ofw << " " << w_step << " " << R << " " << r_step << " "
                                                                                                << S << " " << s_step << " " << std::endl;

  std::cout << "pad_h_in pad_w_in pad_h_out pad_w_out = " << pad_h_in << " " << pad_w_in << " " << pad_h_out << " " << pad_w_out << std::endl;
  std::cout << "h_block w_block c_block k_block = " << h_block << " " << w_block << " " << c_block << " " << k_block << std::endl;
  std::cout << "h_in_gemm = " << h_in_gemm << std::endl;
  std::cout << "cfg.avoid fmas in rim = " <<  cfg.avoid_fmas_in_rim << std::endl;
  std::cout << "unused but internal avoid fmas in rim = " <<  avoid_fmas_in_rim << std::endl;
  std::cout << "logical_padding = " << logical_padding << std::endl;
  std::cout << "pack_input = " << pack_input << std::endl;
#endif

  /* FIXME: Fix this! */
  //char loop_specs_str[256] = "Abcdefg";
  //char loop_specs_str[256] = "ABc";
  //char loop_specs_str[256] = "aBC";

  char loop_specs_str[256];
  std::strcpy(loop_specs_str, tuning_string.c_str());

#ifdef VERBOSE
  std::cout << "tuning_params = " << tuning_params << std::endl;
  std::cout << "loop_specs_str = " << loop_specs_str << std::endl;
#endif

#ifdef VERBOSE
  printf("parlooper fwd string: OMP_NUM_THREADS=%d USE_BF16=%d ./run_conv_fwd.sh %s %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d 1000 %d\n", N, (sizeof(T) == 2 ? 1 : 0), loop_specs_str,
                                        N, ifh, ifw, cfg.C, cfg.K, R, S, stride_h, stride_w, pad_h_out, pad_w_out, bc, bk, h_block, w_block, c_block, k_block, h_in_gemm, cfg.avoid_fmas_in_rim, logical_padding);
#endif

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, false},//, true},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, Kb, k_step, {k_block}},
      LoopSpecs{0, ofh, h_step, {h_block}},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      loop_specs_str);

#ifdef TIMING
  t_conv_start = getTime();
#endif

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

            if (Cb_step != Cb || r_step != R || s_step != S) {
              if (i_c == 0 && i_r == 0 && i_s == 0) {
                zero_tpp(output_off[i_n][i_k][i_h][i_w]);
              }
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
          } else { /* else for if cfg.avoid_fmas_in_rim == 0 */
            if (Cb_step != Cb || r_step != R || s_step != S) {
              if (i_c == 0 && i_r == 0 && i_s == 0) {
                zero_tpp(output_off[i_n][i_k][i_h][i_w]);
              }
            }

            if (R == 7 && S == 7) {
              if (i_h * stride_h + i_r - R/2 < 0) {
                /* Do no FLOPS  */
              } else if (i_h *stride_h + i_r - R/2 >= ifh ) {
                /* Do no FLOPS  */
              } else if ( i_s < R/2 && i_w * stride_w + (i_s - R/2) < 0 && (i_w + 1) * stride_w + (i_s - R/2) >= 0  ) {
                /* the case when left i_s is out of input image for the first pitch only */
                brgemm_1less_tpp(inp       [i_n][i_c][pad_h_in + i_h * stride_h + (i_r - R/2)][pad_w_in + (i_w + 1) * stride_w + (i_s - S/2)],
                                 weight    [i_k][i_c][i_r][i_s][0],
                                 output_off[i_n][i_k][i_h]                 [i_w + 1],
                                 Cb_step,
                                 true);
              } else if ( i_s < R/2 && i_w * stride_w + (i_s - R/2) < 0 && (i_w + 1) * stride_w + (i_s - R/2) < 0 && (i_w + 2) * stride_w + (i_s - R/2) >= 0  ) {
                /* the case when left i_s is out of input image for the first two pitches */
                brgemm_2less_tpp(inp       [i_n][i_c][pad_h_in + i_h * stride_h + (i_r - R/2)][pad_w_in + (i_w + 2) * stride_w + (i_s - S/2)],
                                 weight    [i_k][i_c][i_r][i_s][0],
                                 output_off[i_n][i_k][i_h]                 [i_w + 2],
                                 Cb_step,
                                 true);
              } else if ( i_s > R/2 && (i_w + w_step - 1)*stride_w + (i_s - R/2) >= ifw && (i_w + w_step - 2)*stride_w + (i_s - R/2) < ifw ) {
                /* the case when right i_s is out of input image for the last pitch only */
                brgemm_1less_tpp(inp       [i_n][i_c][pad_h_in + i_h * stride_h + (i_r - R/2)][pad_w_in + i_w * stride_w + (i_s - S/2)],
                                 weight    [i_k][i_c][i_r][i_s][0],
                                 output_off[i_n][i_k][i_h]                 [i_w],
                                 Cb_step,
                                 true);
              } else if ( i_s > R/2 && (i_w + w_step - 1)*stride_w + (i_s - R/2) >= ifw && (i_w + w_step - 2)*stride_w + (i_s - R/2) >= ifw && (i_w + w_step - 3)*stride_w + (i_s - R/2) < ifw ) {
                /* for the case when right i_s is out of input image for the last 2 pitches */
                brgemm_2less_tpp(inp       [i_n][i_c][pad_h_in + i_h * stride_h + (i_r - R/2)][pad_w_in + i_w * stride_w + (i_s - S/2)],
                                 weight    [i_k][i_c][i_r][i_s][0],
                                 output_off[i_n][i_k][i_h]                 [i_w],
                                 Cb_step,
                                 true);
              } else {
                brgemm_tpp(inp       [i_n][i_c][pad_h_in + i_h * stride_h + (i_r - R/2)][pad_w_in + i_w * stride_w + (i_s - S/2)],
                           weight    [i_k][i_c][i_r][i_s][0],
                           output_off[i_n][i_k][i_h]                 [i_w],
                           Cb_step,
                           true);
              }
            } else if (R == 3 && S == 3) {
              if (i_r == 0 && i_h == 0) {
                /* Do no FLOPS  */
              //} else if (i_r == R-1 && i_h == ofh-1 ) {
              } else if (i_r == R-1 && (i_h + h_step - 1)*stride_h + i_r == ifh + 1 ) {
                /* Do no FLOPS  */
              } else if ( i_w == 0 && i_s == 0 ) {
                brgemm_1less_tpp(inp       [i_n][i_c][pad_h_in + i_h * stride_h + (i_r - R/2)][pad_w_in + (i_w + 1) * stride_w + (i_s - S/2)],
                                 weight    [i_k][i_c][i_r][i_s][0],
                                 output_off[i_n][i_k][i_h]                 [i_w + 1],
                                 Cb_step,
                                 true);
              //} else if ( i_w + w_step == ofw  && i_s == S-1) {
              } else if ( (i_w + w_step - 1)*stride_w + i_s == ifw + 1 && i_s == S-1) {
                brgemm_1less_tpp(inp       [i_n][i_c][pad_h_in + i_h * stride_h + (i_r - R/2)][pad_w_in + i_w * stride_w + (i_s - S/2)],
                                 weight    [i_k][i_c][i_r][i_s][0],
                                 output_off[i_n][i_k][i_h]                 [i_w],
                                 Cb_step,
                                 true);
              } else {
                brgemm_tpp(inp       [i_n][i_c][pad_h_in + i_h * stride_h + (i_r - R/2)][pad_w_in + i_w * stride_w + (i_s - S/2)],
                           weight    [i_k][i_c][i_r][i_s][0],
                           output_off[i_n][i_k][i_h]                 [i_w],
                           Cb_step,
                           true);
              }
            } /* if-else if for the filter size (7x7 and 3x3) */
          } /* for if-else cfg.avoid_fmas_in_rim == 0 */
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});
    } /* end of the scope with recorded parallel for */
  } /* end of the conv_fwd_scale scope */

} /* end of the dummy scope */

#ifdef TIMING
  t_end = getTime();
#endif


#ifdef TIMING
  auto buf = tuning_timings.request();
  float* ptr = (float*)buf.ptr;
  ptr[0] += t_end - t_conv_start;
  ptr[1] += t_end - t_start;
//  printf("updating timings here in conv fwd\n");
#endif

#ifdef VERBOSE
  #undef VERBOSE
#endif

#ifdef TIMING
  #undef TIMING
#endif

return t_O;

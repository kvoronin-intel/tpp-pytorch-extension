{

//#define VERBOSE

// t_CI and t_CW should be defined outside
// t_BW, t_BB, t_BM, t_BV, t_BIA should be defined outside
// t_BW_prev, t_BB_prev, t_BM_prev, t_BV_prev, t_relu_mask_prev, eltwise_prev must be defined outside if fuse_scaling = 1
// h_block, w_block, c_block, k_block, avoid_fmas_in_rim and fuse_stats must be defined outside

auto sizes = t_CI.sizes();

//const int fuse_stats = (avoid_fmas_in_rim == 0 ? 1 : 0);
const int separate_stats_reduction = 1; /* only value currently supported is 1 */
//const int fuse_scaling = 0; /* must be defined in the calling code */

char conv_fwd_loop_specs_str[256];
std::strcpy(conv_fwd_loop_specs_str, conv_loop_string.c_str());

#ifdef VERBOSE
std::cout << "CONV+BN meta setup info"           << std::endl;
std::cout << "fuse_stats    = " << fuse_stats    << std::endl;
std::cout << "fuse_scaling  = " << fuse_scaling  << std::endl;
std::cout << "tuning_params = " << tuning_params << std::endl;
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

const long N  = sizes[0];

std::vector<long> conv_output_size{N, Kb, ofhp, ofwp, bk};

#ifdef VERBOSE
std::cout << "CONV setup info" << std::endl;
std::cout << "t_CI sizes = " << t_CI.sizes() << std::endl;
std::cout << "size of T = " << sizeof(T) << std::endl;
std::cout << "conv_output_size = " << conv_output_size << std::endl;
std::cout << "Cb bc Kb bk = " << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;
std::cout << "stride_h stride_w = " << stride_h << " " << stride_w << std::endl;
std::cout << "conv_pad_h_out conv_pad_w_out = " << conv_pad_h_out << " " << conv_pad_w_out << std::endl;
#endif

CONV_OUT = at::empty(conv_output_size, torch::TensorOptions().dtype(t_CI.dtype()));
auto t_CO = CONV_OUT;

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

const long full_fwd_scratch_size = sumsq_N_offset + LIBXSMM_UP2((size_t)Kb * (size_t)N * (size_t)bk, 64);
const long full_bwd_scratch_size = dbeta_N_offset + LIBXSMM_UP2(Kb * N * bk, 64);
const long full_scratch_size     = std::max(full_fwd_scratch_size, full_bwd_scratch_size);
std::vector<long> scratch_size{full_scratch_size};
BN_SCRATCH_OUT = at::empty(scratch_size, torch::TensorOptions().dtype(at::kFloat));
auto t_scratch = BN_SCRATCH_OUT;

bool use_hw_blocking = true;

const long num_HW_blocks = (H > W ? H : W);
const long num_W_blocks  = (W % 64 == 0 ? W / 64 : 1);

long spatial_block_size = 0;

if (bn_pad_h_in != 0 || bn_pad_w_in != 0 || bn_pad_h_out != 0 || bn_pad_w_out != 0 ) {
  use_hw_blocking    = false; /* alternative is w blocking ([w, bc] blocks) */
  spatial_block_size = W / num_W_blocks;
} else {
  use_hw_blocking    = true; /* using [hw, bc] blocks */
  spatial_block_size = H * W / num_HW_blocks;
}

#ifdef VERBOSE
std::cout << "BN setup info" << std::endl;
std::cout << "bn_padding = " << bn_padding << std::endl;
std::cout << "size of T = " << sizeof(T) << std::endl;
std::cout << "bn_output_size = " << bn_output_size << std::endl;
std::cout << "t_BI sizes = " << t_BI.sizes() << std::endl;
std::cout << "use_hw_blocking = " << use_hw_blocking << std::endl;
#endif

#ifdef VERBOSE
std::cout << "Setting up the conv in conv/bn fusion" << std::endl;
#endif
//return std::vector<at::Tensor>({t_CO});

  auto gemm_n = ofw / w_block;
  auto gemm_m = bk;
  auto gemm_k = bc;

  long Cb_step = Cb / c_block;

  //std::cout << "gemm_n gemm_m gemm_k = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;

  std::unique_ptr<unsigned long long[]> A_offsets, B_offsets;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>) brgemm_tpp, brgemm2_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>) zero_tpp;

  zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);
  /* n,m,k, stride_b, stride_a, ldb, lda, ldc, beta, a_trans, unroll_hint because of the row-major */
  if ((R == 1 && S == 1) || (avoid_fmas_in_rim == 1)) {
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n  , gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);
    brgemm2_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-1, gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);

  } else {
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* no strides due to reduce_offset */ bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);

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
  long w_step = ofw / w_block;
  long r_step = R;
  long s_step = S;

  if (avoid_fmas_in_rim == 1) {
    r_step = 1;
    s_step = 1;
  }

#ifdef VERBOSE
  std::cout << "debug: N n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " " << n_step << " " << Cb << " " << c_step << " "
                                                                                                << Kb << " " << k_step << " " << ofh << " " << h_step << " "
                                                                                                << ofw << " " << w_step << " " << R << " " << r_step << " "
                                                                                                << S << " " << s_step << " " << std::endl;
  std::cout << "h_block w_block c_block k_block = " << h_block << " " << w_block << " " << c_block << " " << k_block << std::endl;
  std::cout << "avoid fmas in rim = " <<  avoid_fmas_in_rim << std::endl;
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
std::cout << "Setting up the bn in conv/bn fusion" << std::endl;
#endif

  auto zero_bk_tpp = SCOPEIT(SetZeroTPP<float>(bk), EW_ZERO);

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bk, bk, bk), EW_ADD); /* 1, bc because of row-major for unary */

  auto reduce_beta0_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size, bk, bk, bk, 1)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto reduce_beta1_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size, bk, bk, bk, 0)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto mean_var_tpp = SCOPEIT(MeanVarTPP<float>(bk, scale), EW_MEAN_VAR);

  auto coeffs_tpp = SCOPEIT(BatchNormStatCoeffsTPP<float>(bk, eps), NORMALIZE);

  auto zero_hp_tpp = SCOPEIT(SetZeroTPP<T>((bn_pad_h_out * bn_ofwp), bk, bk), EW_ZERO); /* (pad_h_out * ofwp), bc because of row-major for unary */

  auto zero_wp_tpp = SCOPEIT(SetZeroTPP<T>(bn_pad_w_out, bk, bk), EW_ZERO);          /* pad_w_out, bc because of row-major for unary */

  auto normalize_tpp = SCOPEIT((BatchNormFwdScaleTPP<T,T>(bk, spatial_block_size, relu, eltwise)), NORMALIZE);

  char nkb_loop_specs_str[256] = "AB";
  const long bn_n_step = 1, bn_kb_step = 1;
  auto nkb_loop = ThreadedLoop<2>({
      LoopSpecs{0, N,  bn_n_step,  {/*l1_n_step, l0_n_step*/}},   // Logical N  loop specs
      LoopSpecs{0, Kb, bn_kb_step, {/*l1_k_step, l0_k_step*/}}},  // Logical Kb loop specs
      nkb_loop_specs_str);

  char kb_loop_specs_str[256] = "A";
  auto kb_loop = ThreadedLoop<1>({
      LoopSpecs{0, Kb, bn_kb_step, {/*l1_k_step, l0_k_step*/}}},  // Logical Kb loop specs
      kb_loop_specs_str);

#ifdef VERBOSE
std::cout << "Running conv part in conv/bn fusion" << std::endl;
#endif

  {
    RECORD_SCOPE(fusedbtlnk_conv_fwd, {});
    {
      conv_loop(
        [&](int* ind) {
          int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

          DECL_VLA_PTR_PT_EXT(T,     output_off, [Kb][ofhp][ofwp][bk],   t_CO, (conv_pad_h_out * ofwp * bk + conv_pad_w_out * bk));
          DECL_VLA_PTR_PT    (T,     inp,        [Cb][ifhp][ifwp][bc],   t_CI);
          DECL_VLA_PTR_PT    (T,     weight,     [Cb][R][S][bc][bk],     t_CW);

          if (avoid_fmas_in_rim == 0) {

            if (i_c == 0 && i_r == 0 && i_s == 0) {
              zero_tpp(output_off[i_n][i_k][i_h][i_w]);
            }

            if (fuse_scaling && i_k == 0 && i_r == 0 && i_s == 0) {
              printf("fuse_scaling = 1 has not been implemented yet\n");
              exit(-1);
            } /* if fuse_scaling + extra conditions */

            brgemm_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                       weight    [i_k][i_c][i_r][i_s][0],
                       output_off[i_n][i_k][i_h]                 [i_w],
                       B_offsets.get(), A_offsets.get(),
                       Cb_step * r_step * s_step,
                       true);

            /* Computing local stats */
            if (training && fuse_stats && i_c == Cb - c_step && i_r == R - r_step && i_s == S - s_step) {
              DECL_VLA_PTR_PT_EXT(float, sums_N,    [N][2*bk],             t_scratch, sum_N_offset);

              if (!use_hw_blocking && (i_w * w_step) % spatial_block_size == 0) {
                if (i_h == 0 && i_w == 0)
                  reduce_beta0_tpp(output_off[i_n][i_k][i_h][i_w], sums_N[i_k][i_n]);
                else
                  reduce_beta1_tpp(output_off[i_n][i_k][i_h][i_w], sums_N[i_k][i_n]);
              } else if ( (i_h * h_step * ofwp + (i_w * w_step)) % spatial_block_size == 0) {
                if (i_h == 0 && i_w == 0)
                  reduce_beta0_tpp(output_off[i_n][i_k][i_h][i_w], sums_N[i_k][i_n]);
                else
                  reduce_beta1_tpp(output_off[i_n][i_k][i_h][i_w], sums_N[i_k][i_n]);
              }
            } /* for computing local stats */
          } else { /* for if avoid_fmas_in_rim == 0 */

            if (fuse_scaling || fuse_stats) {
              printf("No fusion has been implemented for the case avoid_fmas_in_rim != 0\n");
              exit(-1);
            }

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
          } /* for if-else avoid_fmas_in_rim == 0 */
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});
    } /* end of the fusedbtlnk_conv_fwd scope with recorded parallel for */
  } /* end of the dummy scope */

#ifdef VERBOSE
std::cout << "Running bn part in conv/bn fusion" << std::endl;
#endif

  if (training) {
    if (!fuse_stats) {
      RECORD_SCOPE(fusedbtlnk_bn_fwd_reduce, {});
      {
        nkb_loop(
          [&](int *ind) {
            const int n = ind[0], kb = ind[1];

            DECL_VLA_PTR_PT_EXT(T,     inp,      [Kb][bn_ifhp][bn_ifwp][bk], t_BI, (hi_start * bn_ifwp + wi_start) * bk);
            DECL_VLA_PTR_PT_EXT(float, sums_N,   [N][2*bk],              t_scratch, sum_N_offset);

            if (!use_hw_blocking) {
              for (int hi = 0; hi < H; hi++) {
                for (int w = 0; w < W; w += spatial_block_size) {
                  if (hi == 0 && w == 0)
                    reduce_beta0_tpp(inp[n][kb][hi][w], sums_N[kb][n]);
                  else
                    reduce_beta1_tpp(inp[n][kb][hi][w], sums_N[kb][n]);
                }
              }
            } else {
              for(int hwb=0; hwb < num_HW_blocks; hwb++){
                int hi = (hwb*(H*W/num_HW_blocks))/W;
                int w  = (hwb*(H*W/num_HW_blocks))%W;
                if (hwb == 0)
                  reduce_beta0_tpp(inp[n][kb][hi][w], sums_N[kb][n]);
                else
                  reduce_beta1_tpp(inp[n][kb][hi][w], sums_N[kb][n]);
              }
            }
          },
          [&]() {},
          [&]() {});
      } /* end of the fusedbtlnk_bn_fwd_reduce scope with recorded parallel for */
    } /* for if (!fuse_stats) */

    if (separate_stats_reduction) {
      RECORD_SCOPE(fusedbtlnk_bn_fwd_stats, {});
      {
        kb_loop(
          [&](int *ind) {
            const int kb = ind[0];

            DECL_VLA_PTR_PT    (float, sum_X_X2, [Kb][bk], t_scratch);
            DECL_VLA_PTR_PT_EXT(float, sums_N,   [N][2*bk],  t_scratch, sum_N_offset);
            DECL_VLA_PTR_PT    (float, mean,     [bk],     t_BM);
            DECL_VLA_PTR_PT    (float, var,      [bk],     t_BV);

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

          if (!use_hw_blocking) {

            if (bn_pad_h_out != 0) {
              zero_hp_tpp(out[n][kb][0][0]);
            }

            for (int hi = 0, ho = ho_start; hi < H; hi++, ho++) {
              /* zeroing out starting [0, wo_start) x bk and [wo_end, bn_ofwp] x bk blocks for fixed ho */
              if (bn_pad_w_out != 0) {
                zero_wp_tpp(out[n][kb][ho][0]);
              }

              for (int wb = 0; wb < num_W_blocks; wb++) {
                normalize_tpp(inp[n][kb][hi][wb*(W/num_W_blocks)], &s[0], &b[0], gamma[kb], beta[kb],
                                eltwise ? inp_add[n][kb][hi][wb*(W/num_W_blocks)] : NULL,
                                out[n][kb][ho][wo_start + wb*(W/num_W_blocks)],
                                relu ? relumask[n][kb][ho][wo_start + wb*(W/num_W_blocks)] : NULL);
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
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int hi = (hwb*(H*W/num_HW_blocks))/W;
              int ho = hi;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

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

} /* end of the scope for conv1 + bn1 */

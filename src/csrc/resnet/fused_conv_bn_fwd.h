{

#define VERBOSE
// ( input, weight) = inputs

// t_CI and t_CW should be defined outside
// t_BW, t_BB, t_BM, t_BV, t_BIA should be defined outside

auto sizes = t_CI.sizes();

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
std::cout << "t_CI sizes = " << t_CI.sizes() << std::endl;
std::cout << "size of T = " << sizeof(T) << std::endl;
std::cout << "conv_output_size = " << conv_output_size << std::endl;
std::cout << "Cb bc Kb bk = " << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;
std::cout << "stride_h stride_w = " << conv_cfg.u << " " << conv_cfg.v << std::endl;
#endif

std::cout << "Got here 0" << std::endl;
CONV_OUT = at::empty(conv_output_size, torch::TensorOptions().dtype(t_CI.dtype()));
auto t_CO = CONV_OUT;
std::cout << "Got here 1" << std::endl;

const long bn_pad_h_in  = bn_padding[0];
const long bn_pad_w_in  = bn_padding[1];
const long bn_pad_h_out = bn_padding[2];
const long bn_pad_w_out = bn_padding[3];

std::cout << "Got here 2" << std::endl;

auto t_BI = BN_IN;

auto bn_sizes = t_BI.sizes();
//const long N  = sizes[0];
//const long Kb = sizes[1];
const long H  = bn_sizes[2] - 2 * bn_pad_h_in;
const long W  = bn_sizes[3] - 2 * bn_pad_w_in;
//const long bk = sizes[4];
std::cout << "bn_sizes = " << bn_sizes << std::endl;
std::cout << "bn_padding = " << bn_padding << std::endl;

std::cout << "Got here 3" << std::endl;

const long hi_start      = bn_pad_h_in;
const long wi_start      = bn_pad_w_in;
const long bn_ifhp          = H + 2 * bn_pad_h_in;
const long bn_ifwp          = W + 2 * bn_pad_w_in;

const long ho_start      = bn_pad_h_out;
const long ho_end        = ho_start + H;
const long wo_start      = bn_pad_w_out;
const long wo_end        = wo_start + W;
const long bn_ofhp          = H + 2 * bn_pad_h_out;
const long bn_ofwp          = W + 2 * bn_pad_w_out;

const float scale = 1.0f /((float)N * H * W);

std::cout << "Got here 4" << std::endl;

std::vector<long> bn_output_size  {N, Kb, bn_ofhp, bn_ofwp, bk};
std::vector<long> bn_relumask_size{N, Kb, bn_ofhp, bn_ofwp, bk/BITS_PER_CHAR};

std::cout << "Got here 5" << std::endl;

BN_OUT = at::empty(bn_output_size, torch::TensorOptions().dtype(t_BI.dtype()));
auto t_BO = BN_OUT;

BN_RELU_OUT = at::empty(bn_relumask_size, torch::TensorOptions().dtype(at::kByte));
auto t_relu_mask = BN_RELU_OUT;

std::cout << "Got here 6" << std::endl;

const long sum_N_offset          = LIBXSMM_UP2(Kb * 2 * bk, 64);
const long sumsq_N_offset        = LIBXSMM_UP2(sum_N_offset + Kb * N * bk, 64);

const long dbeta_N_offset        = LIBXSMM_UP2(Kb * N * bk, 64);

const long full_fwd_scratch_size = sumsq_N_offset + LIBXSMM_UP2((size_t)Kb * (size_t)N * (size_t)bk, 64);
const long full_bwd_scratch_size = dbeta_N_offset + LIBXSMM_UP2(Kb * N * bk, 64);
const long full_scratch_size     = std::max(full_fwd_scratch_size, full_bwd_scratch_size);
std::vector<long> scratch_size{full_scratch_size};
BN_SCRATCH_OUT = at::empty(scratch_size, torch::TensorOptions().dtype(at::kFloat));
auto t_scratch = BN_SCRATCH_OUT;

std::cout << "Got here 7" << std::endl;

bool use_hw_blocking = true;

const long num_HW_blocks = (H > W ? H : W);
const long num_W_blocks  = (W % 64 == 0 ? W / 64 : 1);

std::cout << "Got here 8" << std::endl;

long spatial_block_size = 0;

if (bn_pad_h_in != 0 || bn_pad_w_in != 0 || bn_pad_h_out != 0 || bn_pad_w_out != 0 ) {
  std::cout << "W = " << W << " num_W_blocks = " << num_W_blocks << std::endl;
  use_hw_blocking    = false; /* alternative is w blocking ([w, bc] blocks) */
  spatial_block_size = W / num_W_blocks;
} else {
  std::cout << "H = " << H << " W = " << W << " num_HW_blocks = " << num_HW_blocks << std::endl;
  use_hw_blocking    = true; /* using [hw, bc] blocks */
  spatial_block_size = H * W / num_HW_blocks;
}

std::cout << "Got here 9" << std::endl;

#ifdef VERBOSE
std::cout << "bn_padding = " << bn_padding << std::endl;
std::cout << "size of T = " << sizeof(T) << std::endl;
std::cout << "bn_output_size = " << bn_output_size << std::endl;
std::cout << "t_BI sizes = " << t_BI.sizes() << std::endl;
std::cout << "use_hw_blocking = " << use_hw_blocking << std::endl;
#endif

std::cout << "Got here 10" << std::endl;


#ifdef VERBOSE
std::cout << "Setting up the conv in conv/bn fusion" << std::endl;
#endif
//return std::vector<at::Tensor>({t_CO});

  auto gemm_n = ofw;
  auto gemm_m = bk;
  auto gemm_k = bc;

  long Cb_step = Cb;

  //std::cout << "gemm_n gemm_m gemm_k = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;

  std::unique_ptr<unsigned long long[]> A_offsets, B_offsets;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>) brgemm_tpp, brgemm2_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>) zero_tpp;

  zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);
  /* n,m,k, stride_b, stride_a, ldb, lda, ldc, beta, a_trans, unroll_hint because of the row-major */
  if ((R == 1 && S == 1) || (conv_cfg.avoid_fmas_in_rim == 1)) {
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
  long w_step = ofw;
  long r_step = R;
  long s_step = S;

  if (conv_cfg.avoid_fmas_in_rim == 1) {
    r_step = 1;
    s_step = 1;
  }

#ifdef VERBOSE
  std::cout << "debug: N n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " " << n_step << " " << Cb << " " << c_step << " "
                                                                                                << Kb << " " << k_step << " " << ofh << " " << h_step << " "
                                                                                                << ofw << " " << w_step << " " << R << " " << r_step << " "
                                                                                                << S << " " << s_step << " " << std::endl;

  std::cout << "conv_pad_h_out conv_pad_w_out = " << conv_pad_h_out << " " << conv_pad_w_out << std::endl;
  std::cout << "avoid fmas in rim = " <<  conv_cfg.avoid_fmas_in_rim << std::endl;
#endif

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, false},//, true},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      conv_fwd_loop_specs_str);


#ifdef VERBOSE
std::cout << "Setting up the bn in conv/bn fusion" << std::endl;
#endif

  auto zero_bk_tpp = SCOPEIT(SetZeroTPP<float>(bk), EW_ZERO);

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bk, bk, bk), EW_ADD); /* 1, bc because of row-major for unary */

  auto reduce_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size, bk, bk, bk)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto mean_var_tpp = SCOPEIT(MeanVarTPP<float>(bk, scale), EW_MEAN_VAR);

  auto coeffs_tpp = SCOPEIT(BatchNormStatCoeffsTPP<float>(bk, eps), NORMALIZE);

  auto zero_hp_tpp = SCOPEIT(SetZeroTPP<T>((bn_pad_h_out * bn_ofwp), bk, bk), EW_ZERO); /* (pad_h_out * ofwp), bc because of row-major for unary */

  auto zero_wp_tpp = SCOPEIT(SetZeroTPP<T>(bn_pad_w_out, bk, bk), EW_ZERO);          /* pad_w_out, bc because of row-major for unary */

  auto normalize_tpp = SCOPEIT((BatchNormFwdScaleTPP<T,T>(bk, spatial_block_size, relu, eltwise)), NORMALIZE);

  char nkb_loop_specs_str[256] = "ab"; // "AB";
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

          if (conv_cfg.avoid_fmas_in_rim == 0) {

            if (i_c == 0 && i_r == 0 && i_s == 0) {
              zero_tpp(output_off[i_n][i_k][i_h][i_w]);
            }

            brgemm_tpp(inp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                       weight    [i_k][i_c][i_r][i_s][0],
                       output_off[i_n][i_k][i_h]                 [i_w],
                       B_offsets.get(), A_offsets.get(),
                       Cb_step * r_step * s_step,
                       true);
            if (i_c == Cb - c_step && i_r == R - r_step && i_s == S - s_step) {
              //!!! place to compute partial stats
            }
          } else { /* for if conv_cfg.avoid_fmas_in_rim == 0 */
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
          } /* for if-else conv_cfg.avoid_fmas_in_rim == 0 */
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});
    } /* end of the scope with recorded parallel for */
  } /* end of the conv_fwd scope */

//#if 0
/*
          DECL_VLA_PTR_PT_EXT(T,     output_off, [Kb][ofhp][ofwp][bk],   t_CO, (conv_pad_h_out * ofwp * bk + conv_pad_w_out * bk));
for (int i = 0; i < 20; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(output_off[0][0][0][0][i])), &tmp, 1);
                      printf("inp[%d] = %u = %f\n", i, *(unsigned short*)(&output_off[0][0][0][0][i]), tmp);
                    } else
                      printf("inp[%d] = %f\n", i, output_off[0][0][0][0][i]);
}
*/
/*
printf("t_CO data ptr = %p \n", t_CO.data_ptr<T>());
printf("CONV_OUT data ptr = %p \n", CONV_OUT.data_ptr<T>());
printf("t_BI data ptr = %p \n", t_BI.data_ptr<T>());
printf("BN_IN data ptr = %p \n", BN_IN.data_ptr<T>());

            DECL_VLA_PTR_PT_EXT(T,     inp_dbg,      [Kb][bn_ifhp][bn_ifwp][bk], t_BI, (hi_start * bn_ifwp + wi_start) * bk);
for (int i = 0; i < 20; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(inp_dbg[0][0][0][0][i])), &tmp, 1);
                      printf("inp_dbg[%d] = %u = %f\n", i, *(unsigned short*)(&inp_dbg[0][0][0][0][i]), tmp);
                    } else
                      printf("inp_dbg[%d] = %f\n", i, inp_dbg[0][0][0][0][i]);
}
*/

#ifdef VERBOSE
std::cout << "Running bn part in conv/bn fusion" << std::endl;
#endif

  if (training) {
    {
      RECORD_SCOPE(fusedbtlnk_bn_fwd_reduce, {});
      {
        nkb_loop(
          [&](int *ind) {
            const int n = ind[0], kb = ind[1];

            DECL_VLA_PTR_PT_EXT(T,     inp,      [Kb][bn_ifhp][bn_ifwp][bk], t_BI, (hi_start * bn_ifwp + wi_start) * bk);
            DECL_VLA_PTR_PT_EXT(float, sum_N,    [N][bk],              t_scratch, sum_N_offset);
            DECL_VLA_PTR_PT_EXT(float, sumsq_N,  [N][bk],              t_scratch, sumsq_N_offset);

            zero_bk_tpp(sum_N  [kb][n]);
            zero_bk_tpp(sumsq_N[kb][n]);

            LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*bk], 64);

            if (!use_hw_blocking) {
              for (int hi = 0; hi < H; hi++) {
                for (int w = 0; w < W; w += spatial_block_size) {
                  reduce_tpp(inp[n][kb][hi][w], &lcl_sum_X_X2[0]);
                  helper_add_tpp(sum_N  [kb][n], &lcl_sum_X_X2[0],  sum_N  [kb][n] );
                  helper_add_tpp(sumsq_N[kb][n], &lcl_sum_X_X2[bk], sumsq_N[kb][n] );
                }
              }
            } else {
              for(int hwb=0; hwb < num_HW_blocks; hwb++){
                int hi = (hwb*(H*W/num_HW_blocks))/W;
                int w  = (hwb*(H*W/num_HW_blocks))%W;
                reduce_tpp(inp[n][kb][hi][w], &lcl_sum_X_X2[0]);
                helper_add_tpp(sum_N  [kb][n], &lcl_sum_X_X2[0],  sum_N  [kb][n] );
                helper_add_tpp(sumsq_N[kb][n], &lcl_sum_X_X2[bk], sumsq_N[kb][n] );
              }
            }
          },
          [&]() {},
          [&]() {});
      } /* end of the scope with recorded parallel for */
    } /* end of the bn_fwd_reduce scope */

    {
      RECORD_SCOPE(fusedbtlnk_bn_fwd_stats, {});
      {
        kb_loop(
          [&](int *ind) {
            const int kb = ind[0];

            DECL_VLA_PTR_PT    (float, sum_X_X2, [Kb][bk], t_scratch);
            DECL_VLA_PTR_PT_EXT(float, sum_N,    [N][bk],  t_scratch, sum_N_offset);
            DECL_VLA_PTR_PT_EXT(float, sumsq_N,  [N][bk],  t_scratch, sumsq_N_offset);
            DECL_VLA_PTR_PT    (float, mean,     [bk],     t_BM);
            DECL_VLA_PTR_PT    (float, var,      [bk],     t_BV);

            zero_bk_tpp(sum_X_X2[0][kb]);
            zero_bk_tpp(sum_X_X2[1][kb]);

            for(int ni = 0; ni < N; ni++){
              helper_add_tpp(sum_X_X2[0][kb], sum_N  [kb][ni],  sum_X_X2[0][kb]);
              helper_add_tpp(sum_X_X2[1][kb], sumsq_N[kb][ni],  sum_X_X2[1][kb]);
            }

            mean_var_tpp( sum_X_X2[0][kb], sum_X_X2[1][kb], mean[kb], var[kb]);
          },
          [&]() {},
          [&]() {});
      } /* end of the scope with recorded parallel for */
    } /* end of the bn_fwd_stats scope */
  } /* end of if (training) for computing the stats */

/*
          DECL_VLA_PTR_PT    (T,             dbg_bn2_out,      [Kb][bn_ofhp][bn_ofwp][bk], t_BO);
for (int i = 0; i < 100; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(dbg_bn2_out[0][0][0][0][i])), &tmp, 1);
                      printf("BO before[%d] = %u = %f\n", i, *(unsigned short*)(&dbg_bn2_out[0][0][0][0][i]), tmp);
                    } else
                      printf("BO before[%d] = %f\n", i, dbg_bn2_out[0][0][0][0][i]);

}
*/

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
/*
          if (n == 0 && kb == 0) {
            for (int i = 0; i < 10; i++)
              printf("s[%d] = %f b[%d] = %f \n", i, s[i], i, b[i]);
          }
*/
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
/*
          if (n == 0 && kb == 0 && hi == 0 && wb == 0) {
            for (int i = 0; i < 10; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(inp[n][kb][hi][wb*(W/num_W_blocks)][i])), &tmp, 1);
                      printf("bn inp[%d] = %u = %f\n", i, *(unsigned short*)(&inp[n][kb][hi][wb*(W/num_W_blocks)][i]), tmp);
                    } else
                      printf("bn inp[%d] = %f\n", i, inp[n][kb][hi][wb*(W/num_W_blocks)][i]);
            }
            for (int i = 0; i < 10; i++)
              printf("gamma[%d] = %f beta[%d] = %f \n", i, gamma[kb][i], i, beta[kb][i]);
          }
*/
                normalize_tpp(inp[n][kb][hi][wb*(W/num_W_blocks)], &s[0], &b[0], gamma[kb], beta[kb],
                                eltwise ? inp_add[n][kb][hi][wb*(W/num_W_blocks)] : NULL,
                                out[n][kb][ho][wo_start + wb*(W/num_W_blocks)],
                                relu ? relumask[n][kb][ho][wo_start + wb*(W/num_W_blocks)] : NULL);
/*
          if (n == 0 && kb == 0 && hi == 0 && wb == 0) {
            for (int i = 0; i < 10; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(out[n][kb][ho][wo_start + wb*(W/num_W_blocks)][i])), &tmp, 1);
                      printf("bn out[%d] = %u = %f\n", i, *(unsigned short*)(&out[n][kb][ho][wo_start + wb*(W/num_W_blocks)][i]), tmp);
                    } else
                      printf("bn out[%d] = %f\n", i, out[n][kb][ho][wo_start + wb*(W/num_W_blocks)][i]);
            }
          }
*/
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
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_fwd_scale scope */
//#endif

/*
printf("t_CO data ptr = %p \n", t_CO.data_ptr<T>());
printf("CONV_OUT data ptr = %p \n", CONV_OUT.data_ptr<T>());

          DECL_VLA_PTR_PT_EXT(T,     output_end_off, [Kb][ofhp][ofwp][bk],   CONV_OUT, (conv_pad_h_out * ofwp * bk + conv_pad_w_out * bk));
for (int i = 0; i < 20; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(output_end_off[0][0][0][0][i])), &tmp, 1);
                      printf("CO out in the end[%d] = %u = %f\n", i, *(unsigned short*)(&output_end_off[0][0][0][0][i]), tmp);
                    } else
                      printf("CO out in the end[%d] = %f\n", i, output_end_off[0][0][0][0][i]);
}

          DECL_VLA_PTR_PT    (T,             dbg_bn_out,      [Kb][bn_ofhp][bn_ofwp][bk], t_BO);
for (int i = 0; i < 100; i++) {
                    if (sizeof(T) == 2) {
                      float tmp = 0.0;
                      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(dbg_bn_out[0][0][0][0][i])), &tmp, 1);
                      printf("BO out in the end[%d] = %u = %f\n", i, *(unsigned short*)(&dbg_bn_out[0][0][0][0][i]), tmp);
                    } else
                      printf("BO out in the end[%d] = %f\n", i, dbg_bn_out[0][0][0][0][i]);

}
*/

} /* end of the scope for conv1 + bn1 */

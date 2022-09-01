RECORD_FUNCTION("batchnorm_fwd", std::vector<c10::IValue>());

/*        ( input, input_add, weight, bias, mean, var ) = inputs */

//#define VERBOSE

auto t_I  = inputs[0]; // [N][CP][H][W][bc]
at::Tensor t_IA, t_W, t_B, t_M, t_V;
if (eltwise) {
  t_IA = inputs[1];
  t_W  = inputs[2];
  t_B  = inputs[3];
  t_M  = inputs[4];
  t_V  = inputs[5];
} else {
  t_IA = at::empty({0},  torch::TensorOptions().dtype(t_I.dtype()));
  t_W  = inputs[1];
  t_B  = inputs[2];
  t_M  = inputs[3];
  t_V  = inputs[4];
}

const long pad_h_in  = padding[0];
const long pad_w_in  = padding[1];
const long pad_h_out = padding[2];
const long pad_w_out = padding[3];

auto sizes = t_I.sizes();
const long N  = sizes[0];
const long CP = sizes[1];
const long H  = sizes[2] - 2 * pad_h_in;
const long W  = sizes[3] - 2 * pad_w_in;
const long bc = sizes[4];

const long hi_start      = pad_h_in;
const long wi_start      = pad_w_in;
const long ifhp          = H + 2 * pad_h_in;
const long ifwp          = W + 2 * pad_w_in;

const long ho_start      = pad_h_out;
const long ho_end        = ho_start + H;
const long wo_start      = pad_w_out;
const long wo_end        = wo_start + W;
const long ofhp          = H + 2 * pad_h_out;
const long ofwp          = W + 2 * pad_w_out;

const float scale = 1.0f /((float)N * H * W);

std::vector<long> output_size  {N, CP, ofhp, ofwp, bc};
std::vector<long> relumask_size{N, CP, ofhp, ofwp, bc/BITS_PER_CHAR};

auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));

auto t_relu_mask = at::empty(relumask_size, torch::TensorOptions().dtype(at::kByte));

const long sum_N_offset          = LIBXSMM_UP2(CP * 2 * bc, 64);
const long sumsq_N_offset        = LIBXSMM_UP2(sum_N_offset + CP * N * bc, 64);

const long dbeta_N_offset        = LIBXSMM_UP2(CP * N * bc, 64);

const long full_fwd_scratch_size = sumsq_N_offset + LIBXSMM_UP2((size_t)CP * (size_t)N * (size_t)bc, 64);
const long full_bwd_scratch_size = dbeta_N_offset + LIBXSMM_UP2(CP * N * bc, 64);
const long full_scratch_size     = std::max(full_fwd_scratch_size, full_bwd_scratch_size);
std::vector<long> scratch_size{full_scratch_size};
auto t_scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kFloat));

bool use_hw_blocking = true;

const long num_HW_blocks = (H > W ? H : W);
const long num_W_blocks  = (W % 64 == 0 ? W / 64 : 1);

long spatial_block_size = 0;

if (pad_h_in != 0 || pad_w_in != 0 || pad_h_out != 0 || pad_w_out != 0 ) {
  use_hw_blocking    = false; /* alternative is w blocking ([w, bc] blocks) */
  spatial_block_size = W / num_W_blocks;
} else {
  use_hw_blocking    = true; /* using [hw, bc] blocks */
  spatial_block_size = H * W / num_HW_blocks;
}

#ifdef VERBOSE
std::cout << "padding = " << padding << std::endl;
std::cout << "size of T = " << sizeof(T) << std::endl;
std::cout << "output_size = " << output_size << std::endl;
std::cout << "t_I sizes = " << t_I.sizes() << std::endl;
std::cout << "use_hw_blocking = " << use_hw_blocking << std::endl;
#endif

{

#ifndef THREADED_LOOPS
  DECL_VLA_PTR_PT_EXT(T,             inp,      [CP][ifhp][ifwp][bc], t_I, (hi_start * ifwp + wi_start) * bc);
  DECL_VLA_PTR_PT    (float,         gamma,    [bc],                 t_W);
  DECL_VLA_PTR_PT    (float,         beta,     [bc],                 t_B);
  DECL_VLA_PTR_PT    (float,         mean,     [bc],                 t_M);
  DECL_VLA_PTR_PT    (float,         var,      [bc],                 t_V);
  DECL_VLA_PTR_PT_EXT(T,             inp_add,  [CP][ifhp][ifwp][bc], t_IA, (hi_start * ifwp + wi_start) * bc);

  DECL_VLA_PTR_PT    (T,             out,      [CP][ofhp][ofwp][bc], t_O);
  DECL_VLA_PTR_PT    (unsigned char, relumask, [CP][ofhp][ofwp][bc/BITS_PER_CHAR], t_relu_mask);

  DECL_VLA_PTR_PT    (float, sum_X_X2, [CP][bc],      t_scratch);
  DECL_VLA_PTR_PT_EXT(float, sums_N,   [N][2*bc],     t_scratch, sum_N_offset);
  //DECL_VLA_PTR_PT_EXT(float, sum_N,    [N][bc],       t_scratch, sum_N_offset);
  //DECL_VLA_PTR_PT_EXT(float, sumsq_N,  [N][bc],       t_scratch, sumsq_N_offset);
#endif

  auto zero_tpp = SCOPEIT(SetZeroTPP<float>(bc), EW_ZERO);

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bc, bc, bc), EW_ADD); /* 1, bc because of row-major for unary */

  auto reduce_beta0_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size, bc, bc, bc, 1)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto reduce_beta1_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size, bc, bc, bc, 0)), EW_RED); /* spatial_block_size, bc because of row-major for unary */

  auto mean_var_tpp = SCOPEIT(MeanVarTPP<float>(bc, scale), EW_MEAN_VAR);

  auto coeffs_tpp = SCOPEIT(BatchNormStatCoeffsTPP<float>(bc, eps), NORMALIZE);

  auto zero_hp_tpp = SCOPEIT(SetZeroTPP<T>((pad_h_out * ofwp), bc, bc), EW_ZERO); /* (pad_h_out * ofwp), bc because of row-major for unary */

  auto zero_wp_tpp = SCOPEIT(SetZeroTPP<T>(pad_w_out, bc, bc), EW_ZERO);          /* pad_w_out, bc because of row-major for unary */

  auto normalize_tpp = SCOPEIT((BatchNormFwdScaleTPP<T,T>(bc, spatial_block_size, relu, eltwise)), NORMALIZE);

#ifdef THREADED_LOOPS
  char ncp_loop_specs_str[256];// = "AB";
  std::strcpy(ncp_loop_specs_str, tuning_string_ncp.c_str());
  const long n_step = 1, cp_step = 1;
  auto ncp_loop = ThreadedLoop<2>({
      LoopSpecs{0, N,  n_step,  {/*l1_k_step, l0_k_step*/}},   // Logical N  loop specs
      LoopSpecs{0, CP, cp_step, {/*l1_n_step, l0_n_step*/}}},  // Logical CP loop specs
      ncp_loop_specs_str);

  char cp_loop_specs_str[256];// = "A";
  std::strcpy(cp_loop_specs_str, tuning_string_cp.c_str());
  auto cp_loop = ThreadedLoop<1>({
      LoopSpecs{0, CP, cp_step, {/*l1_k_step, l0_k_step*/}}},  // Logical CP loop specs
      cp_loop_specs_str);
#endif

  if (training) {
    {
      RECORD_SCOPE(bn_fwd_reduce, {});//{t_HS, t_Wq_V});
      {
#ifdef THREADED_LOOPS
        ncp_loop(
          [&](int *ind) {
            const int n = ind[0], cp = ind[1];

            DECL_VLA_PTR_PT_EXT(T,     inp,      [CP][ifhp][ifwp][bc], t_I,      (hi_start * ifwp + wi_start) * bc);
            DECL_VLA_PTR_PT_EXT(float, sums_N,   [N][2*bc],            t_scratch, sum_N_offset);

            if (!use_hw_blocking) {
              for (int hi = 0; hi < H; hi++) {
                for (int w = 0; w < W; w += spatial_block_size) {
                  if (hi == 0 && w == 0)
                    reduce_beta0_tpp(inp[n][cp][hi][w], sums_N[cp][n]);
                  else
                    reduce_beta1_tpp(inp[n][cp][hi][w], sums_N[cp][n]);
                }
              }
            } else {
              for(int hwb=0; hwb < num_HW_blocks; hwb++){
                int hi = (hwb*(H*W/num_HW_blocks))/W;
                int w  = (hwb*(H*W/num_HW_blocks))%W;
                if (hwb == 0)
                  reduce_beta0_tpp(inp[n][cp][hi][w], sums_N[cp][n]);
                else
                  reduce_beta1_tpp(inp[n][cp][hi][w], sums_N[cp][n]);
              }
            }
          },
          [&]() {},
          [&]() {});
#else /* THREADED_LOOPS */
        RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
        for (int n = 0; n < N; n++) {
          for (int cp = 0; cp < CP; cp++) {

            if (!use_hw_blocking) {
              for (int hi = 0; hi < H; hi++) {
                for (int w = 0; w < W; w += spatial_block_size) {
                  if (hi == 0 && w == 0)
                    reduce_beta0_tpp(inp[n][cp][hi][w], sums_N[cp][n]);
                  else
                    reduce_beta1_tpp(inp[n][cp][hi][w], sums_N[cp][n]);
                }
              }
            } else {
              for(int hwb=0; hwb < num_HW_blocks; hwb++){
                int hi = (hwb*(H*W/num_HW_blocks))/W;
                int w  = (hwb*(H*W/num_HW_blocks))%W;
                if (hwb == 0)
                  reduce_beta0_tpp(inp[n][cp][hi][w], sums_N[cp][n]);
                else
                  reduce_beta1_tpp(inp[n][cp][hi][w], sums_N[cp][n]);
              }
            }
          } /* end of cp loop */
        } /* end of n loop */
#endif /* THREADED_LOOPS */
      } /* end of the scope with recorded parallel for */
    } /* end of the bn_fwd_reduce scope */

    {
      RECORD_SCOPE(bn_fwd_stats, {});
      {
#ifdef THREADED_LOOPS
        cp_loop(
          [&](int *ind) {
            const int cp = ind[0];

            DECL_VLA_PTR_PT    (float, sum_X_X2, [CP][bc],  t_scratch);
            DECL_VLA_PTR_PT_EXT(float, sums_N,   [N][2*bc], t_scratch, sum_N_offset);
            DECL_VLA_PTR_PT    (float, mean,     [bc],      t_M);
            DECL_VLA_PTR_PT    (float, var,      [bc],      t_V);

            zero_tpp(sum_X_X2[0][cp]);
            zero_tpp(sum_X_X2[1][cp]);

            for(int ni = 0; ni < N; ni++){
              helper_add_tpp(sum_X_X2[0][cp], &(sums_N[cp][ni][0]),  sum_X_X2[0][cp]);
              helper_add_tpp(sum_X_X2[1][cp], &(sums_N[cp][ni][bc]), sum_X_X2[1][cp]);
            }

            mean_var_tpp( sum_X_X2[0][cp], sum_X_X2[1][cp], mean[cp], var[cp]);
          },
          [&]() {},
          [&]() {});
#else /* THREADED_LOOPS */
        RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
        for (int cp = 0; cp < CP; cp++) {
          zero_tpp(sum_X_X2[0][cp]);
          zero_tpp(sum_X_X2[1][cp]);

          for(int ni = 0; ni < N; ni++){
              helper_add_tpp(sum_X_X2[0][cp], &(sums_N[cp][ni][0]),  sum_X_X2[0][cp]);
              helper_add_tpp(sum_X_X2[1][cp], &(sums_N[cp][ni][bc]), sum_X_X2[1][cp]);
          }

          mean_var_tpp( sum_X_X2[0][cp], sum_X_X2[1][cp], mean[cp], var[cp]);

        } /* end of cp loop */
#endif /* THREADED_LOOPS */
      } /* end of the scope with recorded parallel for */
    } /* end of the bn_fwd_stats scope */
  } /* end of if (training) for computing the stats */

  {
    RECORD_SCOPE(bn_fwd_scale, {});
    {
#ifdef THREADED_LOOPS
      ncp_loop(
        [&](int *ind) {
          const int n = ind[0], cp = ind[1];

          DECL_VLA_PTR_PT_EXT(T,             inp,      [CP][ifhp][ifwp][bc], t_I, (hi_start * ifwp + wi_start) * bc);
          DECL_VLA_PTR_PT_EXT(T,             inp_add,  [CP][ifhp][ifwp][bc], t_IA, (hi_start * ifwp + wi_start) * bc);
          DECL_VLA_PTR_PT    (T,             out,      [CP][ofhp][ofwp][bc], t_O);
          DECL_VLA_PTR_PT    (unsigned char, relumask, [CP][ofhp][ofwp][bc/BITS_PER_CHAR], t_relu_mask);
          DECL_VLA_PTR_PT    (float,         gamma,    [bc],                 t_W);
          DECL_VLA_PTR_PT    (float,         beta,     [bc],                 t_B);
          DECL_VLA_PTR_PT    (float,         mean,     [bc],     t_M);
          DECL_VLA_PTR_PT    (float,         var,      [bc],     t_V);

          LIBXSMM_ALIGNED(float s[bc], 64);
          LIBXSMM_ALIGNED(float b[bc], 64);

          coeffs_tpp(mean[cp], var[cp], &s[0], &b[0]);

          if (!use_hw_blocking) {

            if (pad_h_out != 0) {
              zero_hp_tpp(out[n][cp][0][0]);
            }

            for (int hi = 0, ho = ho_start; hi < H; hi++, ho++) {
              /* zeroing out starting [0, wo_start) x bc and [wo_end, ofwp] x bc blocks for fixed ho */
              if (pad_w_out != 0) {
                zero_wp_tpp(out[n][cp][ho][0]);
              }

              for (int wb = 0; wb < num_W_blocks; wb++) {
                normalize_tpp(inp[n][cp][hi][wb*(W/num_W_blocks)], &s[0], &b[0], gamma[cp], beta[cp],
                                eltwise ? inp_add[n][cp][hi][wb*(W/num_W_blocks)] : NULL,
                                out[n][cp][ho][wo_start + wb*(W/num_W_blocks)],
                                relu ? relumask[n][cp][ho][wo_start + wb*(W/num_W_blocks)] : NULL);
              }
              /* zeroing out ending [wo_end, ofwp] x bc block for fixed ho */
              if (pad_w_out != 0) {
                zero_wp_tpp(out[n][cp][ho][wo_end]);
              }
            }
            /* zeroing out strip [ho_end, ofhp) x ofwp x bc */
            if (pad_h_out != 0) {
              zero_hp_tpp(out[n][cp][ho_end][0]);
            }

          } else {
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int hi = (hwb*(H*W/num_HW_blocks))/W;
              int ho = hi;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
              normalize_tpp(inp[n][cp][hi][w], &s[0], &b[0], gamma[cp], beta[cp],
                              eltwise ? inp_add[n][cp][hi][w] : NULL,
                              out[n][cp][ho][w],
                              relu ? relumask[n][cp][ho][w] : NULL);
            }
          } /* if-else for the presence of padding */
        },
        [&]() {},
        [&]() {});
#else /* THREADED_LOOPS */
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < N; n++) {
        for (int cp = 0; cp < CP; cp++) {

          LIBXSMM_ALIGNED(float s[bc], 64);
          LIBXSMM_ALIGNED(float b[bc], 64);

          coeffs_tpp(mean[cp], var[cp], &s[0], &b[0]);

          if (!use_hw_blocking) {
            if (pad_h_out != 0) {
              zero_hp_tpp(out[n][cp][0][0]);
            }
            for (int hi = 0, ho = ho_start; hi < H; hi++, ho++) {
              /* zeroing out starting [0, wo_start) x bc and [wo_end, ofwp] x bc blocks for fixed ho */
              if (pad_w_out != 0) {
                zero_wp_tpp(out[n][cp][ho][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                normalize_tpp(inp[n][cp][hi][wb*(W/num_W_blocks)], &s[0], &b[0], gamma[cp], beta[cp],
                                eltwise ? inp_add[n][cp][hi][wb*(W/num_W_blocks)] : NULL,
                                out[n][cp][ho][wo_start + wb*(W/num_W_blocks)],
                                relu ? relumask[n][cp][ho][wo_start + wb*(W/num_W_blocks)] : NULL);
              }
              /* zeroing out ending [wo_end, ofwp] x bc block for fixed ho */
              if (pad_w_out != 0) {
                zero_wp_tpp(out[n][cp][ho][wo_end]);
              }
            }
            /* zeroing out strip [ho_end, ofhp) x ofwp x bc */
            if (pad_h_out != 0) {
              zero_hp_tpp(out[n][cp][ho_end][0]);
            }

          } else {
            for(int hwb = 0; hwb < num_HW_blocks; hwb++) {
              int hi = (hwb*(H*W/num_HW_blocks))/W;
              int ho = hi;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
              normalize_tpp(inp[n][cp][hi][w], &s[0], &b[0], gamma[cp], beta[cp],
                              eltwise ? inp_add[n][cp][hi][w] : NULL,
                              out[n][cp][ho][w],
                              relu ? relumask[n][cp][ho][w] : NULL);
            }
          } /* if-else for the presence of padding */
        } /* end of cp loop */
      } /* end of n loop */
#endif /* THREADED_LOOPS */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_fwd_scale scope */

} /* end of the dummy scope */

//return std::vector<at::Tensor>({t_O, t_relu_mask, inputs[6]});
return std::vector<at::Tensor>({t_O, t_relu_mask, t_scratch});

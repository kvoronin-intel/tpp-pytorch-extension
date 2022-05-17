RECORD_FUNCTION("batchnorm_fwd", std::vector<c10::IValue>());

/*        ( input, input_add, weight, bias, mean, var, invstd ) = inputs */

auto t_I  = inputs[0]; // [N][CP][H][W][bc]
auto t_IA = inputs[1];
auto t_W  = inputs[2];
auto t_B  = inputs[3];
auto t_M  = inputs[4];
auto t_V  = inputs[5];
auto t_IV = inputs[6]; /* should be unused */

long pad_h_in  = padding[0];
long pad_w_in  = padding[1];
long pad_h_out = padding[2];
long pad_w_out = padding[3];

auto sizes = t_I.sizes();
long N  = sizes[0];
long CP = sizes[1];
long H  = sizes[2] - 2 * pad_h_in;
long W  = sizes[3] - 2 * pad_w_in;
long bc = sizes[4];
//long C  = CP * bc;

long ifhp = H + 2 * pad_h_in;
long ifwp = W + 2 * pad_w_in;
long ofhp = H + 2 * pad_h_out;
long ofwp = W + 2 * pad_w_out;

const float scale = 1.0f /((float)N * H * W);

std::vector<long> output_size{N, CP, ofhp, ofwp, bc};

std::cout << "output_size = " << output_size << std::endl;

auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));

auto t_relu_mask = at::empty(t_O.sizes(), torch::TensorOptions().dtype(at::kShort));

//  sum_N_offset   = LIBXSMM_UP2(res.CP * 2 * res.bc, 64);
//  sumsq_N_offset = LIBXSMM_UP2(sum_N_offset + res.CP * res.N * res.bc, 64);
//  res.scratch_size =  sizeof(float) * ( sumsq_N_offset /*sum_X_X2 + sumsq_N */ + LIBXSMM_UP2((size_t)res.CP * (size_t)res.N * (size_t)res.bc, 64) /* sumsq_N */ );

//  /* init scratch */
//  dbeta_N_offset = LIBXSMM_UP2(res.CP * res.N * res.bc, 64);
//  res.scratch_size =  sizeof(float) * ( dbeta_N_offset /* dbeta_N*/ + LIBXSMM_UP2(res.CP * res.N * res.bc, 64) /*dgamma_N */ );

long sum_N_offset          = LIBXSMM_UP2(CP * 2 * bc, 64);
long sumsq_N_offset        = LIBXSMM_UP2(sum_N_offset + CP * N * bc, 64);
long full_fwd_scratch_size = sumsq_N_offset + LIBXSMM_UP2((size_t)CP * (size_t)N * (size_t)bc, 64);

long dbeta_N_offset        = LIBXSMM_UP2(CP * N * bc, 64);
long full_bwd_scratch_size = dbeta_N_offset + LIBXSMM_UP2(CP * N * bc, 64);

long full_scratch_size     = std::max(full_fwd_scratch_size, full_bwd_scratch_size);

// FIXME: Save scratch somewhere to not allocate each time
std::vector<long> scratch_size{full_scratch_size};

auto scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kFloat));

bool use_hw_blocking = true;

long num_HW_blocks = (H > W ? H : W);
long num_W_blocks  = (W % 64 == 0 ? W / 64 : 1);

long spatial_block_size = 0;

if (pad_h_in != 0 || pad_w_in != 0 || pad_h_out != 0 || pad_w_out != 0 ) {
  use_hw_blocking    = false; /* alternative is w blocking ([w, bc] blocks) */
  spatial_block_size = W / num_W_blocks;
} else {
  use_hw_blocking    = true; /* using [hw, bc] blocks */
  spatial_block_size = H * W / num_HW_blocks;
}

{
  DECL_VLA_PTR_PT    (T,             inp,      [N][CP][ifhp][ifwp][bc], t_I);
  DECL_VLA_PTR_PT    (float,         gamma,    [CP][bc],                t_W);
  DECL_VLA_PTR_PT    (float,         beta,     [CP][bc],                t_B);
  DECL_VLA_PTR_PT    (float,         mean,     [CP][bc],                t_M);
  DECL_VLA_PTR_PT    (float,         var,      [CP][bc],                t_V);
  DECL_VLA_PTR_PT    (T,             inp_add,  [N][CP][ifhp][ifwp][bc], t_IA);

  DECL_VLA_PTR_PT    (T,             out,      [N][CP][ofhp][ofwp][bc], t_O);
  DECL_VLA_PTR_PT    (unsigned char, relumask, [N][CP][ofhp][ofwp][bc], t_relu_mask);

  DECL_VLA_PTR_PT    (float, sum_X_X2, [2][CP][bc],       scratch);
  DECL_VLA_PTR_PT_EXT(float, sum_N,    [CP][N][bc],       scratch, sum_N_offset);
  DECL_VLA_PTR_PT_EXT(float, sumsq_N,  [CP][N][bc],       scratch, sumsq_N_offset);

  auto zero_tpp = SCOPEIT(SetZeroTPP<float>(bc), EW_ZERO);

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bc, bc, bc), EW_ADD);

  auto reduce_tpp = SCOPEIT((ReduceColsTPP<T, float>(spatial_block_size, bc, bc, bc)), EW_RED);

  auto mean_var_tpp = SCOPEIT(MeanVarTPP<float>(bc, scale), EW_MEAN_VAR);

  auto coeffs_tpp = SCOPEIT(BatchNormCoeffsTPP<float>(bc, eps), NORMALIZE);

  auto normalize_tpp = SCOPEIT((BatchNormFwdScale<T,T>(bc, spatial_block_size, relu, eltwise)), NORMALIZE);
/*
  auto reduce_tpp = SCOPEIT(UnaryTPP(
            H * W / num_HW_blocks, bc, bc, bc,
            XsmmDtype<T>(), LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD), EW_RED);
*/
  {
    RECORD_SCOPE(bn_reduce, {});//{t_HS, t_Wq_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < N; n++) {
        for (int cp = 0; cp < CP; cp++) {
          zero_tpp(sum_N  [cp][n][0]);
          zero_tpp(sumsq_N[cp][n][0]);

          LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*bc], 64);

          if (!use_hw_blocking) {
            printf("First part of parallel for is not implemented for w blocking\n");
            exit(-1);
          } else {
            int hwb = 0, hi = 0, w = 0;
            for(hwb=0; hwb < num_HW_blocks; hwb++){
              hi = (hwb*(H*W/num_HW_blocks))/W;
              w  = (hwb*(H*W/num_HW_blocks))%W;
              reduce_tpp(inp[n][cp][hi][w][0], &lcl_sum_X_X2[0]);
              helper_add_tpp(sum_N  [cp][n][0], &lcl_sum_X_X2[0],  sum_N  [cp][n][0] );
              helper_add_tpp(sumsq_N[cp][n][0], &lcl_sum_X_X2[bc], sumsq_N[cp][n][0] );
            }
          }
        } /* end of cp loop */
      } /* end of n loop */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_reduce scope */

  {
    RECORD_SCOPE(bn_stats, {});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int cp = 0; cp < CP; cp++) {
        zero_tpp(sum_X_X2[0][cp][0]);
        zero_tpp(sum_X_X2[1][cp][0]);

        //int cb, ni;
        for(int ni = 0; ni < N; ni++){
            helper_add_tpp(sum_X_X2[0][cp][0], sum_N[cp][ni][0],    sum_X_X2[0][cp][0]);
            helper_add_tpp(sum_X_X2[1][cp][0], sumsq_N[cp][ni][0],  sum_X_X2[1][cp][0]);
        }

        mean_var_tpp( sum_X_X2[0][cp][0], sum_X_X2[1][cp][0], mean[cp][0], var[cp][0]);
#if 0
        for(int cb = 0; cb < bc; cb++){
          //mean[cp*bc + cb] = (LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, cb, CP, bc)) * scale;                 /* E[X] */
          //var [cp*bc + cb] = ((LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, cb, CP, bc)) * scale) - (mean[cp*bc + cb]*mean[cp*bc + cb]);
          mean[cp*bc + cb] = sum_X_X2[0][cp][cb] * scale;                 /* E[X] */
          var [cp*bc + cb] = sum_X_X2[1][cp][cb] * scale - (mean[cp*bc + cb]*mean[cp*bc + cb]);
        }
#endif

      } /* end of cp loop */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_stats scope */

  {
    RECORD_SCOPE(bn_scale, {});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < N; n++) {
        for (int cp = 0; cp < CP; cp++) {

          int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0;

          LIBXSMM_ALIGNED(float s[bc], 64);
          LIBXSMM_ALIGNED(float b[bc], 64);

          coeffs_tpp(mean[cp][0], var[cp][0], &s[0], &b[0]);

          if (!use_hw_blocking) {
            printf("Third part of parallel for is not implemented for w blocking\n");
            exit(-1);
          } else {
            int hwb = 0, hi = 0, w = 0;
            for(hwb=0; hwb < num_HW_blocks; hwb++){
              hi = (hwb*(H*W/num_HW_blocks))/W;
              w  = (hwb*(H*W/num_HW_blocks))%W;
            }
            /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
            //bn_normalize_tpp(Tin* inp, float* s, float* b, float *gamma, float *beta, float *inp_add, Tout* out, unsigned char* relumask);
            normalize_tpp(inp[n][cp][hi][w][0], &s[0], &b[0], gamma[cp][0], beta[cp][0],
                            eltwise ? inp_add[n][cp][hi][w][0] : NULL,
                            out[n][cp][ho][w][0],
                            relu ? relumask[n][cp][ho][w][0] : NULL);
          } /* if-else for the presence of padding */
        } /* end of cp loop */
      } /* end of n loop */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_scale scope */

} /* end of the dummy scope */

return std::vector<at::Tensor>({t_O, t_relu_mask});

#if 0
// B - Batch size
// S - Max seq len
// N - Number of attention heads
// H - Head size
auto t_Wq = inputs[0]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bq = inputs[1]; // [HS]
auto t_Wk = inputs[2]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bk = inputs[3]; // [HS]
auto t_Wv = inputs[4]; // [HS][NH] --> [N1][N2][H2][H1]
auto t_Bv = inputs[5]; // [HS]
auto t_HS = inputs[6]; // [B][S][HS] --> [B][S1][N][S2][H]
auto t_AM = inputs[7]; // Optional [B][S]
auto t_HM = inputs[8]; // Optional [B][N][S][S]
auto t_EHS = inputs[9]; // [B][S][HS] --> [B][S1][N][S2][H]
auto t_EAM = inputs[10]; // Optional [B][S]

auto sizes = t_HS.sizes();
long B = sizes[0];
long S1 = sizes[1];
long N = sizes[2];
long S2 = sizes[3];
long H = sizes[4];
// long NH = N*H;
float one_by_sqrt_H = 1.0 / sqrt(H);
bool null_EHS = false;
bool dt_bf16 = (t_HS.dtype() == at::kBFloat16);
bool bf16_training = (training && dt_bf16);
auto t_EHS_orig = t_EHS;

// std::cout << "B: " << B << " S1: " << S1 << " S2: " << S2 << " N: " << N << "
// H: " << H << std::endl;
if (t_EHS.numel() == 0) {
  null_EHS = true;
  t_EHS = t_HS;
} else {
  t_AM = t_EAM;
}

//#define PRINT_T(x) std::cout << #x << ": " << x << std::endl
auto t_HS_T = t_HS;
auto t_EHS_T = t_EHS;

auto t_Wq_V = wt_tensor_for_fwd(N, H, N, H, t_Wq);
auto t_Wk_V = wt_tensor_for_fwd(N, H, N, H, t_Wk);
auto t_Wv_V = wt_tensor_for_fwd(N, H, N, H, t_Wv);

auto t_QL = t_HS.new_empty({B, S1, N, S2, H});
auto t_QL_T = t_QL;
auto t_KL_TV = t_EHS.new_empty({B, S1, N, H, S2});
if (dt_bf16)
  t_KL_TV = t_KL_TV.view({B, S1, N, H / 2, S2, 2});
auto t_KL_V = t_KL_TV;
auto t_VL_V = t_EHS.new_empty({B, S1, N, S2, H});
if (dt_bf16)
  t_VL_V = t_VL_V.view({B, S1, N, S2 / 2, H, 2});
auto t_VL_TV = t_VL_V;
auto t_AP = t_QL.new_empty({B, S1, N, S1, S2, S2});
auto t_CL = t_AP.new_empty({B, S1, N, S2, H});

auto t_APD = t_AP;
auto t_APD_mask = at::empty({B, S1, N, (S1 * S2 * S2 + 15) / 16}, at::kShort);
if (p > 0 || t_HM.numel() != 0) {
  t_APD = at::empty_like(t_AP);
}

auto t_APD_T = t_APD;

if (bf16_training) {
  t_HS_T = t_HS.new_empty({B, S1, N, H, S2}); // For BWD only
  t_EHS_T =
      null_EHS ? t_HS_T : t_HS.new_empty({B, S1, N, H, S2}); // For BWD only

  t_QL_T = t_HS.new_empty({B, S1, N, H, S2}); // For BWD only
  t_APD_T = t_QL.new_empty({B, S1, N, S1, S2, S2}); // For BWD only
}
if (training) {
  if (dt_bf16) {
    t_KL_V = t_EHS.new_empty({B, S1, N, S2 / 2, H, 2}); // Saved For BWD
    t_VL_TV = t_EHS.new_empty({B, S1, N, H / 2, S2, 2}); // For BWD only
  } else {
    t_KL_V = t_EHS.new_empty({B, S1, N, S2, H}); // Saved For BWD
    t_VL_TV = t_EHS.new_empty({B, S1, N, H, S2}); // For BWD only
  }
}

{
  // float (*QL)[S1][N][S2][H] = (float
  // (*)[S1][N][S2][H])t_QL.data_ptr<float>();
  DECL_VLA_PTR_PT(T, Wq_V, [N][H * H], t_Wq_V);
  DECL_VLA_PTR_PT(T, Wk_V, [N][H * H], t_Wk_V);
  DECL_VLA_PTR_PT(T, Wv_V, [N][H * H], t_Wv_V);
  DECL_VLA_PTR_PT(T, Bq, [H], t_Bq);
  DECL_VLA_PTR_PT(T, Bk, [H], t_Bk);
  DECL_VLA_PTR_PT(T, Bv, [H], t_Bv);
  DECL_VLA_PTR_PT(T, QL, [S1][N][S2 * H], t_QL);
  DECL_VLA_PTR_PT(T, QL_T, [S1][N][H * S2], t_QL_T); // For BWD only
  DECL_VLA_PTR_PT(T, KL_V, [S1][N][S2 * H], t_KL_V);
  DECL_VLA_PTR_PT(T, KL_TV, [S1][N][H * S2], t_KL_TV);
  DECL_VLA_PTR_PT(T, VL_V, [S1][N][S2 * H], t_VL_V);
  DECL_VLA_PTR_PT(T, VL_TV, [S1][N][H * S2], t_VL_TV);
  DECL_VLA_PTR_PT(T, AP, [S1][N][S1][S2 * S2], t_AP);
  DECL_VLA_PTR_PT(T, APD, [S1][N][S1][S2 * S2], t_APD);
  DECL_VLA_PTR_PT(T, APD_T, [S1][N][S1][S2 * S2], t_APD_T); // For BWD only
  DECL_VLA_PTR_PT(
      short, APD_mask, [S1][N][(S1 * S2 * S2 + 15) / 16], t_APD_mask);
  DECL_VLA_PTR_PT(T, CL, [S1][N][S2 * H], t_CL);
  DECL_VLA_PTR_PT(T, HS, [S1][N][S2 * H], t_HS);
  DECL_VLA_PTR_PT(T, HS_T, [S1][N][H * S2], t_HS_T); // for BWD only
  DECL_VLA_PTR_PT(T, EHS, [S1][N][S2 * H], t_EHS);
  DECL_VLA_PTR_PT(T, EHS_T, [S1][N][H * S2], t_EHS_T); // for BWD only
  DECL_VLA_PTR_PT(T, AM, [S1][S2], t_AM);

  auto copy_bias_tpp = SCOPEIT(CpyBiasTPP<T>(S2, H), BIAS);
  auto qkv_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, H, S2 * H, H * H, 1.0, XformTPP::XFORM_NONE_TPP, 0, N)));
  auto xpose_tpp =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  auto k_xpose_tpp_1 = SCOPEIT(
      XformExtTPP<T>(
          S2,
          H,
          training ? XformTPP::XFORM_N2V_TPP : XformTPP::XFORM_XPOSE_N2V_TPP,
          true),
      XPOSE);
  auto kv_xpose_tpp_2 =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_XPOSE_N2V_TPP, true), VNNI);
  auto v_xpose_tpp_1 =
      SCOPEIT(XformExtTPP<T>(S2, H, XformTPP::XFORM_N2V_TPP, true), VNNI);
  auto a_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, float>(
      S2, S2, H, S2 * H, H * S2, 0.0, XformTPP::XFORM_NONE_TPP, 0, 1)));
  auto scale_tpp = SCOPEIT((ScaleTPP<float, float>(S2 * S2)), EW_SCL);
  auto add_mask_tpp = SCOPEIT(AddBiasTPP<T>(S2, S2), EW_ADD);
  auto softmax_fwd_tpp =
      SCOPEIT((SoftMaxFwdTPP<float, T>(S1, S2, S2)), SOFTMAX);
  auto dropout_fwd_tpp = SCOPEIT(DropOutFwdTPP<T>(S1 * S2 * S2, p), DROPOUT);
  auto a_xpose_tpp =
      SCOPEIT(XformExtTPP<T>(S2, S2, XformTPP::XFORM_XPOSE_TPP), XPOSE);
  auto c_gemm_tpp = SCOPEITGEMM((BrgemmExtTPP<T, T>(
      S2, H, S2, S2 * S2, N * S2 * H, 0.0, XformTPP::XFORM_NONE_TPP, 0, S1)));

  {
    RECORD_SCOPE(q_gemm, {t_HS, t_Wq_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nk = 0; nk < N; nk++) {
            if (bf16_training && nk == 0)
              xpose_tpp(N, S2 * H, S2 * H, HS[b][s1][0], HS_T[b][s1][0]);
            copy_bias_tpp(Bq[nk], QL[b][s1][nk]);
            qkv_gemm_tpp(HS[b][s1][0], Wq_V[nk][0], QL[b][s1][nk], N);
            if (bf16_training)
              xpose_tpp(QL[b][s1][nk], QL_T[b][s1][nk]);
          }
        }
      }
    }
  }

  // PRINT_T(t_QL.permute({0,1,3,2,4}).contiguous().view({B,S1*S2,N*H}));

  {
    RECORD_SCOPE(k_gemm, {t_EHS, t_Wk_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nk = 0; nk < N; nk++) {
            T tmp[S2 * H];
            T* tmpp = (training && !bf16_training) ? KL_V[b][s1][nk] : tmp;
            if (!null_EHS && bf16_training && nk == 0)
              xpose_tpp(N, S2 * H, S2 * H, EHS[b][s1][0], EHS_T[b][s1][0]);
            copy_bias_tpp(Bk[nk], tmpp);
            qkv_gemm_tpp(EHS[b][s1][0], Wk_V[nk][0], tmpp, N);
            k_xpose_tpp_1(
                tmpp, KL_V[b][s1][nk]); // KL_V = KL_VT if not training
            if (training)
              kv_xpose_tpp_2(tmpp, KL_TV[b][s1][nk]);
          }
        }
      }
    }
  }
  // PRINT_T(t_EHS);
  // PRINT_T(t_Wk_V.permute({0,1,2,4,3}).contiguous().view({N,N,H,H}));
  // PRINT_T(t_Wk_V);
  // PRINT_T(t_Bk);
  // PRINT_T(t_KL_V.permute({0,1,3,5,2,4}).contiguous().view({B,S1*S2,N*H}));

  {
    RECORD_SCOPE(v_gemm, {t_EHS, t_Wv_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int s1 = 0; s1 < S1; s1++) {
          for (int nk = 0; nk < N; nk++) {
            T tmp[S2 * H];
            T* tmpp = (!dt_bf16) ? VL_V[b][s1][nk] : tmp;
            copy_bias_tpp(Bv[nk], tmpp);
            qkv_gemm_tpp(EHS[b][s1][0], Wv_V[nk][0], tmpp, N);
            v_xpose_tpp_1(tmpp, VL_V[b][s1][nk]);
            if (training)
              kv_xpose_tpp_2(tmpp, VL_TV[b][s1][nk]);
          }
        }
      }
    }
  }
  // Take the dot product between "query" and "key" to get the raw attention
  // scores.
  {
    RECORD_SCOPE(a_gemm, {t_QL, t_KL_TV});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          for (int s11 = 0; s11 < S1; s11++) {
            float AS[S1][S2][S2];
            for (int s21 = 0; s21 < S1; s21++) {
              a_gemm_tpp(QL[b][s11][n], KL_TV[b][s21][n], AS[s21][0], 1);
              scale_tpp(AS[s21][0], AS[s21][0], one_by_sqrt_H);
              if (t_AM.numel() != 0)
                add_mask_tpp(AM[b][s21], AS[s21][0]);
            }
            softmax_fwd_tpp(AS[0][0], AP[b][s11][n][0]);
            if (p > 0) {
              dropout_fwd_tpp(
                  AP[b][s11][n][0],
                  (void*)get_rng_state(),
                  APD[b][s11][n][0],
                  APD_mask[b][s11][n]);
            }
            if (t_HM.numel() != 0) {
              // FIXME: shape of head mask is not correct here yet
              PCL_ASSERT(0, "t_HM used");
              // t_APD[b][s11][n] *= t_HM[b][s11][n];
            }
            if (bf16_training)
              a_xpose_tpp(
                  S1, S2 * S2, S2 * S2, APD[b][s11][n][0], APD_T[b][s11][n][0]);
          }
        }
      }
    }
  }

  {
    RECORD_SCOPE(c_gemm, {t_APD, t_VL_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(3)
      for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
          for (int s11 = 0; s11 < S1; s11++) {
            c_gemm_tpp(APD[b][s11][n][0], VL_V[b][0][n], CL[b][s11][n], S1);
          }
        }
      }
    }
  }
}

return std::vector<at::Tensor>({t_O, t_relu_mask});

#endif


#if 0

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      if (cfg.dtype == 0)
        libxsmm_dnn_bn_fwd_exec_f32( cfg.fwd_cfg, (const float*)input_pt, (const float*)input_add_pt, (const float*)gamma_pt, (const float*)beta_pt, (float*)mean_pt, (float*)var_pt,
                            (float*)output_pt, (unsigned char*)relu_mask_pt, cfg.eps, 0, tid, cfg.scratch, (norm_type == 0 ? LIBXSMM_DNN_BN_FULL_NORM : LIBXSMM_DNN_BN_SCALE_ONLY) );
      else
        libxsmm_dnn_bn_fwd_exec_bf16( cfg.fwd_cfg, (const libxsmm_bfloat16*)input_pt, (const libxsmm_bfloat16*)input_add_pt, (const float*)gamma_pt, (const float*)beta_pt, (float*)mean_pt, (float*)var_pt,
                            (libxsmm_bfloat16*)output_pt, (unsigned char*)relu_mask_pt, cfg.eps, 0, tid, cfg.scratch, (norm_type == 0 ? LIBXSMM_DNN_BN_FULL_NORM : LIBXSMM_DNN_BN_SCALE_ONLY) );
    }
  }


LIBXSMM_API void libxsmm_dnn_bn_fwd_exec_f32( libxsmm_dnn_bn_fwd_config cfg, const float *pinp, const float *pinp_add, const float *pgamma, const float *pbeta, float *mean, float *var, float *pout,
                         unsigned char *prelumask, float eps, int start_tid, int my_tid, void *scratch, libxsmm_dnn_bn_norm_type norm_type ) {

  const libxsmm_blasint N  = cfg.N;
  const libxsmm_blasint CP = cfg.CP;
  const libxsmm_blasint H  = cfg.H;
  const libxsmm_blasint W  = cfg.W;
  const libxsmm_blasint HW = cfg.H * cfg.W;
  const libxsmm_blasint bc = cfg.bc;
  const libxsmm_blasint num_HW_blocks = cfg.num_HW_blocks;
  const libxsmm_blasint num_W_blocks  = cfg.num_W_blocks;

  const libxsmm_blasint hi_start      = cfg.pad_h_in;
  const libxsmm_blasint wi_start      = cfg.pad_w_in;
  const libxsmm_blasint ifhp = cfg.H + 2 * cfg.pad_h_in;
  const libxsmm_blasint ifwp = cfg.W + 2 * cfg.pad_w_in;

  const libxsmm_blasint ho_start      = cfg.pad_h_out;
  const libxsmm_blasint ho_end        = ho_start + cfg.H;
  const libxsmm_blasint wo_start      = cfg.pad_w_out;
  const libxsmm_blasint wo_end        = wo_start + cfg.W;
  const libxsmm_blasint ofhp = cfg.H + 2 * cfg.pad_h_out;
  const libxsmm_blasint ofwp = cfg.W + 2 * cfg.pad_w_out;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_dN = CP * N;
  /* compute chunk size */
  const libxsmm_blasint chunksize_dN = (work_dN % cfg.threads == 0) ?
    (work_dN / cfg.threads) : ((work_dN / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_dN = (ltid * chunksize_dN < work_dN) ? (ltid * chunksize_dN) : work_dN;
  const libxsmm_blasint thr_end_dN = ((ltid + 1) * chunksize_dN < work_dN) ? ((ltid + 1) * chunksize_dN) : work_dN;

  /* number of tasks that could be run in parallel for 1d blocking */
  /* Question: each thread should take a number of full (of length CP chunks) or can we really do a partial split here? */
  const libxsmm_blasint work_C = CP;
  /* compute chunk size */
  const libxsmm_blasint chunksize_C = (work_C % cfg.threads == 0) ?
    (work_C / cfg.threads) : ((work_C / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin_C = (ltid * chunksize_C < work_C) ? (ltid * chunksize_C) : work_C;
  const libxsmm_blasint thr_end_C = ((ltid + 1) * chunksize_C < work_C) ? ((ltid + 1) * chunksize_C) : work_C;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  LIBXSMM_VLA_DECL(5, const float,         inp ,     pinp     + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5, const float,         inp_add,  pinp_add + (hi_start * ifwp + wi_start) * bc, CP, ifhp, ifwp, bc);      /* [N, CP, ifhp, ifwp, bc] + "padding" offset */
  LIBXSMM_VLA_DECL(5,       float,         out,      pout,      CP, ofhp, ofwp, bc);                                         /* [N, CP, ofhp, ofwp, bc] */
  LIBXSMM_VLA_DECL(5,       unsigned char, relumask, prelumask, CP, ofhp, ofwp, bc/BITS_PER_CHAR);                           /* [N, CP, ofhp, ofwp, bc/BITS_PER_CHAR] */

  LIBXSMM_VLA_DECL(2, const float,         gamma,    pgamma, bc);                   /* [CP, bc] */
  LIBXSMM_VLA_DECL(2, const float,         beta,     pbeta, bc);                    /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,         mean,     mean,  bc);                    /* [CP, bc] */
  LIBXSMM_VLA_DECL(2,       float,         var,      var,   bc);                    /* [CP, bc] */

  const float scale = 1.0f /((float)N * HW);

  LIBXSMM_VLA_DECL(3, float, sum_X_X2, ((float*)scratch), CP, bc);                  /* [2, CP, bc] */
  LIBXSMM_ASSUME_ALIGNED(sum_X_X2_, 64);
  const libxsmm_blasint sum_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + CP * 2 * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sum_N, ((float*)scratch) + sum_N_offset, N, bc);       /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(sum_N_, 64);
  const libxsmm_blasint sumsq_N_offset = (LIBXSMM_UP2((uintptr_t)(((float*)scratch) + sum_N_offset + CP * N * bc), 64) - ((uintptr_t)(scratch))) / sizeof(float);
  LIBXSMM_VLA_DECL(3, float, sumsq_N, ((float*)scratch) + sumsq_N_offset, N, bc);   /* [CP, N, bc] */
  LIBXSMM_ASSUME_ALIGNED(sumsq_N_, 64);

  libxsmm_meltw_unary_param  all_zero_param;
  libxsmm_meltw_binary_param add_param;
  libxsmm_meltw_unary_param  reduce_param;
  libxsmm_meltw_unary_param  all_relu_param;

  libxsmm_matrix_arg arg_array[6];

  libxsmm_matrix_eqn_param eqn_param;

  memset( &all_zero_param,  0, sizeof(all_zero_param));
  memset( &add_param,       0, sizeof(add_param));
  memset( &reduce_param,    0, sizeof(reduce_param));
  memset( &all_relu_param,  0, sizeof(all_relu_param));

  memset( &eqn_param,       0, sizeof(eqn_param));

  LIBXSMM_ALIGNED(float s[bc], 64);
  LIBXSMM_ALIGNED(float b[bc], 64);
  int n, cp;

  int cpxnt;
  if (norm_type == LIBXSMM_DNN_BN_FULL_NORM) {

    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      n  = cpxnt%N;
      cp = cpxnt/N;

      int hi, w, wb, hwb;

      float *sum_ncp_ptr   = &LIBXSMM_VLA_ACCESS(3, sum_N,   cp, n, 0, N, bc);
      float *sumsq_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, n, 0, N, bc);

      all_zero_param.out.primary = sum_ncp_ptr;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = sumsq_ncp_ptr;
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd  */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_ncp_ptr[cb] = 0.0f;    */
      /*   sumsq_ncp_ptr[cb] = 0.0f;  */
      /* } */

      LIBXSMM_ALIGNED(float lcl_sum_X_X2[2*bc], 64);
      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        reduce_param.out.primary = lcl_sum_X_X2;                                                    /* [2*bc]  */
        for (hi = 0; hi < H; hi++) {
          for (wb = 0; wb < num_W_blocks; wb++) {
            reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            cfg.reduce_kernel(&reduce_param);                                                       /* [HW, bc] -----> [2 * bc] */

            add_param.in0.primary = sum_ncp_ptr;
            add_param.in1.primary = lcl_sum_X_X2;
            add_param.out.primary = sum_ncp_ptr;
            cfg.helper_add_kernel(&add_param);

            add_param.in0.primary = sumsq_ncp_ptr;
            add_param.in1.primary = &lcl_sum_X_X2[bc];
            add_param.out.primary = sumsq_ncp_ptr;
            cfg.helper_add_kernel(&add_param);
          }
        }
      } else { /* hw-blocking (implies no padding) */
        reduce_param.out.primary = lcl_sum_X_X2;                                                   /* [2*bc]  */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          hi = (hwb*(HW/num_HW_blocks))/W;
          w  = (hwb*(HW/num_HW_blocks))%W;
          reduce_param.in.primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);
          cfg.reduce_kernel(&reduce_param);                                                       /* [HW, bc] -----> [2 * bc] */

          add_param.in0.primary = sum_ncp_ptr;
          add_param.in1.primary = lcl_sum_X_X2;
          add_param.out.primary = sum_ncp_ptr;
          cfg.helper_add_kernel(&add_param);

          add_param.in0.primary = sumsq_ncp_ptr;
          add_param.in1.primary = &lcl_sum_X_X2[bc];
          add_param.out.primary = sumsq_ncp_ptr;
          cfg.helper_add_kernel(&add_param);

          /* #pragma omp simd */
          /* for (int cb = 0; cb < bc; cb++) {  */
          /*   sum_ncp_ptr[cb] += lcl_sum_X_X2[cb];  */
          /*   sumsq_ncp_ptr[cb] += lcl_sum_X_X2[bc + cb];  */
          /* }  */
        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */
    } /* loop over cpxnt for temporary arrays */

    libxsmm_barrier_wait(cfg.barrier, ltid);

    for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {

      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) {  */
      /*   sum_X_X2[cp*bc + cb] = 0.0f;   */
      /*   sum_X_X2[CP*bc + (cp*bc + cb)] = 0.0f;  */
      /* } */

      int cb, ni;
      for(ni = 0; ni < N; ni++){

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sum_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, 0, CP, bc);
        cfg.helper_add_kernel(&add_param);

        add_param.in0.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, sumsq_N, cp, ni, 0, N, bc);
        add_param.out.primary = &LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, 0, CP, bc);
        cfg.helper_add_kernel(&add_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   sum_X_X2[cp*bc + cb] += sum_N[cp*N*bc + n*bc + cb]; */
        /*   sum_X_X2[CP*bc + (cp*bc + cb)] += sumsq_N[cp*N*bc + n*bc + cb]; */
        /* } */
      }

      for(cb = 0; cb < bc; cb++){
        mean[cp*bc + cb] = (LIBXSMM_VLA_ACCESS(3, sum_X_X2, 0, cp, cb, CP, bc)) * scale;                 /* E[X] */
        var[cp*bc + cb] = ((LIBXSMM_VLA_ACCESS(3, sum_X_X2, 1, cp, cb, CP, bc)) * scale) - (mean[cp*bc + cb]*mean[cp*bc + cb]);
      }
    } /* loop over cp for computing mean and var */

    libxsmm_barrier_wait(cfg.barrier, ltid);

  } /* mean and var computation are for the full norm only */

  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    n  = cpxnt%N;
    cp = cpxnt/N;

    int hi, ho, w, wb, hwb, cb;

    for(cb = 0; cb < bc; cb++){
      float lvar   = LIBXSMM_VLA_ACCESS(2, var,   cp, cb, bc);
      float lmean  = LIBXSMM_VLA_ACCESS(2, mean,  cp, cb, bc);

      s[cb] = 1.0f / ((float)sqrt(lvar + eps));                                 /* s = 1/sqrt(var(X) + eps)     [bc] */
      b[cb] = -1 * lmean * s[cb];                                               /* b = -E[X]/sqrt(var(X) + eps) [bc] */

      /* s[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps)); */                /* s = 1/sqrt(var(X) + eps)     [bc] */
      /* b[cb] = -1 * mean[cp*bc + cb] * s[cb];               */                /* b = -E[X]/sqrt(var(X) + eps) [bc] */
    }
    arg_array[1].primary = s;                                                   /* [bc] */
    arg_array[2].primary = b;                                                   /* [bc] */
    arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);     /* [bc] */
    arg_array[4].primary = (void*)&LIBXSMM_VLA_ACCESS(2, beta,  cp, 0, bc);     /* [bc] */

    if (cfg.use_hw_blocking == 0) { /* w-blocking */
      /* zeroing out strip [0, ho_start) x ofwp x bc */
      if (cfg.pad_h_out != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, 0, 0, 0, CP, ofhp, ofwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
      for (hi = 0, ho = ho_start; hi < H; hi++, ho++) {
        /* zeroing out starting [0, wo_start) x bc and [wo_end, ofwp] x bc blocks for fixed ho */
        if (cfg.pad_w_out != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, 0, 0, CP, ofhp, ofwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
        for (wb = 0; wb < num_W_blocks; wb++) {
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);             /* [bw, bc] */
          eqn_param.inputs = arg_array;
          eqn_param.output.primary   = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, wo_start + wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);   /* [bw, bc] */

          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, hi, wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);       /* [bw, bc] */
          }

          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                            (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wo_start + wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, (bc/BITS_PER_CHAR)) : NULL );
          }
          cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
        }
        /* zeroing out ending [wo_end, ofwp] x bc block for fixed ho */
        if (cfg.pad_w_out != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho, wo_end, 0, CP, ofhp, ofwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
      }
      /* zeroing out strip [ho_end, ofhp) x ofwp x bc */
      if (cfg.pad_h_out != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, out, n, cp, ho_end, 0, 0, CP, ofhp, ofwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }

    } else { /* hw-blocking (implies no padding) */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        hi = (hwb*(HW/num_HW_blocks))/W;
        ho = hi;
        w  = (hwb*(HW/num_HW_blocks))%W;
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp, n, cp, hi, w, 0, CP, H, W, bc);          /* [HW, bc] */
        eqn_param.inputs = arg_array;
        eqn_param.output.primary   = &LIBXSMM_VLA_ACCESS(5, out, n, cp, hi, w, 0, CP, H, W, bc);           /* [HW,bc] */

        if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU ||  cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
          arg_array[5].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp_add, n, cp, ho, w, 0, CP, H, W, bc);    /* [HW, bc] */
        }

        if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
          eqn_param.output.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                          (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, (bc/BITS_PER_CHAR)) : NULL );
        }
        cfg.func10(&eqn_param);                                                   /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */
      }
    } /* if-else for the presence of padding */
  } /* loop over cpxnt for computing din */

  libxsmm_barrier_wait(cfg.barrier, ltid);
}

#endif

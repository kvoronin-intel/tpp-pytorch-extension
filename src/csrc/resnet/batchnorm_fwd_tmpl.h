RECORD_FUNCTION("batchnorm_fwd", std::vector<c10::IValue>());

/*        ( input, input_add, weight, bias, mean, var ) = inputs */
#define TIMING

#ifdef WITH_VTUNE
  #define USE_VTUNE
#endif

#define NUM_ITER_PERF_DEBUG 1

//#define MAKE_INPUT_HOT

//#define MEMCPY

// Debugging macro which initializes coeffs and inp for scaling part with simple constants
//#define SIMPLE_COEFFS

#ifdef TIMING
  double t_start = 0.0, t_end = 0.0, t_bn_start = 0.0, t_bn_reduce_start = 0.0, t_bn_scale_start = 0.0;
#endif

#ifdef TIMING
t_start = getTime();
#endif

//#define VERBOSE

#ifdef MAKE_INPUT_HOT
  #ifdef VERBOSE
  printf("MAKE_INPUT_HOT is active!\n");
  #endif
  #ifdef TIMING
    double t_extra_tweak_start = 0.0, t_extra_tweak = 0.0;
  #endif
#endif

#ifdef USE_VTUNE
  __itt_domain* bn_domain = __itt_domain_create("bn_domain");
  bn_domain->flags = 1;
  #define ITT_DOMAIN bn_domain
#endif

int zero_output_upfront = 0;
/*
const char* const env_prec_str = getenv("ZERO_OUTPUT_UPFRONT");
if (0 == env_prec_str) {
  zero_output_upfront = 0;
} else {
  zero_output_upfront = atoi(env_prec_str);
}
*/

int bcast_upfront = 0;
/*
const char* const env_prec_str2 = getenv("BCAST_UPFRONT");
if (0 == env_prec_str2) {
  bcast_upfront = 0;
} else {
  bcast_upfront = atoi(env_prec_str2);
}
*/

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

int num_HW_blocks = 1;
//#if 0
if (H * W * bc * sizeof(T) <= 16384)
  num_HW_blocks = 1;
else if ((H%2 == 0 || W%2 == 0) && (H * W * bc * sizeof(T) / 2 <= 16384))
  num_HW_blocks = 2;
else if ((H%2 == 0 && W%2 == 0) && (H * W * bc * sizeof(T) / 4 <= 16384))
  num_HW_blocks = 4;
else if (H > W)
  num_HW_blocks = H;
else
  num_HW_blocks = W;
//#endif

//const int num_HW_blocks = (H > W ? H : W);
const int num_W_blocks  = (W % 64 == 0 ? W / 64 : 1);

int spatial_block_size = 0;

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
std::cout << "training = " << training << std::endl;
std::cout << "relu = " << relu << std::endl;
std::cout << "eltwise = " << eltwise << std::endl;
std::cout << "output_size = " << output_size << std::endl;
std::cout << "t_I sizes = " << t_I.sizes() << std::endl;
std::cout << "use_hw_blocking = " << use_hw_blocking << std::endl;
std::cout << "num_HW_blocks = " << num_HW_blocks << " num_W_blocks = " << num_W_blocks<< std::endl;
std::cout << "spatial_block_size = " << spatial_block_size << std::endl;

if (zero_output_upfront)
std::cout << "ZERO_OUTPUT_UPFRONT is enabled (with full H-W-bc processing without any blocking)" << std::endl;
else
std::cout << "ZERO_OUTPUT_UPFRONT is not enabled (with full H-W-bc processing without any blocking)" << std::endl;

if (bcast_upfront)
std::cout << "BCAST_UPFRONT is enabled (with full H-W-bc processing without any blocking)" << std::endl;
else
std::cout << "BCAST_UPFRONT is not enabled (with full H-W-bc processing without any blocking)" << std::endl;
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

  
  SCOPEIT_DECL(SetZeroTPP<T>)               zero_upfront_tpp;
//  SCOPEIT_DECL(UnaryTPP)                    bcast_upfront_tpp;
  SCOPEIT_DECL(CpyBiasTPP<float,float>)     bcast_upfront_tpp;
  SCOPEIT_DECL(BatchNormFwdScaleTPP<T,T>)   normalize_tpp;

if (zero_output_upfront)
  zero_upfront_tpp = SCOPEIT(SetZeroTPP<T>(ofhp*ofwp*bc), EW_ZERO);


#ifdef VERBOSE
  printf("JITTING normalize_tpp\n");
#endif

if (zero_output_upfront) {
if (bcast_upfront) {
  //bcast_upfront_tpp = SCOPEIT(UnaryTPP<T>(bc, W, bc, bc*W, ofhp*ofwp*bc, dt,dt,dt, XsmmDtype<T>, XsmmDtype<T>, XsmmDtype<T>, LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, EW_ZERO);
  //printf("bcast_upfront arguments\n");
  bcast_upfront_tpp = SCOPEIT((CpyBiasTPP<float,float>(W, bc)), EW_ZERO);
  //printf("constructor arguments: %d %d %d %d %d %d\n", bc * W, H, bc * ifwp, bc * ofwp, relu, eltwise);
  normalize_tpp = SCOPEIT((BatchNormFwdScaleTPP<T,T>(bc * W, H, bc * ifwp, bc * ofwp, relu, eltwise, 1)), NORMALIZE);
} else {
  //printf("normalize_tpp constructor call needs to be changed to work for ZERO_OUTPUT_UPFRONT\n");
  //exit(-1);
  //printf("constructor arguments: %d %d %d %d %d %d\n", bc * W, H, bc * ifwp, bc * ofwp, relu, eltwise);
  normalize_tpp = SCOPEIT((BatchNormFwdScaleTPP<T,T>(bc * W, H, bc * ifwp, bc * ofwp, relu, eltwise)), NORMALIZE);
}
} else {
  normalize_tpp = SCOPEIT((BatchNormFwdScaleTPP<T,T>(bc, spatial_block_size, relu, eltwise)), NORMALIZE);
}

#ifdef VERBOSE
  printf("JITTING dbg_copy_tpp\n");
#endif
  auto dbg_copy_tpp = SCOPEIT((CpyTPP<T>(1, bc*spatial_block_size)), EW_COPY);

#ifdef VERBOSE
  printf("JITTING dbg_copy_hwbc_tpp\n");
#endif
  auto dbg_copy_hwbc_tpp = SCOPEIT((CpyTPP<T>(bc*H*W)), EW_COPY);

#ifdef VERBOSE
  printf("JITTING dbg_copy_full_tpp\n");
#endif
  auto dbg_copy_full_tpp = SCOPEIT((CpyTPP<T>(bc*CP*H*W)), EW_COPY);

#ifdef VERBOSE
  printf("bc * spatial_block_size (block for batchnorm scale) = %d (bytes) = %3.3f (Kb) = %3.3f (Mb)\n", bc * spatial_block_size * sizeof(T), bc * spatial_block_size * sizeof(T) / 1024.0, bc * spatial_block_size * sizeof(T) / 1024.0 / 1024.0);
#endif

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

#ifdef TIMING
  t_bn_start = getTime();
#endif

  if (training) {
    for (int i = 0; i < NUM_ITER_PERF_DEBUG; i++)
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


#ifdef TIMING
    t_bn_reduce_start = getTime();
#endif

    for (int i = 0; i < NUM_ITER_PERF_DEBUG; i++)
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

#ifdef USE_VTUNE
    if (CP*bc == 256 && ifhp == 56 && ifwp == 56) {
      //__itt_resume();
      //printf("Called resume\n");
      __itt_frame_begin_v3(ITT_DOMAIN, NULL);
      printf("Called frame_begin\n");
    } else {
      //printf("Cp*bc = %d ifhwp = %d ifwp = %d \n", CP*bc, ifhp, ifwp);
      printf("Did not call frame_begin\n");
    }
#endif

#ifdef TIMING
    t_bn_scale_start = getTime();
#endif

  for (int i = 0; i < NUM_ITER_PERF_DEBUG; i++)
  {
#ifdef MAKE_INPUT_HOT
#ifdef TIMING
    t_extra_tweak_start = getTime();
#endif

      DECL_VLA_PTR_PT_EXT(T,             inp,      [CP][ifhp][ifwp][bc], t_I, (hi_start * ifwp + wi_start) * bc);
      volatile int resint = 0.0;
#pragma omp parallel
{
      #pragma omp for reduction(+: resint)
      for (int n = 0; n < N; n++) {
        for (int cp = 0; cp < CP; cp++) {
          for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
              for (int k = 0; k < bc; k++) {
                T res = inp[n][cp][h][w][k];
                resint += (int)res;
              }
        }
      }
}

#ifdef TIMING
    t_extra_tweak += getTime() - t_extra_tweak_start;
#endif
#endif /* MAKE_INPUT_HOT */

    RECORD_SCOPE(bn_fwd_scale, {});
    {
#ifdef MEMCPY
      DECL_VLA_PTR_PT_EXT(T,             inp,      [CP][ifhp][ifwp][bc], t_I, (hi_start * ifwp + wi_start) * bc);
      DECL_VLA_PTR_PT    (T,             out,      [CP][ofhp][ofwp][bc], t_O);
      DECL_VLA_PTR_PT_EXT(T,             out_shifted,      [CP][ofhp][ofwp][bc], t_O, (ho_start * ofwp + wo_start) * bc);
#pragma omp parallel for
      for (int n = 0; n < N; n++) {
        //memcpy(out[n][0][0][0], inp[n][0][0][0], CP*H*W*bc*sizeof(T));
        //dbg_copy_full_tpp(inp[n][0][0][0], out[n][0][0][0]);
        for (int cp = 0; cp < CP; cp++) {
          dbg_copy_hwbc_tpp(inp[n][cp][0][0], out_shifted[n][cp][0][0]);
          //memcpy(out_shifted[n][cp][0][0], inp[n][cp][0][0], H*W*bc*sizeof(T));
        }
      }
#else /* MEMCPY */
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
//#ifdef ZERO_OUTPUT_UPFRONT
          DECL_VLA_PTR_PT_EXT(T,             out_shifted,      [CP][ofhp][ofwp][bc], t_O, (ho_start * ofwp + wo_start) * bc);
          DECL_VLA_PTR_PT_EXT(unsigned char, relumask_shifted, [CP][ofhp][ofwp][bc/BITS_PER_CHAR], t_relu_mask, (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR);
//#endif

//#if 0
          LIBXSMM_ALIGNED(float s[bc], 64);
          LIBXSMM_ALIGNED(float b[bc], 64);

          LIBXSMM_ALIGNED(float s_bcast[bc*W], 64);
          LIBXSMM_ALIGNED(float b_bcast[bc*W], 64);
          LIBXSMM_ALIGNED(float gamma_bcast[bc*W], 64);
          LIBXSMM_ALIGNED(float beta_bcast[bc*W], 64);

          coeffs_tpp(mean[cp], var[cp], &s[0], &b[0]);

#ifdef SIMPLE_COEFFS
          for (int k = 0; k < bc; k++) {
            s[k]     = k + 1;
            b[k]     = 0.0f;
            gamma[cp][k] = 1.0f;
            beta[cp][k]  = 0.0f;
          }
          for (int h = 0; h < ifhp; h++) {
            for (int w = 0; w < ifwp; w++) {
              for (int k = 0; k < bc; k++) {
                inp[n][cp][h][w][k] = 1.0;
              }
            }
          }
          // while debugging
#endif

//#ifdef BCAST_UPFRONT
if (bcast_upfront) {
          bcast_upfront_tpp(&s[0], &s_bcast[0]);
          bcast_upfront_tpp(&b[0], &b_bcast[0]);
          bcast_upfront_tpp(gamma[cp], &gamma_bcast[0]);
          bcast_upfront_tpp(beta[cp], &beta_bcast[0]);
#if 0
          for (int l = 0; l < W; l++) {
            for (int k = 0; k < bc; k++) {
              s_bcast[l*bc + k] = s[k];
              b_bcast[l*bc + k] = b[k];
              gamma_bcast[l*bc + k] = gamma[cp][k];
              beta_bcast [l*bc + k] = beta [cp][k];
            }
          }
#endif // for #if 0

}

//#endif // for #if 0

//#if 0

if (zero_output_upfront) {
//#ifdef ZERO_OUTPUT_UPFRONT
          zero_upfront_tpp(out[n][cp][0][0]);
//#ifdef BCAST_UPFRONT
if (bcast_upfront) {
          normalize_tpp(inp[n][cp][0][0], &s_bcast[0], &b_bcast[0], &gamma_bcast[0], &beta_bcast[0],
                          eltwise ? inp_add[n][cp][0][0] : NULL,
                          out_shifted[n][cp][0][0],
                          relu ? relumask_shifted[n][cp][0][0] : NULL);
} else { // #else
          normalize_tpp(inp[n][cp][0][0], &s[0], &b[0], gamma[cp], beta[cp],
                          eltwise ? inp_add[n][cp][0][0] : NULL,
                          out_shifted[n][cp][0][0],
                          relu ? relumask_shifted[n][cp][0][0] : NULL);
} //#endif

} else { //#else

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

//          dbg_copy_hwbc_tpp(inp[n][cp][0][0], out[n][cp][0][0]);
/*
              dbg_copy_tpp(inp[n][cp][0][0], out[n][cp][0][0]);
              dbg_copy_tpp(inp[n][cp][1][0], out[n][cp][1][0]);
              dbg_copy_tpp(inp[n][cp][2][0], out[n][cp][2][0]);
              dbg_copy_tpp(inp[n][cp][3][0], out[n][cp][3][0]);
              dbg_copy_tpp(inp[n][cp][4][0], out[n][cp][4][0]);
              dbg_copy_tpp(inp[n][cp][5][0], out[n][cp][5][0]);
              dbg_copy_tpp(inp[n][cp][6][0], out[n][cp][6][0]);
              dbg_copy_tpp(inp[n][cp][7][0], out[n][cp][7][0]);
              dbg_copy_tpp(inp[n][cp][8][0], out[n][cp][8][0]);
              dbg_copy_tpp(inp[n][cp][9][0], out[n][cp][9][0]);
              dbg_copy_tpp(inp[n][cp][10][0], out[n][cp][10][0]);
              dbg_copy_tpp(inp[n][cp][11][0], out[n][cp][11][0]);
              dbg_copy_tpp(inp[n][cp][12][0], out[n][cp][12][0]);
              dbg_copy_tpp(inp[n][cp][13][0], out[n][cp][13][0]);
              dbg_copy_tpp(inp[n][cp][14][0], out[n][cp][14][0]);
              dbg_copy_tpp(inp[n][cp][15][0], out[n][cp][15][0]);
*/
//#if 0
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
//                int hi = hwb, ho = hwb, w = 0;
              int hi = (hwb*(H*W/num_HW_blocks))/W;
              int ho = hi;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              /* Normalization equation + relu + eltwise -> y = relu( ((s*x + b)*gamma + beta) + inp_add) */

//              for (int k = 0; k < spatial_block_size*bc; k++)
//                  out[n][cp][ho][w][k] = inp[n][cp][hi][w][k];

//              dbg_copy_tpp(inp[n][cp][hi][w], out[n][cp][ho][w]);


              normalize_tpp(inp[n][cp][hi][w], &s[0], &b[0], gamma[cp], beta[cp],
                              eltwise ? inp_add[n][cp][hi][w] : NULL,
                              out[n][cp][ho][w],
                              relu ? relumask[n][cp][ho][w] : NULL);
            }
//#endif
          } /* if-else for the presence of padding */
}
//#endif /* for ZERO_OUTPUT_UPFRONT */

//#endif // for #if 0
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
#endif /* MEMCPY */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_fwd_scale scope */

} /* end of the dummy scope */

#ifdef TIMING
  t_end = getTime();
#endif

#ifdef USE_VTUNE
    if (CP*bc == 256 && ifhp == 56 && ifwp == 56)
    {
      __itt_frame_end_v3(ITT_DOMAIN, NULL);
      ITT_DOMAIN->flags = 0;
      //__itt_pause();
    }
#endif

#ifdef TIMING
  auto buf = tuning_timings.request();
  float* ptr = (float*)buf.ptr;
  ptr[0] += t_end - t_bn_start;
#ifdef MAKE_INPUT_HOT
  ptr[0] -= t_extra_tweak;
#endif
  ptr[1] += t_end - t_start;
#ifdef MAKE_INPUT_HOT
  ptr[1] -= t_extra_tweak;
#endif
//  ptr[2] += t_bn_reduce_start - t_bn_start;
//  ptr[3] += t_bn_scale_start  - t_bn_reduce_start;
  ptr[4] += t_end  - t_bn_scale_start;
#ifdef MAKE_INPUT_HOT
  ptr[4] -= t_extra_tweak;
#endif

#ifdef MAKE_INPUT_HOT
printf("t_extra_tweak per one of 1000 iter = %6.6f \n", t_extra_tweak / NUM_ITER_PERF_DEBUG);
printf("t scale = %6.6f \n", t_end  - t_bn_scale_start);
#endif

#endif

#ifdef VERBOSE
  #undef VERBOSE
#endif

#ifdef TIMING
  #undef TIMING
#endif

#ifdef USE_VTUNE
  ITT_DOMAIN->flags = 0;
  #undef ITT_DOMAIN
  #undef USE_VTUNE
#endif

//return std::vector<at::Tensor>({t_O, t_relu_mask, inputs[6]});
return std::vector<at::Tensor>({t_O, t_relu_mask, t_scratch});

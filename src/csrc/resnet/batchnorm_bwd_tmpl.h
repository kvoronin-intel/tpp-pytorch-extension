RECORD_FUNCTION("batchnorm_bwd", std::vector<c10::IValue>());

// inputs ~ grad_outs
//   ctx.save_for_backward(input, input_add, weight, mean, var, invstd, relu_mask, relu, eltwise, paddings, output)
//   inputs += ctx.saved_tensors

#define VERBOSE

auto t_GO = inputs[0]; /* grad_output */
auto t_I  = inputs[1]; /* input */
auto t_IA = inputs[2]; /* input_add */
auto t_W  = inputs[3]; /* weight */
auto t_M  = inputs[4]; /* mean */
auto t_V  = inputs[5]; /* var  */
auto t_R  = inputs[6]; /* relumask */
auto t_scratch  = inputs[7]; /* pre-allocated scratch */

auto t_grad_input     = at::empty(t_I.sizes(),  torch::TensorOptions().dtype(t_I.dtype()));
auto t_grad_input_add = at::empty(t_IA.sizes(), torch::TensorOptions().dtype(t_I.dtype()));

auto t_grad_weight    = at::empty(t_W.sizes(),  torch::TensorOptions().dtype(t_W.dtype()));
auto t_grad_bias      = at::empty(t_W.sizes(),  torch::TensorOptions().dtype(t_W.dtype()));

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

const long hi_start      = pad_h_in;
const long hi_end        = hi_start + H;
const long wi_start      = pad_w_in;
const long wi_end        = W + pad_w_in;
const long ifhp = H + 2 * pad_h_in;
const long ifwp = W + 2 * pad_w_in;

const long ho_start      = pad_h_out;
const long wo_start      = pad_w_out;
const long ofhp = H + 2 * pad_h_out;
const long ofwp = W + 2 * pad_w_out;

const float scale = 1.0f /((float)N * H * W);

const long sum_N_offset          = LIBXSMM_UP2(CP * 2 * bc, 64);
const long sumsq_N_offset        = LIBXSMM_UP2(sum_N_offset + CP * N * bc, 64);

const long dbeta_N_offset        = LIBXSMM_UP2(CP * N * bc, 64);

/*
const long full_fwd_scratch_size = sumsq_N_offset + LIBXSMM_UP2((size_t)CP * (size_t)N * (size_t)bc, 64);

const long full_bwd_scratch_size = dbeta_N_offset + LIBXSMM_UP2(CP * N * bc, 64);

const long full_scratch_size     = std::max(full_fwd_scratch_size, full_bwd_scratch_size);

// FIXME: Save scratch somewhere to not allocate each time
std::vector<long> scratch_size{full_scratch_size};

auto scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kFloat));
*/

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
  std::cout << "t_I sizes = " << t_I.sizes() << std::endl;
  std::cout << "use_hw_blocking spatial_block_size = " << use_hw_blocking << " " << spatial_block_size << std::endl;
  std::cout << "relu eltwise = " << relu << " " << eltwise << std::endl;
#endif

{
#ifndef THREADED_LOOPS
  DECL_VLA_PTR_PT    (T,             inp,      [CP][ifhp][ifwp][bc], t_I);
  DECL_VLA_PTR_PT    (float,         gamma,    [bc],                 t_W);
  DECL_VLA_PTR_PT    (float,         mean,     [bc],                 t_M);
  DECL_VLA_PTR_PT    (float,         var,      [bc],                 t_V);
  DECL_VLA_PTR_PT    (float,         dgamma_N, [N][bc],              t_scratch);
  DECL_VLA_PTR_PT_EXT(float,         dbeta_N,  [N][bc],              t_scratch, dbeta_N_offset);
  DECL_VLA_PTR_PT    (T,             din,      [CP][ifhp][ifwp][bc], t_grad_input);
  DECL_VLA_PTR_PT    (T,             din_add,  [CP][ifhp][ifwp][bc], t_grad_input_add);
  DECL_VLA_PTR_PT    (float,         dgamma,   [bc],                 t_grad_weight);
  DECL_VLA_PTR_PT    (float,         dbeta,    [bc],                 t_grad_bias);

  DECL_VLA_PTR_PT_EXT(T,             dout,     [CP][ofhp][ofwp][bc],               t_GO, (ho_start * ofwp + wo_start) * bc);
  DECL_VLA_PTR_PT_EXT(unsigned char, relumask, [CP][ofhp][ofwp][bc/BITS_PER_CHAR], t_R,  (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR);

#endif

  auto zero_tpp = SCOPEIT(SetZeroTPP<float>(bc), EW_ZERO);

  auto zero_hp_tpp = SCOPEIT(SetZeroTPP<T>((pad_h_in * ifwp), bc, bc), EW_ZERO); /* (pad_h_in * ifwp), bc because of row-major for unary */

  auto zero_wp_tpp = SCOPEIT(SetZeroTPP<T>(pad_w_in, bc, bc), EW_ZERO);          /* pad_w_in, bc because of row-major for unary */

  auto coeffs_tpp = SCOPEIT(BatchNormStatCoeffsTPP<float>(bc, eps), NORMALIZE);

  auto helper_copy_tpp = SCOPEIT((CpyTPP<float>(1, bc, bc, bc)), EW_COPY); /* 1, bc because of row-major for unary */

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bc, bc, bc), EW_ADD); /* 1, bc because of row-major for unary */

  auto grad_w_inpadd_tpp = SCOPEIT((BatchNormBwdWTPP<T,T>(bc, spatial_block_size, relu, eltwise)), NORMALIZE);

  auto abc_coeffs_tpp = SCOPEIT(BatchNormABCCoeffsTPP<float>(bc, scale, eps), NORMALIZE);

  auto grad_d_tpp = SCOPEIT((BatchNormBwdDTPP<T,T>(bc, spatial_block_size)), NORMALIZE);

#ifdef THREADED_LOOPS
  char ncp_loop_specs_str[256] = "AB";
  const long n_step = 1, cp_step = 1;
  auto ncp_loop = ThreadedLoop<2>({
      LoopSpecs{0, N,  n_step,  {/*l1_k_step, l0_k_step*/}},   // Logical N  loop specs
      LoopSpecs{0, CP, cp_step, {/*l1_n_step, l0_n_step*/}}},  // Logical CP loop specs
      ncp_loop_specs_str);

  char cp_loop_specs_str[256] = "A";
  auto cp_loop = ThreadedLoop<1>({
      LoopSpecs{0, CP, cp_step, {/*l1_k_step, l0_k_step*/}}},  // Logical CP loop specs
      cp_loop_specs_str);
#endif

/*
  T* inp_dbg = t_I.data_ptr<T>();
  for (int i = 0; i < 20; i++) {
    if (sizeof(T) == 2) {
      float tmp = 0.0;
      libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&(inp_dbg[i])), &tmp, 1);
      printf("for bn3 inp_dbg[%d] = %u = %f\n", i, *(unsigned short*)(&inp_dbg[i]), tmp);
    } else
      printf("for bn3 inp_dbg[%d] = %f\n", i, inp_dbg[i]);
  }
*/
  {
    RECORD_SCOPE(bn_bwd_w_inpadd, {});
    {
#ifdef THREADED_LOOPS
      ncp_loop(
        [&](int *ind) {
          const int n = ind[0], cp = ind[1];

          DECL_VLA_PTR_PT    (T,             inp,      [CP][ifhp][ifwp][bc], t_I);
          DECL_VLA_PTR_PT    (float,         gamma,    [bc],                 t_W);
          DECL_VLA_PTR_PT    (float,         mean,     [bc],                 t_M);
          DECL_VLA_PTR_PT    (float,         var,      [bc],                 t_V);
          DECL_VLA_PTR_PT_EXT(T,             dout,     [CP][ofhp][ofwp][bc],               t_GO, (ho_start * ofwp + wo_start) * bc);
          DECL_VLA_PTR_PT_EXT(unsigned char, relumask, [CP][ofhp][ofwp][bc/BITS_PER_CHAR], t_R,  (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR);
          DECL_VLA_PTR_PT    (float,         dgamma_N, [N][bc],              t_scratch);
          DECL_VLA_PTR_PT_EXT(float,         dbeta_N,  [N][bc],              t_scratch, dbeta_N_offset);
          DECL_VLA_PTR_PT    (T,             din_add,  [CP][ifhp][ifwp][bc], t_grad_input_add);

          LIBXSMM_ALIGNED(float lcl_dgamma_ptr[bc], 64);
          LIBXSMM_ALIGNED(float lcl_dbeta_ptr[bc], 64);

          LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
          LIBXSMM_ALIGNED(float b[bc], 64);

          zero_tpp(&lcl_dgamma_ptr[0]);
          zero_tpp(&lcl_dbeta_ptr [0]);

          coeffs_tpp(mean[cp], var[cp], &a[0], &b[0]);
          /*
          if (n == 0 && cp == 0) {
            for (int i = 0; i < 10; i++)
              printf("mean[... + %d] = %f var[...+%d] = %f a[%d] = %f b[%d] = %f \n", i, mean[cp][i], i, var[cp][i], i, a[i], i, b[i]);
          }
          */

          if (!use_hw_blocking) {

            if (pad_h_in != 0 && eltwise ) {
              zero_hp_tpp(din_add[n][cp][0][0]);
            }
            for (int ho = 0, hi = hi_start; ho < H; ho++, hi++) {
              /* zeroing out starting [0, wi_start) x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                zero_wp_tpp(din_add[n][cp][hi][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                grad_w_inpadd_tpp(inp[n][cp][hi][wi_start + wb*(W/num_W_blocks)], &a[0], &b[0], dout[n][cp][ho][wb*(W/num_W_blocks)], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][wi_start + wb*(W/num_W_blocks)] : NULL,
                                relu ? relumask[n][cp][ho][wb*(W/num_W_blocks)] : NULL);
              }
              /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                zero_wp_tpp(din_add[n][cp][hi][wi_end]);
              }
            } /* loop over ho */
            /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
            if (pad_h_in != 0 && eltwise ) {
              zero_hp_tpp(din_add[n][cp][hi_end][0]);
            }
          } else { /* for if (!use_hw_blocking) */
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int ho = (hwb*(H*W/num_HW_blocks))/W;
              int hi = ho;
              int w  = (hwb*(H*W/num_HW_blocks))%W;
/*
                if (n == 0 && cp == 0 && hwb == 0) {
                  for (int i = 0; i < 10; i++)
                    printf("inpmean[... + %d] = %f var[...+%d] = %f a[%d] = %f b[%d] = %f \n", i, mean[cp][i], i, var[cp][i], i, a[i], i, b[i]);
                }
*/
              grad_w_inpadd_tpp(inp[n][cp][hi][w], &a[0], &b[0], dout[n][cp][ho][w], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][w] : NULL,
                                relu ? relumask[n][cp][ho][w] : NULL);
            }
          } /* if-else for the presence of input padding */

          helper_copy_tpp(&lcl_dgamma_ptr[0], dgamma_N[cp][n]);
          helper_copy_tpp(&lcl_dbeta_ptr[0],  dbeta_N [cp][n]);
        },
        [&]() {},
        [&]() {});
#else /* THREADED_LOOPS */
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < N; n++) {
        for (int cp = 0; cp < CP; cp++) {

          LIBXSMM_ALIGNED(float lcl_dgamma_ptr[bc], 64);
          LIBXSMM_ALIGNED(float lcl_dbeta_ptr[bc], 64);

          LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
          LIBXSMM_ALIGNED(float b[bc], 64);

          zero_tpp(&lcl_dgamma_ptr[0]);
          zero_tpp(&lcl_dbeta_ptr [0]);

          coeffs_tpp(mean[cp], var[cp], &a[0], &b[0]);

          if (!use_hw_blocking) {

            if (pad_h_in != 0 && eltwise ) {
              zero_hp_tpp(din_add[n][cp][0][0]);
            }
            for (int ho = 0, hi = hi_start; ho < H; ho++, hi++) {
              /* zeroing out starting [0, wi_start) x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                zero_wp_tpp(din_add[n][cp][hi][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                grad_w_inpadd_tpp(inp[n][cp][hi][wi_start + wb*(W/num_W_blocks)], &a[0], &b[0], dout[n][cp][ho][wb*(W/num_W_blocks)], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][wi_start + wb*(W/num_W_blocks)] : NULL,
                                relu ? relumask[n][cp][ho][wb*(W/num_W_blocks)] : NULL);
              }
              /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                zero_wp_tpp(din_add[n][cp][hi][wi_end]);
              }
            } /* loop over ho */
            /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
            if (pad_h_in != 0 && eltwise ) {
              zero_hp_tpp(din_add[n][cp][hi_end][0]);
            }
          } else {
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int ho = (hwb*(H*W/num_HW_blocks))/W;
              int hi = ho;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              grad_w_inpadd_tpp(inp[n][cp][hi][w], &a[0], &b[0], dout[n][cp][ho][w], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][w] : NULL,
                                relu ? relumask[n][cp][ho][w] : NULL);
            }
          } /* if-else for the presence of input padding */

          helper_copy_tpp(&lcl_dgamma_ptr[0], dgamma_N[cp][n]);
          helper_copy_tpp(&lcl_dbeta_ptr[0],  dbeta_N [cp][n]);

        } /* end of cp loop */
      } /* end of n loop */
#endif /* THREADED_LOOPS */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_w_inpadd scope */

  {
    RECORD_SCOPE(bn_bwd_w_add, {});
    {
#ifdef THREADED_LOOPS
      cp_loop(
        [&](int *ind) {
          const int cp = ind[0];

          DECL_VLA_PTR_PT    (float, dgamma,   [bc],    t_grad_weight);
          DECL_VLA_PTR_PT    (float, dbeta,    [bc],    t_grad_bias);
          DECL_VLA_PTR_PT    (float, dgamma_N, [N][bc], t_scratch);
          DECL_VLA_PTR_PT_EXT(float, dbeta_N,  [N][bc], t_scratch, dbeta_N_offset);

          zero_tpp(dgamma[cp]);
          zero_tpp(dbeta [cp]);

          for(int ni = 0; ni < N; ni++) {
            //for (int i = 0; i < 10; i++)
            //  printf("cp = %d dgamma_N[i] = %f \n", cp, (float)dgamma_N[cp][ni][bc]);
            helper_add_tpp(dgamma[cp], dgamma_N[cp][ni], dgamma[cp]);
            helper_add_tpp(dbeta [cp], dbeta_N [cp][ni], dbeta [cp]);
          } /* end of ni loop */
        },
        [&]() {},
        [&]() {});
#else /* THREADED_LOOPS */
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int cp = 0; cp < CP; cp++) {
        zero_tpp(dgamma[cp]);
        zero_tpp(dbeta [cp]);

        for(int ni = 0; ni < N; ni++) {
          helper_add_tpp(dgamma[cp], dgamma_N[cp][ni], dgamma[cp]);
          helper_add_tpp(dbeta [cp], dbeta_N [cp][ni], dbeta [cp]);
        } /* end of ni loop */
      } /* end of cp loop */
#endif /* THREADED_LOOPS */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_w_add scope */

  {
    RECORD_SCOPE(bn_bwd_d, {});
    {

#ifdef THREADED_LOOPS
      ncp_loop(
        [&](int *ind) {
          const int n = ind[0], cp = ind[1];

          DECL_VLA_PTR_PT    (T,             inp,      [CP][ifhp][ifwp][bc], t_I);
          DECL_VLA_PTR_PT    (T,             din,      [CP][ifhp][ifwp][bc], t_grad_input);
          DECL_VLA_PTR_PT    (float,         gamma,    [bc],                 t_W);
          DECL_VLA_PTR_PT    (float,         mean,     [bc],                 t_M);
          DECL_VLA_PTR_PT    (float,         var,      [bc],                 t_V);
          DECL_VLA_PTR_PT_EXT(T,             dout,     [CP][ofhp][ofwp][bc], t_GO, (ho_start * ofwp + wo_start) * bc);
          DECL_VLA_PTR_PT    (float,         dgamma,   [bc],                 t_grad_weight);
          DECL_VLA_PTR_PT    (float,         dbeta,    [bc],                 t_grad_bias);

          LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
          LIBXSMM_ALIGNED(float b[bc], 64);
          LIBXSMM_ALIGNED(float c[bc], 64);

          abc_coeffs_tpp(gamma[cp], dgamma[cp], var[cp], mean[cp], dbeta[cp], &a[0], &b[0], &c[0]);

          if (!use_hw_blocking) {
            if (pad_h_in != 0) {
              zero_hp_tpp(din[n][cp][0][0]);
            }
            for (int ho = 0, hi = hi_start; ho < H; ho++, hi++) {
              /* zeroing out starting [0, wi_start) x bc block for fixed hi */
              if (pad_w_in != 0) {
                zero_wp_tpp(din[n][cp][hi][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                grad_d_tpp(inp[n][cp][hi][wi_start + wb*(W/num_W_blocks)], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][wb*(W/num_W_blocks)], din[n][cp][hi][wi_start + wb*(W/num_W_blocks)]);
              }
              /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
              if (pad_w_in != 0) {
                zero_wp_tpp(din[n][cp][hi][wi_end]);
              }

            }
            /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
            if (pad_h_in != 0) {
              zero_hp_tpp(din[n][cp][hi_end][0]);
            }
          } else { /* for if (!use_hw_blocking) */
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int ho = (hwb*(H*W/num_HW_blocks))/W;
              int hi = ho;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              grad_d_tpp(inp[n][cp][hi][w], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][w], din[n][cp][hi][w]);
            }
          } /* if-else for the presence of input padding */
        },
        [&]() {},
        [&]() {});
#else /* THREADED_LOOPS */
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < N; n++) {
        for (int cp = 0; cp < CP; cp++) {

          LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
          LIBXSMM_ALIGNED(float b[bc], 64);
          LIBXSMM_ALIGNED(float c[bc], 64);

          abc_coeffs_tpp(gamma[cp], dgamma[cp], var[cp], mean[cp], dbeta[cp], &a[0], &b[0], &c[0]);

          if (!use_hw_blocking) {
            if (pad_h_in != 0) {
              zero_hp_tpp(din[n][cp][0][0]);
            }
            for (int ho = 0, hi = hi_start; ho < H; ho++, hi++) {
              /* zeroing out starting [0, wi_start) x bc block for fixed hi */
              if (pad_w_in != 0) {
                zero_wp_tpp(din[n][cp][hi][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                grad_d_tpp(inp[n][cp][hi][wi_start + wb*(W/num_W_blocks)], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][wb*(W/num_W_blocks)], din[n][cp][hi][wi_start + wb*(W/num_W_blocks)]);
              } /* loop over wb */
              /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
              if (pad_w_in != 0 ) {
                zero_wp_tpp(din[n][cp][hi][wi_end]);
              }

            }
            /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
            if (pad_h_in != 0) {
              zero_hp_tpp(din[n][cp][hi_end][0]);
            }
          } else { /* for if (!use_hw_blocking) */
            for(int hwb = 0; hwb < num_HW_blocks; hwb++) {
              int ho = (hwb*(H*W/num_HW_blocks))/W;
              int hi = ho;
              int w  = (hwb*(H*W/num_HW_blocks))%W;
              grad_d_tpp(inp[n][cp][hi][w], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][w], din[n][cp][hi][w]);
            }
          } /* if-else for the presence of input padding */
        } /* end of cp loop */
      } /* end of n loop */
#endif /* THREADED_LOOPS */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_d scope */

}

return std::vector<at::Tensor>({t_grad_input, t_grad_input_add, t_grad_weight, t_grad_bias});


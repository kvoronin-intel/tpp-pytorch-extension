RECORD_FUNCTION("batchnorm_bwd", std::vector<c10::IValue>());

// inputs ~ grad_outs
//   ctx.save_for_backward(input, input_add, weight, mean, var, invstd, relu_mask, relu, eltwise, paddings, output)
//   inputs += ctx.saved_tensors

auto t_GO = inputs[0]; /* grad_output */
auto t_I  = inputs[1]; /* input */
auto t_IA = inputs[2]; /* input_add */
auto t_W  = inputs[3]; /* weight */
auto t_M  = inputs[4]; /* mean */
auto t_V  = inputs[5]; /* var  */
//auto t_IV = inputs[6]; /* invstd  */
auto t_R  = inputs[7]; /* relumask */

std::cout << "padding = " << padding << std::endl;

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
//long C  = CP * bc;

long ifhp = H + 2 * pad_h_in;
long ifwp = W + 2 * pad_w_in;
long ofhp = H + 2 * pad_h_out;
long ofwp = W + 2 * pad_w_out;

const float scale = 1.0f /((float)N * H * W);

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
/*
  DECL_VLA_PTR_PT    (T,             inp,      [N][CP][ifhp][ifwp][bc], t_I);
  DECL_VLA_PTR_PT    (float,         gamma,    [CP][bc],                t_W);
  //DECL_VLA_PTR_PT    (float,         beta,     [CP][bc],                t_B);
  DECL_VLA_PTR_PT    (float,         mean,     [CP][bc],                t_M);
  DECL_VLA_PTR_PT    (float,         var,      [CP][bc],                t_V);
  //DECL_VLA_PTR_PT    (T,             inp_add,  [N][CP][ifhp][ifwp][bc], t_IA);

  DECL_VLA_PTR_PT    (T,             dout,     [N][CP][ofhp][ofwp][bc], t_GO);
  DECL_VLA_PTR_PT    (unsigned char, relumask, [N][CP][ofhp][ofwp][bc], t_R);

  DECL_VLA_PTR_PT    (float, dgamma_N, [CP][N][bc],       scratch);
  DECL_VLA_PTR_PT_EXT(float, dbeta_N,  [CP][N][bc],       scratch, dbeta_N_offset);

  DECL_VLA_PTR_PT    (T,             din,      [N][CP][ifhp][ifwp][bc], t_grad_input);
  DECL_VLA_PTR_PT    (T,             din_add,  [N][CP][ifhp][ifwp][bc], t_grad_input_add);
  DECL_VLA_PTR_PT    (float,         dgamma,   [CP][bc],                t_grad_weight);
  DECL_VLA_PTR_PT    (float,         dbeta,    [CP][bc],                t_grad_bias);
*/
  DECL_VLA_PTR_PT    (T,             inp,      [CP][ifhp][ifwp][bc], t_I);
  DECL_VLA_PTR_PT    (float,         gamma,    [bc],                t_W);
  DECL_VLA_PTR_PT    (float,         mean,     [bc],                t_M);
  DECL_VLA_PTR_PT    (float,         var,      [bc],                t_V);

  DECL_VLA_PTR_PT    (T,             dout,     [CP][ofhp][ofwp][bc], t_GO);
  DECL_VLA_PTR_PT    (unsigned char, relumask, [CP][ofhp][ofwp][bc], t_R);

  DECL_VLA_PTR_PT    (float, dgamma_N, [N][bc],       scratch);
  DECL_VLA_PTR_PT_EXT(float, dbeta_N,  [N][bc],       scratch, dbeta_N_offset);

  DECL_VLA_PTR_PT    (T,             din,      [CP][ifhp][ifwp][bc], t_grad_input);
  DECL_VLA_PTR_PT    (T,             din_add,  [CP][ifhp][ifwp][bc], t_grad_input_add);
  DECL_VLA_PTR_PT    (float,         dgamma,   [bc],                t_grad_weight);
  DECL_VLA_PTR_PT    (float,         dbeta,    [bc],                t_grad_bias);

  auto zero_tpp = SCOPEIT(SetZeroTPP<float>(bc), EW_ZERO);

  auto coeffs_tpp = SCOPEIT(BatchNormStatCoeffsTPP<float>(bc, eps), NORMALIZE);

  auto helper_copy_tpp = SCOPEIT((CpyTPP<float>(1, bc, bc, bc)), EW_COPY); /* 1, bc because of row-major for unary */

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bc, bc, bc), EW_ADD); /* 1, bc because of row-major for unary */

  auto grad_w_inpadd_tpp = SCOPEIT((BatchNormBwdWTPP<T,T>(bc, spatial_block_size, relu, eltwise)), NORMALIZE);

  auto abc_coeffs_tpp = SCOPEIT(BatchNormABCCoeffsTPP<float>(bc, scale, eps), NORMALIZE);

  auto grad_d_tpp = SCOPEIT((BatchNormBwdDTPP<T,T>(bc, spatial_block_size)), NORMALIZE);

  {
    RECORD_SCOPE(bn_bwd_w_inpadd, {});
    {
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
            printf("First parallel for is not implemented for w blocking in bwd\n");
            exit(-1);
          } else {
            int hwb = 0, ho = 0, hi = 0, w = 0;
            for(hwb=0; hwb < num_HW_blocks; hwb++){
              ho = (hwb*(H*W/num_HW_blocks))/W;
              hi = ho;
              w  = (hwb*(H*W/num_HW_blocks))%W;

              //void operator()(Tin* inp, float *a, float *b, Tout *dout, float *dgamma_local, float *dbeta_local, float* gamma, Tin* din_add, unsigned char* relumask) {

              grad_w_inpadd_tpp(inp[n][cp][hi][w], &a[0], &b[0], dout[n][cp][ho][w], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][w] : NULL,
                                relu ? relumask[n][cp][ho][w] : NULL);
            }
          } /* if-else for the presence of input padding */

          helper_copy_tpp(&lcl_dgamma_ptr[0], dgamma_N[cp][n]);
          helper_copy_tpp(&lcl_dbeta_ptr[0],  dbeta_N [cp][n]);

        } /* end of cp loop */
      } /* end of n loop */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_w_inpadd scope */

  {
    RECORD_SCOPE(bn_bwd_w_add, {});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int cp = 0; cp < CP; cp++) {

        int ni = 0;

        zero_tpp(dgamma[cp]);
        zero_tpp(dbeta [cp]);

        for(ni = 0; ni < N; ni++){

          helper_add_tpp(dgamma[cp], dgamma_N[cp][ni], dgamma[cp]);
          helper_add_tpp(dbeta [cp], dbeta_N [cp][ni], dbeta [cp]);

        } /* end of ni loop */
      } /* end of cp loop */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_w_add scope */

  {
    RECORD_SCOPE(bn_bwd_d, {});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < N; n++) {
        for (int cp = 0; cp < CP; cp++) {

          LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
          LIBXSMM_ALIGNED(float b[bc], 64);
          LIBXSMM_ALIGNED(float c[bc], 64);

          //void operator()(Tin* gamma, Tin* dgamma, Tin* var, Tin* mean, Tin* dbeta, Tout* a, Tout* b, Tout* c)
          abc_coeffs_tpp(gamma[cp], dgamma[cp], var[cp], mean[cp], dbeta[cp], &a[0], &b[0], &c[0]);

          if (!use_hw_blocking) {
            printf("Third parallel for is not implemented for w blocking in bwd\n");
            exit(-1);
          } else {
            int hwb = 0, ho = 0, hi = 0, w = 0;
            for(hwb=0; hwb < num_HW_blocks; hwb++){
              ho = (hwb*(H*W/num_HW_blocks))/W;
              hi = ho;
              w  = (hwb*(H*W/num_HW_blocks))%W;

              //void operator()(Tin* inp, float* a, float* b, float *c, float *gamma, Tout* dout, Tout* din)
              grad_d_tpp(inp[n][cp][hi][w], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][w], din[n][cp][hi][w]);
            }
          } /* if-else for the presence of input padding */
        } /* end of cp loop */
      } /* end of n loop */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_d scope */

}

return std::vector<at::Tensor>({t_grad_input, t_grad_input_add, t_grad_weight, t_grad_bias});


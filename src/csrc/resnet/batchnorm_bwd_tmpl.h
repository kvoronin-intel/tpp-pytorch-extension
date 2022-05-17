RECORD_FUNCTION("batchnorm_bwd", std::vector<c10::IValue>());

// inputs ~ grad_outs
//   ctx.save_for_backward(input, input_add, weight, mean, var, invstd, relu_mask, relu, eltwise, paddings, output)
//   inputs += ctx.saved_tensors

//auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));

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

//        (grad_input, grad_input_add, grad_weight, grad_bias) = batchnorm_cpp.batchnorm_bwd( inputs )

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

//std::vector<long> output_size{N, CP, ofhp, ofwp, bc};
//std::cout << "output_size = " << output_size << std::endl;

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
  //DECL_VLA_PTR_PT    (float,         beta,     [CP][bc],                t_B);
  DECL_VLA_PTR_PT    (float,         mean,     [CP][bc],                t_M);
  DECL_VLA_PTR_PT    (float,         var,      [CP][bc],                t_V);
  DECL_VLA_PTR_PT    (T,             inp_add,  [N][CP][ifhp][ifwp][bc], t_IA);

  DECL_VLA_PTR_PT    (T,             dout,     [N][CP][ofhp][ofwp][bc], t_GO);
  DECL_VLA_PTR_PT    (unsigned char, relumask, [N][CP][ofhp][ofwp][bc], t_R);

  DECL_VLA_PTR_PT    (float, dgamma_N, [CP][N][bc],       scratch);
  DECL_VLA_PTR_PT_EXT(float, dbeta_N,  [CP][N][bc],       scratch, dbeta_N_offset);

  DECL_VLA_PTR_PT    (float,         dgamma,  [CP][bc], t_grad_weight);
  DECL_VLA_PTR_PT    (float,         dbeta,   [CP][bc], t_grad_bias);

  auto zero_tpp = SCOPEIT(SetZeroTPP<float>(bc), EW_ZERO);

  auto coeffs_tpp = SCOPEIT(BatchNormStatCoeffsTPP<float>(bc, eps), NORMALIZE);

  auto helper_copy_tpp = SCOPEIT((ConvertTPP<float, float>(1, bc, bc, bc)), EW_COPY); /* 1, bc because of row-major for unary */

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bc, bc, bc), EW_ADD); /* 1, bc because of row-major for unary */

  auto grad_coeffs_tpp = SCOPEIT((BatchNormBwdWTPP<T,T>(bc, spatial_block_size, relu, eltwise)), NORMALIZE);

/*
  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bc, bc, bc), EW_ADD);

  auto normalize_tpp = SCOPEIT((BatchNormFwdScale<T,T>(bc, spatial_block_size, relu, eltwise)), NORMALIZE);
*/

/*
  auto reduce_tpp = SCOPEIT(UnaryTPP(
            H * W / num_HW_blocks, bc, bc, bc,
            XsmmDtype<T>(), LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
            LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS,
            LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD), EW_RED);
*/
  {
    RECORD_SCOPE(bn_bwd_w_first, {});
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

          //float *dgamma_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, n, 0, N, bc);
          //float *dbeta_ncp_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, n, 0, N, bc);

          coeffs_tpp(mean[cp][0], var[cp][0], &a[0], &b[0]);

          if (!use_hw_blocking) {
            printf("First part of parallel for is not implemented for w blocking in bwd\n");
            exit(-1);
          } else {
            int hwb = 0, ho = 0, hi = 0, w = 0;
            for(hwb=0; hwb < num_HW_blocks; hwb++){
              ho = (hwb*(H*W/num_HW_blocks))/W;
              hi = ho;
              w  = (hwb*(H*W/num_HW_blocks))%W;

              //void operator()(Tin* inp, float *a, float *b, Tout *dout, float *dgamma_local, float *dbeta_local, float* gamma, Tin* din_add, unsigned char* relumask) {

              grad_coeffs_tpp(inp[n][cp][hi][w][0], &a[0], &b[0], dout[n][cp][ho][w][0], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp][0],
                                eltwise ? inp_add[n][cp][hi][w][0] : NULL,
                                relu ? relumask[n][cp][ho][w][0] : NULL);
            }
          }

          helper_copy_tpp(&lcl_dgamma_ptr[0], dgamma_N[cp][n][0]);
          helper_copy_tpp(&lcl_dbeta_ptr[0],  dbeta_N [cp][n][0]);

        } /* end of cp loop */
      } /* end of n loop */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_w_first scope */

  {
    RECORD_SCOPE(bn_bwd_w_second, {});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int cp = 0; cp < CP; cp++) {

        int ni = 0;

        zero_tpp(dgamma[cp][0]);
        zero_tpp(dbeta [cp][0]);

        for(ni = 0; ni < N; ni++){

          helper_add_tpp(dgamma[cp][0], dgamma_N[cp][ni][0], dgamma[cp][0]);
          helper_add_tpp(dbeta [cp][0], dbeta_N [cp][ni][0], dbeta [cp][0]);

          /* #pragma omp simd */
          /* for (int cb = 0; cb < bc; cb++) { */
          /*   pdgamma[cp*bc + cb] += dgamma_N[cp*N*bc + n*bc + cb];  */
          /*   pdbeta[cp*bc + cb] += dbeta_N[cp*N*bc + n*bc + cb];  */
          /* } */
        } /* end of ni loop */
      } /* end of cp loop */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_w_second scope */

}


return std::vector<at::Tensor>({t_grad_input, t_grad_input_add, t_grad_weight, t_grad_bias});

#if 0
  for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
    int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0, cb = 0;
    int n  = cpxnt%N;
    int cp = cpxnt/N;


    for(cb = 0; cb < bc; cb++){
      float lgamma  = LIBXSMM_VLA_ACCESS(2, gamma,  cp, cb, bc);
      float ldgamma = LIBXSMM_VLA_ACCESS(2, dgamma, cp, cb, bc);
      float lvar    = LIBXSMM_VLA_ACCESS(2, var,    cp, cb, bc);
      float lmean   = LIBXSMM_VLA_ACCESS(2, mean,   cp, cb, bc);
      float ldbeta  = LIBXSMM_VLA_ACCESS(2, dbeta,  cp, cb, bc);

      a[cb]        = lgamma / ((float)sqrt(lvar + eps));                            /* a = gamma_ptr[bc] * brstd_ptr[bc] */
      b[cb]        = -a[cb] * scale * ldgamma / ((float)sqrt(lvar + eps));          /* b = gamma_ptr[bc] * brstd_ptr[bc] * del_gamma_ptr[v] * brstd_ptr[bc] * recp_nhw */
      c[cb]        = -b[cb] * lmean - a[cb] * scale * ldbeta ;                      /* c = -gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * del_beta_ptr[bc] + gamma_ptr[bc] * brstd_ptr[bc] * recp_nhw * bmean_ptr[bc] * del_gamma_ptr[bc] * brstd_ptr[bc]) */
    }

    arg_array[1].primary = a;
    arg_array[2].primary = b;
    arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);
    arg_array[7].primary = c;

    if (cfg.use_hw_blocking == 0) { /* w-blocking */
      /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
         while the other arrays are non-shifted (and hence accesses require offsets */
      /* Notice: Zeroing out the rim for din is not strictly necessary but for safety is done here */
      /* zeroing out strip [0, hi_start) x ifwp x bc */
      if (cfg.pad_h_in != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
      for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
        /* zeroing out starting [0, wi_start) x bc block for fixed hi */
        if (cfg.pad_w_in != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
        for (wb = 0; wb < num_W_blocks; wb++) {
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp , n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary =    &LIBXSMM_VLA_ACCESS(5, din , n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
          cfg.din_func(&eqn_param);                                                                     /* din = dout * a + b * inp + c */
        }
        /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
        if (cfg.pad_w_in != 0) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_wp_kernel(&all_zero_param);
        }
      }
      /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
      if (cfg.pad_h_in != 0) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
        cfg.all_zero_hp_kernel(&all_zero_param);
      }
    } else { /* hw-blocking (implies no padding) */
      for(hwb=0; hwb < num_HW_blocks; hwb++){
        ho = (hwb*(HW/num_HW_blocks))/W;
        hi = ho;
        w  = (hwb*(HW/num_HW_blocks))%W;
        arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp , n, cp, hi, w, 0, CP, H, W, bc);
        arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

        eqn_param.inputs = arg_array;
        eqn_param.output.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, w, 0, CP, H, W, bc);
        cfg.din_func(&eqn_param);                                                                     /* din = dout * a + b * inp + c */
      } /* loop over hw blocks */
    } /* if-else for the presence of input padding */
  } /* loop over cpxnt for computing din */

#endif

#if 0
    {
      int cp = 0;
      int ni = 0;
      for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        cfg.all_zero_kernel(&all_zero_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   pdgamma[cp*bc + cb] = 0.0f; */
        /*   pdbeta[cp*bc + cb] = 0.0f; */
        /* } */

        for(ni = 0; ni < N; ni++){

          add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
          add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, ni, 0, N, bc);
          add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
          cfg.helper_add_kernel(&add_param);

          add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
          add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, ni, 0, N, bc);
          add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
          cfg.helper_add_kernel(&add_param);

          /* #pragma omp simd */
          /* for (int cb = 0; cb < bc; cb++) { */
          /*   pdgamma[cp*bc + cb] += dgamma_N[cp*N*bc + n*bc + cb];  */
          /*   pdbeta[cp*bc + cb] += dbeta_N[cp*N*bc + n*bc + cb];  */
          /* } */
        }
      } /* loop over cp and nt for computing dbeta and dgamma */
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

  } /* this is only computed in case of full backward (norm_type ~ 0) */


#endif


#if 0
    for ( cpxnt = thr_begin_dN; cpxnt < thr_end_dN; ++cpxnt ) {
      int hi = 0, ho = 0, w = 0, wb = 0, hwb = 0, cb = 0;
      int n  = cpxnt%N;
      int cp = cpxnt/N;
      LIBXSMM_ALIGNED(float lcl_dgamma_ptr[bc], 64);
      LIBXSMM_ALIGNED(float lcl_dbeta_ptr[bc], 64);

      float *dgamma_ncp_ptr = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, n, 0, N, bc);
      float *dbeta_ncp_ptr  = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, n, 0, N, bc);

      all_zero_param.out.primary = lcl_dgamma_ptr;
      cfg.all_zero_kernel(&all_zero_param);
      all_zero_param.out.primary = lcl_dbeta_ptr;
      cfg.all_zero_kernel(&all_zero_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   lcl_dgamma_ptr[cb] = 0.0f; */
      /*   lcl_dbeta_ptr[cb] = 0.0f; */
      /* } */

      for(cb = 0; cb < bc; cb++){
        float lvar   = LIBXSMM_VLA_ACCESS(2, var,   cp, cb, bc);
        float lmean  = LIBXSMM_VLA_ACCESS(2, mean,  cp, cb, bc);

        a[cb] = 1.0f / ((float)sqrt(lvar + eps));
        b[cb] = -a[cb] * lmean;

        /* a[cb] = 1.0f / ((float)sqrt(var[cp*bc + cb] + eps)); */
        /* b[cb] = -a[cb]*mean[cp*bc + cb];                     */
      }

      arg_array[1].primary = a;
      arg_array[2].primary = b;
      arg_array[4].primary = lcl_dgamma_ptr;
      arg_array[5].primary = lcl_dbeta_ptr;
      arg_array[6].primary = (void*)&LIBXSMM_VLA_ACCESS(2, gamma, cp, 0, bc);

      if (cfg.use_hw_blocking == 0) { /* w-blocking */
        /* Reminder: dout and relumask are already shifted by the offset (= point to the non-padded part already),
           while the other arrays are non-shifted (and hence accesses require offsets */
        /* Notice: Zeroing out the rim for din_add is not strictly necessary but for safety is done here */
        /* zeroing out strip [0, hi_start) */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }
        for (ho = 0, hi = hi_start; ho < H; ho++, hi++) {
          /* zeroing out starting [0, wi_start) x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }
          for (wb = 0; wb < num_W_blocks; wb++) {
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE ||
              cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
                all_relu_param.op.primary   = (void*)(&alpha);
                all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                                 (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc/8)
                                                 : NULL );
                all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);      /* [HW,bc] */
                cfg.inv_relu_kernel(&all_relu_param);
              } /* ReLU/mask */
              if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
                ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);
                ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
                cfg.ewise_copy_kernel(&ewise_copy_param);
              } /* Eltwise */
            }
            arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, wi_start + wb*(W/num_W_blocks), 0, CP, ifhp, ifwp, bc);
            arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho,            wb*(W/num_W_blocks), 0, CP, ofhp, ofwp, bc);

            eqn_param.inputs = arg_array;
            eqn_param.output.primary = lcl_dgamma_ptr;
            cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

            eqn_param.output.primary = lcl_dbeta_ptr;
            cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
          }

          /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
          if (cfg.pad_w_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
              ) {
            all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
            cfg.all_zero_wp_kernel(&all_zero_param);
          }

        }
        /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
        if (cfg.pad_h_in != 0 &&
              (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK)
            ) {
          all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
          cfg.all_zero_hp_kernel(&all_zero_param);
        }

      } else { /* hw-blocking (implies no padding) */
        for(hwb=0; hwb < num_HW_blocks; hwb++){
          ho = (hwb*(HW/num_HW_blocks))/W;
          hi = ho;
          w  = (hwb*(HW/num_HW_blocks))%W;
          if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE ||
            cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              all_relu_param.op.primary   = (void*)(&alpha);
              all_relu_param.in.primary   = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              all_relu_param.in.secondary = ((cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_RELU_WITH_MASK || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) ?
                                               (void*)&LIBXSMM_VLA_ACCESS(5, relumask, n, cp, ho, w, 0, CP, H, W, bc/8)
                                               : NULL /*&LIBXSMM_VLA_ACCESS(4, dout, n, cp, hwb*(HW/num_HW_blocks), 0, CP, HW, bc) */ ); /* dout_fwd ? nonsense? */
              all_relu_param.out.primary  = &LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);      /* [HW,bc] */
              cfg.inv_relu_kernel(&all_relu_param);
            } /* ReLU/mask */
            if (cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU || cfg.fuse_type == LIBXSMM_DNN_BN_FUSE_ELTWISE_RELU_WITH_MASK) {
              ewise_copy_param.in.primary  = &LIBXSMM_VLA_ACCESS(5, dout,    n, cp, ho, w, 0, CP, H, W, bc);
              ewise_copy_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, w, 0, CP, H, W, bc);
              cfg.ewise_copy_kernel(&ewise_copy_param);
            } /* Eltwise */
          }
          arg_array[0].primary = (void*)&LIBXSMM_VLA_ACCESS(5, inp,  n, cp, hi, w, 0, CP, H, W, bc);
          arg_array[3].primary = (void*)&LIBXSMM_VLA_ACCESS(5, dout, n, cp, ho, w, 0, CP, H, W, bc);

          eqn_param.inputs = arg_array;
          eqn_param.output.primary = lcl_dgamma_ptr;
          cfg.dgamma_func(&eqn_param);                                                             /* dgamma += (a * inp + b) * dout */

          eqn_param.output.primary = lcl_dbeta_ptr;
          cfg.dbeta_func(&eqn_param);                                                              /* dbeta += dout */
        } /* loop over hw blocks */
      } /* if-else for the presence of input padding */

      copy_param.in.primary = lcl_dgamma_ptr;
      copy_param.out.primary = dgamma_ncp_ptr;
      cfg.helper_copy_kernel(&copy_param);

      copy_param.in.primary = lcl_dbeta_ptr;
      copy_param.out.primary = dbeta_ncp_ptr;
      cfg.helper_copy_kernel(&copy_param);

      /* #pragma omp simd */
      /* for (int cb = 0; cb < bc; cb++) { */
      /*   dgamma_ncp_ptr[cb] = lcl_dgamma_ptr[cb]; */
      /*   dbeta_ncp_ptr[cb] = lcl_dbeta_ptr[cb]; */
      /* } */
    } /* loop over cpxnt for computing temporary n-local dbeta and dgamma */

    libxsmm_barrier_wait(cfg.barrier, ltid);

    {
      int cp = 0;
      int ni = 0;
      for ( cp = thr_begin_C; cp < thr_end_C; ++cp ) {
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
        cfg.all_zero_kernel(&all_zero_param);
        all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
        cfg.all_zero_kernel(&all_zero_param);

        /* #pragma omp simd */
        /* for (int cb = 0; cb < bc; cb++) { */
        /*   pdgamma[cp*bc + cb] = 0.0f; */
        /*   pdbeta[cp*bc + cb] = 0.0f; */
        /* } */

        for(ni = 0; ni < N; ni++){

          add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
          add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dgamma_N, cp, ni, 0, N, bc);
          add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dgamma, cp, 0, bc);
          cfg.helper_add_kernel(&add_param);

          add_param.in0.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
          add_param.in1.primary = &LIBXSMM_VLA_ACCESS(3, dbeta_N, cp, ni, 0, N, bc);
          add_param.out.primary = &LIBXSMM_VLA_ACCESS(2, dbeta, cp, 0, bc);
          cfg.helper_add_kernel(&add_param);

          /* #pragma omp simd */
          /* for (int cb = 0; cb < bc; cb++) { */
          /*   pdgamma[cp*bc + cb] += dgamma_N[cp*N*bc + n*bc + cb];  */
          /*   pdbeta[cp*bc + cb] += dbeta_N[cp*N*bc + n*bc + cb];  */
          /* } */
        }
      } /* loop over cp and nt for computing dbeta and dgamma */

#endif

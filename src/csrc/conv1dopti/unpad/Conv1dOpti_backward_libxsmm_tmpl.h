
    RECORD_FUNCTION("Conv1dOpti_backward_libxsmm", std::vector<c10::IValue>({grad, input, weight}));

    int64_t N_t = input.size(0);                                /* Batch */
    int64_t C_t = input.size(1);                                /* Channel */
    int64_t Win_t = input.size(2);                              /* input width */

    int64_t F_t = weight.size(0);                               /* Number of filters */
    int64_t WW_t = weight.size(2);                              /* filter width */

    int64_t dial = dilation;                                    /* dilation parameter */
    int64_t pad_size = ((WW_t- 1))*dial;                        /* Total padding size */
    int64_t W_t = Win_t - pad_size;                             /* output width */

    auto d_input = input.new_empty({N_t,C_t,Win_t});            /* declare data gradiant tensor */
    auto d_weight = weight.new_empty({F_t,C_t,WW_t});           /* declare weight gradiant tensor */

    float* weight_a = weight.data_ptr<float>();
    float* d_weight_a = d_weight.data_ptr<float>();

    DECL_VLA_PTR_PT(T, input_a, [C_t][Win_t], input);
    DECL_VLA_PTR_PT(T, grad_a, [F_t][W_t], grad);
    DECL_VLA_PTR_PT(T, d_input_a, [C_t][Win_t], d_input);

    /*  Backward data part of the code */

    auto flip_weight = weight.new_empty({WW_t,C_t,F_t});                    /* Tensor for permuted weights (width, channels, filters) */
    float* flip_weight_a = flip_weight.data_ptr<float>();

    auto weight_buffer = weight.new_empty({F_t,C_t,WW_t});                  /* Tensor weight buffer */
    float* weight_buffer_a = weight_buffer.data_ptr<float>();

    #pragma omp parallel for
    for(int i = 0; i < F_t*C_t; i++){
        for(int kw = 0; kw < WW_t; kw++){                                   /* reverse copy */
            flip_weight_a[i*WW_t + kw] = weight_a[i*WW_t + WW_t - kw - 1];
        }
    }

    /* jited tranpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, C_t, F_t) in two steps */

    int64_t flip_m1 = WW_t;
    int64_t flip_n1 = F_t*C_t;

    auto trans_flip_1 = SCOPEIT(XformExtTPP<T>(flip_n1, flip_m1, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    trans_flip_1(&flip_weight_a[0], &weight_buffer_a[0]);

    int64_t flip_m2 = C_t;
    int64_t flip_n2 = F_t;

    auto trans_flip_2 = SCOPEIT(XformExtTPP<T>(flip_n2, flip_m2, XformTPP::XFORM_XPOSE_TPP), XPOSE);

    /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
    #pragma omp parallel for
    for(int kw = 0; kw < WW_t; kw++){                          /* permute last two dimensions */
        trans_flip_2(&weight_buffer_a[kw*C_t*F_t], &flip_weight_a[kw*C_t*F_t]);
    }

    int64_t Wpad_t = W_t + 2*(WW_t - 1)*dial;
    int64_t tile_multiple = (Win_t/XS_TILE_DBACKWARD)*XS_TILE_DBACKWARD;

    int64_t lda = F_t;                                              /* Filters (15) */
    int64_t ldb_orig = W_t;                                         /* grad width 60000 */
    /* int ldb = Wpad_t;                                        //    Extra padded grad input case 60800 */
    int64_t ldc = Win_t;                                            /* Input width (60400) */
    unsigned long long l_br = WW_t;                             /* Number of batches for brGEMM (51) */

    int64_t pad_tile_multiple = 2 * (((WW_t - 1)*dial)/XS_TILE_DBACKWARD + 1) * XS_TILE_DBACKWARD;       /* 896 */
    auto grad_shortpad = grad.new_empty({N_t,F_t,2*pad_tile_multiple});
    DECL_VLA_PTR_PT(T, grad_shortpad_a, [F_t][2*pad_tile_multiple], grad_shortpad);
    int64_t ldb_shortpad = 2*pad_tile_multiple;                     /* grad pad 1792 */


    /* Dispatch brgemm kernels for normal and edge cases*/
    auto main_gemm_backdata_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(C_t, XS_TILE_DBACKWARD, F_t, C_t*F_t, dial, lda, ldb_orig, ldc, 1.0, 0, 1)));
    auto lr_gemm_backdata_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(C_t, XS_TILE_DBACKWARD, F_t, C_t*F_t, dial, lda, ldb_shortpad, ldc, 1.0, 0, 1)));
    auto edge_gemm_backdata_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(C_t, (Win_t - tile_multiple), F_t, C_t*F_t, dial, lda, ldb_shortpad, ldc, 1.0, 0, 1)));

    /* Virtual copy kernels */
    int64_t virtual_m1 = pad_tile_multiple - ((WW_t - 1)*dial);     /* columns */
    int64_t virtual_m2 = ((WW_t - 1)*dial);                         /* columns */
    int64_t virtual_n = F_t;                                        /* rows */
    int64_t ldi_virtual = W_t;
    int64_t ldo_virtual = 2*pad_tile_multiple;

    if (ldi_virtual < virtual_m1){                                          /* corner case when width's are very small */
        virtual_m1 = ldi_virtual;
        auto all_zero_backdata = SCOPEIT(SetZeroTPP<T>(virtual_n, ldo_virtual, ldo_virtual), EW_ZERO);
        #pragma omp parallel for
        for(int n = 0; n < N_t; n++){
            all_zero_backdata(&grad_shortpad_a[n][0][0]);
        }
    }

    auto virtual_copy = SCOPEIT(CpyTPP<T>(virtual_n, virtual_m1, ldi_virtual, ldo_virtual), EW_COPY);
    auto virtual_copy_zero = SCOPEIT(SetZeroTPP<T>(virtual_n, virtual_m2, ldo_virtual), EW_ZERO);

    /* Loops for storing the edge portion of gradinant array into grad_shortpad_a */
    {
    RECORD_SCOPE(virtul_copy_loop, {grad_a, grad_shortpad_a});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for
            for(int n = 0; n < N_t; n++){

                virtual_copy_zero(&grad_shortpad_a[n][0][0]);

                virtual_copy(&grad_a[n][0][0], &grad_shortpad_a[n][0][(WW_t - 1)*dial]);
                virtual_copy(&grad_a[n][0][W_t - virtual_m1], &grad_shortpad_a[n][0][ldo_virtual - virtual_m1 - ((WW_t - 1)*dial)]);

                virtual_copy_zero(&grad_shortpad_a[n][0][ldo_virtual - ((WW_t - 1)*dial)]);
            }
        }
    }

    /* JIT eltwise TPPs for initialization... */
    int64_t tpp_m1 = XS_TILE_DBACKWARD;                  /* columns */
    int64_t tpp_m2 = Win_t - tile_multiple;              /* columns */
    int64_t tpp_n = C_t;                                 /* rows */
    int64_t ld_zero = Win_t;
    
    auto main_zero_tpp = SCOPEIT(SetZeroTPP<T>(tpp_n, tpp_m1, ld_zero), EW_ZERO);
    auto edge_zero_tpp = SCOPEIT(SetZeroTPP<T>(tpp_n, tpp_m2, ld_zero), EW_ZERO);

    /* Main compute kernel */
    {
    RECORD_SCOPE(backward_data_loop, {d_input_a, grad_a, flip_weight_a});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for
            for(int n = 0; n < N_t; n++) {
                int last_block=0;

                for(int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

                    main_zero_tpp(&d_input_a[n][0][wb]);

                    if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD){       /* Main case */

                        main_gemm_backdata_tpp(&flip_weight_a[0], &grad_a[n][0][wb - (WW_t-1)*dial], &d_input_a[n][0][wb], l_br);
                    }
                    else if (wb < (WW_t-1)*dial){                                                      /* Right side case */

                        lr_gemm_backdata_tpp(&flip_weight_a[0], &grad_shortpad_a[n][0][wb], &d_input_a[n][0][wb], l_br);
                    }
                    else{                                                                              /* left side case */

                        lr_gemm_backdata_tpp(&flip_weight_a[0], &grad_shortpad_a[n][0][wb - Wpad_t + 2*pad_tile_multiple], &d_input_a[n][0][wb], l_br);
                    }

                    last_block = wb;                                                                    /* store position for last block */
                }

                if (Win_t % XS_TILE_DBACKWARD != 0){                                                    /* Edge case */

                    edge_zero_tpp(&d_input_a[n][0][last_block + XS_TILE_DBACKWARD]);
                    edge_gemm_backdata_tpp(&flip_weight_a[0], &grad_shortpad_a[n][0][last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple], &d_input_a[n][0][last_block + XS_TILE_DBACKWARD], l_br);
                }
            }
        }
    }


    /* ------------------------------- Backward weight part of the code --------------------------------- */

    auto flip_d_weight = weight.new_empty({WW_t,C_t,F_t});                  /* Tensor for storing permuted weight gradiant */
    float* flip_d_weight_a = flip_d_weight.data_ptr<float>();

    for(int w = 0; w < F_t*C_t*WW_t; w++){
        flip_d_weight_a[w] = 0.0f;
    }


    l_br = WW_t;
    tile_multiple = (W_t/XS_TILE_WBACKWARD)*XS_TILE_WBACKWARD;

    /* Blocking on grad_a */
    int64_t lda_g = Win_t;
    /* int ldb_g = W_t; */
    int64_t ldb_trans_g = F_t;
    int64_t ldc_g = F_t;

    int64_t short_W_t = XS_TILE_WBACKWARD;
    int64_t edge_W_t = W_t - tile_multiple;
    int64_t M_g = W_t;
    int64_t N_g = F_t;


    auto grad_shorttrans = grad.new_empty({N_t,F_t,short_W_t});              /* Tensor for storing transposed short buffer */
    DECL_VLA_PTR_PT(T, grad_shorttrans_a, [F_t][short_W_t], grad_shorttrans);

    auto grad_edgetrans = grad.new_empty({N_t,F_t,edge_W_t});                /* Tensor for storing transposed short buffer in edge case */
    DECL_VLA_PTR_PT(T, grad_edgetrans_a, [F_t][edge_W_t], grad_edgetrans);


    auto short_trans_tpp = SCOPEIT(XformExtTPP<T>(N_g, short_W_t, short_W_t, N_g, M_g, N_g, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    auto edge_trans_tpp = SCOPEIT(XformExtTPP<T>(N_g, edge_W_t, edge_W_t, N_g, M_g, N_g, XformTPP::XFORM_XPOSE_TPP), XPOSE);


    /* Dispatch brGEMM kernel for normal and edge cases*/

    auto main_gemm_backweight_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(C_t, F_t, XS_TILE_WBACKWARD, 1, 1, lda_g, ldb_trans_g, ldc_g, 1.0, 0, 1)));
    auto edge_gemm_backweight_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(C_t, F_t, (W_t - tile_multiple), 1, 1, lda_g, ldb_trans_g, ldc_g, 1.0, 0, 1)));

    /* Main compute loop for backward weight */
    {
    RECORD_SCOPE(backward_weight_loop, {flip_d_weight_a, input_a, grad_a});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])                /* Distribute the weight array */
            for(int n = 0; n < N_t; n++) {
                int last_block = 0;

                for(int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {     /* Normal case */

                    short_trans_tpp(&grad_a[n][0][wb],  &grad_shorttrans_a[n][0][0] );

                    for(int kw = 0; kw < WW_t; kw++) {

                        main_gemm_backweight_tpp(&input_a[n][0][wb + kw*dial], &grad_shorttrans_a[n][0][0], &flip_d_weight_a[kw*C_t*F_t], 1);
                    }
                    last_block = wb;
                }

                if (W_t % XS_TILE_WBACKWARD != 0){

                    edge_trans_tpp(&grad_a[n][0][last_block + XS_TILE_WBACKWARD], &grad_edgetrans_a[n][0][0]);

                    for(int kw = 0; kw < WW_t; kw++) {

                        edge_gemm_backweight_tpp(&input_a[n][0][(last_block + XS_TILE_WBACKWARD) + kw*dial], &grad_edgetrans_a[n][0][0], &flip_d_weight_a[kw*F_t*C_t], 1);
                    }
                }
            }
        }
    }


    /* jited tranpose to permute the array dimensions
        Overall Convert (WW_t, C_t, F_t) -----> (F_t, C_t, WW_t)*/
    int per_m1 = F_t;
    int per_n1 = C_t;

    auto trans_permute_1 = SCOPEIT(XformExtTPP<T>(per_n1, per_m1, XformTPP::XFORM_XPOSE_TPP), XPOSE);

    /* Convert (WW_t, C_t, F_t) -----> (WW_t, F_t, C_t) */
    #pragma omp parallel for
    for(int kw = 0; kw < WW_t; kw++){                           /* permute last two dimensions */

        trans_permute_1(&flip_d_weight_a[kw*C_t*F_t], &flip_weight_a[kw*C_t*F_t]);
    }


    int per_m2 = F_t*C_t;
    int per_n2 = WW_t;

    auto trans_permute_2 = SCOPEIT(XformExtTPP<T>(per_n2, per_m2, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
    trans_permute_2(&flip_weight_a[0], &d_weight_a[0]);


    return {d_input, d_weight};         /* return data gradiant and weight gradiant */




// std::tuple<at::Tensor, at::Tensor>
// Conv1dOpti_backward_libxsmm(at::Tensor& grad, at::Tensor& input, at::Tensor& weight, int dilation){

//     /* RECORD_FUNCTION("Conv1dOpti_backward_libxsmm", std::vector<c10::IValue>({grad, input, weight})); */

//     int64_t N_t = input.size(0);                                /* Batch */
//     int64_t C_t = input.size(1);                                /* Channel */
//     int64_t Win_t = input.size(2);                              /* input width */

//     int64_t F_t = weight.size(0);                               /* Number of filters */
//     int64_t WW_t = weight.size(2);                              /* filter width */

//     int64_t dial = dilation;                                    /* dilation parameter */
//     int64_t pad_size = ((WW_t- 1))*dial;                        /* Total padding size */
//     int64_t W_t = Win_t - pad_size;                             /* output width */

//     auto d_input = input.new_empty({N_t,C_t,Win_t});            /* declare data gradiant tensor */
//     auto d_weight = weight.new_empty({F_t,C_t,WW_t});           /* declare weight gradiant tensor */

//     float* input_a = input.data_ptr<float>();                   /* Get data pointers for accessing tensors */
//     float* weight_a = weight.data_ptr<float>();
//     float* grad_a = grad.data_ptr<float>();
//     float* d_input_a = d_input.data_ptr<float>();
//     float* d_weight_a = d_weight.data_ptr<float>();

//     /*  Backward data part of the code */

//     auto flip_weight = weight.new_empty({WW_t,C_t,F_t});                    /* Tensor for permuted weights (width, channels, filters) */
//     float* flip_weight_a = flip_weight.data_ptr<float>();


//     auto weight_buffer = weight.new_empty({F_t,C_t,WW_t});                  /* Tensor weight buffer */
//     float* weight_buffer_a = weight_buffer.data_ptr<float>();

//     #pragma omp parallel for
//     for(int i = 0; i < F_t*C_t; i++){
//         for(int kw = 0; kw < WW_t; kw++){                                   /* reverse copy */
//             flip_weight_a[i*WW_t + kw] = weight_a[i*WW_t + WW_t - kw - 1];
//         }
//     }

//     /* jited tranpose to permute the array dimensions
//         Overall convert (F_t, C_t, WW_t) -----> (WW_t, C_t, F_t)*/

//     libxsmm_blasint flip_m1 = WW_t;
//     libxsmm_blasint flip_n1 = F_t*C_t;
//     libxsmm_blasint flip_ldi_1 = WW_t;
//     libxsmm_blasint flip_ldo_1 = F_t*C_t;

//     libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( flip_m1, flip_n1, flip_ldi_1, flip_ldo_1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_unary_flip_1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_unary_flip_1 == NULL) {
//         fprintf( stderr, "JIT unary TPP for trans_unary_flip_1 (NORM_TO_NORMT transform) in backward data pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t) */
//     libxsmm_meltw_unary_param trans_unary_param_flip_1;
//     trans_unary_param_flip_1.in.primary  = flip_weight_a;
//     trans_unary_param_flip_1.out.primary = weight_buffer_a;
//     trans_unary_flip_1( &trans_unary_param_flip_1 );

//     libxsmm_blasint flip_m2 = C_t;
//     libxsmm_blasint flip_n2 = F_t;
//     libxsmm_blasint flip_ldi_2 = C_t;
//     libxsmm_blasint flip_ldo_2 = F_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( flip_m2, flip_n2, flip_ldi_2, flip_ldo_2, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_unary_flip_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_unary_flip_2 == NULL) {
//         fprintf( stderr, "JIT unary TPP for trans_unary_flip_2 (NORM_TO_NORMT transform) in backward data pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
//     #pragma omp parallel for
//     for(int kw = 0; kw < WW_t; kw++){                          /* permute last two dimensions */
//         libxsmm_meltw_unary_param trans_unary_param_flip_2;
//         trans_unary_param_flip_2.in.primary  = &weight_buffer_a[kw*C_t*F_t];
//         trans_unary_param_flip_2.out.primary = &flip_weight_a[kw*C_t*F_t];
//         trans_unary_flip_2( &trans_unary_param_flip_2 );
//     }

//     int64_t Wpad_t = W_t + 2*(WW_t - 1)*dial;
//     int64_t tile_multiple = (Win_t/XS_TILE_DBACKWARD)*XS_TILE_DBACKWARD;

//     int lda = F_t;                                              /* Filters (15) */
//     int ldb_orig = W_t;                                         /* grad width 60000 */
//     /* int ldb = Wpad_t;                                        //    Extra padded grad input case 60800 */
//     int ldc = Win_t;                                            /* Input width (60400) */
//     unsigned long long l_br = WW_t;                             /* Number of batches for brGEMM (51) */

//     libxsmm_gemm_flags l_flags;
//     libxsmm_gemm_prefetch_type l_prefetch;
//     libxsmm_gemm_shape l_shape;
//     libxsmm_gemm_batch_reduce_config l_brconfig;

//     l_flags = LIBXSMM_GEMM_FLAG_NONE;
//     l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

//     l_shape = libxsmm_create_gemm_shape(XS_TILE_DBACKWARD, C_t, F_t, ldb_orig, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
//     l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
//     l_brconfig.br_stride_a_hint = dial*sizeof(float);
//     l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(float);
//     libxsmm_gemmfunction backdata_kernel_main = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

//     int pad_tile_multiple = 2 * (((WW_t - 1)*dial)/XS_TILE_DBACKWARD + 1) * XS_TILE_DBACKWARD;       /* 896 */

//     auto grad_shortpad_tensor = grad.new_empty({N_t,F_t,2*pad_tile_multiple});
//     float* grad_a_shortpad = grad_shortpad_tensor.data_ptr<float>();


//     int ldb_shortpad = 2*pad_tile_multiple;                     /* grad pad 1792 */

//     /* Dispatch kernels for normal and edge cases*/

//     l_shape = libxsmm_create_gemm_shape(XS_TILE_DBACKWARD, C_t, F_t, ldb_shortpad, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
//     l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
//     l_brconfig.br_stride_a_hint = dial*sizeof(float);
//     l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(float);
//     libxsmm_gemmfunction backdata_kernel_lr = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

//     l_shape = libxsmm_create_gemm_shape(Win_t - tile_multiple, C_t, F_t, ldb_shortpad, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
//     l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
//     l_brconfig.br_stride_a_hint = dial*sizeof(float);
//     l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(float);
//     libxsmm_gemmfunction backdata_kernel_edge = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

//     /* Virtual copy kernels */
//     libxsmm_blasint virtual_m1 = pad_tile_multiple - ((WW_t - 1)*dial);     /* columns */
//     libxsmm_blasint virtual_m2 = ((WW_t - 1)*dial);                         /* columns */
//     libxsmm_blasint virtual_n = F_t;                                        /* rows */
//     libxsmm_blasint ldi_virtual = W_t;
//     libxsmm_blasint ldo_virtual = 2*pad_tile_multiple;

//     if (ldi_virtual < virtual_m1){                                          /* corner case when width's are very small */
//         virtual_m1 = ldi_virtual;
//         unary_shape = libxsmm_create_meltw_unary_shape( ldo_virtual, virtual_n, ldo_virtual, ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//         libxsmm_meltwfunction_unary all_zero_backdata = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//         if ( all_zero_backdata == NULL) {
//             fprintf( stderr, "JIT unary all zero intilization kernel in backward data pass failed. Bailing...!\n");
//             exit(-1);
//         }
//         #pragma omp parallel for
//         for(int n = 0; n < N_t; n++){
//             libxsmm_meltw_unary_param all_zero_params;
//             all_zero_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];     /* Initialize the entire array when widths are small */
//             all_zero_backdata(&all_zero_params);
//         }
//     }

//     unary_shape = libxsmm_create_meltw_unary_shape( virtual_m1, virtual_n, ldi_virtual, ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary virtual_copy = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     unary_shape = libxsmm_create_meltw_unary_shape( virtual_m2, virtual_n, virtual_m2, ldo_virtual, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary virtual_copy_zero = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

//     if ((virtual_copy == NULL) || (virtual_copy_zero == NULL)) {
//         fprintf( stderr, "JIT unary kernel of virtual_copy in backward data pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Loops for storing the edge portion of gradinant array into grad_a_shortpad */
//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++){
//         libxsmm_meltw_unary_param vcopy_params, vcopy_params_zero;                                                  /* Copy parameter variable for holding the pointer */

//         vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];                                        /* copy zeros */
//         virtual_copy_zero(&vcopy_params_zero);

//         vcopy_params.in.primary = &grad_a[n*F_t*W_t];                                                              /* copy after zeros from start of the grad array */
//         vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ((WW_t - 1)*dial)];
//         virtual_copy(&vcopy_params);

//         vcopy_params.in.primary = &grad_a[n*F_t*W_t + W_t - virtual_m1];                                           /* copy from the end of the grad array */
//         vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - virtual_m1 - ((WW_t - 1)*dial)];
//         virtual_copy(&vcopy_params);

//         vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - ((WW_t - 1)*dial)];     /* copy zeros */
//         virtual_copy_zero(&vcopy_params_zero);
//     }

// /*
// #else

//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++){                       // Loops for storing the edge portion of gradinant array into grad_a_shortpad
//         for(int filter=0; filter < F_t; filter++){
//             for(int w = 0; w < pad_tile_multiple; w++){
//                 // initialize start of array
//                 if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
//                 }
//                 else{
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w] = 0.0f;
//                 }
//             }
//             for(int w = Wpad_t - pad_tile_multiple; w < Wpad_t ; w++){
//                 // initialize end of array
//                 if (w >= ((WW_t - 1)*dial) && w < (W_t + (WW_t - 1)*dial)){
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = grad_a[n*F_t*W_t + filter*W_t + w - (WW_t - 1)*dial];
//                 }
//                 else{
//                     grad_a_shortpad[n*F_t*2*pad_tile_multiple + filter*2*pad_tile_multiple + w - Wpad_t + 2*pad_tile_multiple] = 0.0f;
//                 }
//             }
//         }
//     }

// #endif
// */

//     /* JIT eltwise TPPs for initialization... */
//     libxsmm_blasint tpp_m1 = XS_TILE_DBACKWARD;                  /* columns */
//     libxsmm_blasint tpp_m2 = Win_t - tile_multiple;              /* columns */
//     libxsmm_blasint tpp_n = C_t;                                 /* rows */
//     libxsmm_blasint ld_zero = Win_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary copy_kernel_main = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary copy_kernel_edge = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

//     if ((copy_kernel_main == NULL) || (copy_kernel_edge == NULL)) {
//         fprintf( stderr, "JIT unary kernel for copy_kernel_main in backward data pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Main compute kernel */
//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++) {
//         int last_block=0;

//         libxsmm_meltw_unary_param copy_params_main, copy_params_edge;
//         libxsmm_gemm_param gemm_param_main, gemm_param_lr, gemm_param_edge;

//         for(int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

//             copy_params_main.out.primary = &d_input_a[n*C_t*Win_t + wb];                      /* Initialization */
//             copy_kernel_main(&copy_params_main);

//             if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD){       /* Main case */
//                 gemm_param_main.a.primary = &grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial];
//                 gemm_param_main.b.primary = &flip_weight_a[0];
//                 gemm_param_main.c.primary = &d_input_a[n*C_t*Win_t + wb];
//                 gemm_param_main.op.tertiary = &l_br;
//                 backdata_kernel_main( &gemm_param_main );
//             }
//             else if (wb < (WW_t-1)*dial){                                                      /* Right side case */
//                 gemm_param_lr.a.primary = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb];
//                 gemm_param_lr.b.primary = &flip_weight_a[0];
//                 gemm_param_lr.c.primary = &d_input_a[n*C_t*Win_t + wb];
//                 gemm_param_lr.op.tertiary = &l_br;
//                 backdata_kernel_lr( &gemm_param_lr );
//             }
//             else{                                                                              /* left side case */
//                 gemm_param_lr.a.primary = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple];
//                 gemm_param_lr.b.primary = &flip_weight_a[0];
//                 gemm_param_lr.c.primary = &d_input_a[n*C_t*Win_t + wb];
//                 gemm_param_lr.op.tertiary = &l_br;
//                 backdata_kernel_lr( &gemm_param_lr );
//             }

//             last_block = wb;                                                                    /* store position for last block */
//         }

//         if (Win_t % XS_TILE_DBACKWARD != 0){                                                    /* Edge case */

//             copy_params_edge.out.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];            /* Initialization */
//             copy_kernel_edge(&copy_params_edge);

//             gemm_param_edge.a.primary = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple];
//             gemm_param_edge.b.primary = &flip_weight_a[0];
//             gemm_param_edge.c.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];
//             gemm_param_edge.op.tertiary = &l_br;
//             backdata_kernel_edge( &gemm_param_edge );
//         }
//     }



//     /* ------------------------------- Backward weight part of the code --------------------------------- */

//     auto flip_d_weight = weight.new_empty({WW_t,C_t,F_t});                  /* Tensor for storing permuted weight gradiant */
//     float* flip_d_weight_a = flip_d_weight.data_ptr<float>();

//     for(int w = 0; w < F_t*C_t*WW_t; w++){
//         flip_d_weight_a[w] = 0.0f;
//     }

//     /* lda = W_t; */
//     /* ldb = Win_t; */
//     /* int ldb_trans = C_t; */
//     /* ldc = C_t; */
//     l_br = WW_t;
//     tile_multiple = (W_t/XS_TILE_WBACKWARD)*XS_TILE_WBACKWARD;

//     /* Blocking on grad_a */
//     int lda_g = Win_t;
//     /* int ldb_g = W_t; */
//     int ldb_trans_g = F_t;
//     int ldc_g = F_t;

//     libxsmm_blasint short_W_t = XS_TILE_WBACKWARD;
//     libxsmm_blasint edge_W_t = W_t - tile_multiple;
//     libxsmm_blasint M_g = W_t;
//     libxsmm_blasint N_g = F_t;


//     auto grad_shorttrans_tensor = grad.new_empty({N_t,F_t,short_W_t});              /* Tensor for storing transposed short buffer */
//     float* grad_shorttrans = grad_shorttrans_tensor.data_ptr<float>();

//     auto grad_edgetrans_tensor = grad.new_empty({N_t,F_t,edge_W_t});                /* Tensor for storing transposed short buffer in edge case */
//     float* grad_edgetrans = grad_edgetrans_tensor.data_ptr<float>();

//     /* use jited tranpose */
//     unary_shape = libxsmm_create_meltw_unary_shape( short_W_t, N_g, M_g, N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_shortkernel_grad = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     unary_shape = libxsmm_create_meltw_unary_shape( edge_W_t, N_g, M_g, N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_edgekernel_grad = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

//     if ((trans_shortkernel_grad == NULL) || (trans_edgekernel_grad == NULL)) {
//         fprintf( stderr, "JIT unary TPP for trans_shortkernel_grad (NORM_TO_NORM transform) failed in backward weight pass. Bailing...!\n");
//         exit(-1);
//     }

//     /* Dispatch brGEMM kernel for normal and edge cases*/
//     l_flags = LIBXSMM_GEMM_FLAG_NONE;
//     l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

//     l_shape = libxsmm_create_gemm_shape(F_t, C_t, XS_TILE_WBACKWARD, ldb_trans_g, lda_g, ldc_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
//     libxsmm_gemmfunction backweight_kernel_main = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch);

//     l_shape = libxsmm_create_gemm_shape(F_t, C_t, W_t - tile_multiple, ldb_trans_g, lda_g, ldc_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
//     libxsmm_gemmfunction backweight_kernel_edge = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch);

//     /* Main compute loop for backward weight */
//     #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])                /* Distribute the weight array */
//     for(int n = 0; n < N_t; n++) {
//         int last_block = 0;
//         libxsmm_meltw_unary_param trans_param_short, trans_param_edge;                   /* Pointer to hold trans short and edge */
//         libxsmm_gemm_param gemm_param_main, gemm_param_edge;

//         for(int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {     /* Normal case */

//             trans_param_short.in.primary  = &grad_a[n*F_t*W_t + wb];
//             trans_param_short.out.primary = &grad_shorttrans[n*F_t*short_W_t];
//             trans_shortkernel_grad( &trans_param_short );

//             for(int kw = 0; kw < WW_t; kw++) {
//                 gemm_param_main.a.primary = &grad_shorttrans[n*F_t*short_W_t];
//                 gemm_param_main.b.primary = &input_a[n*C_t*Win_t + wb + kw*dial];
//                 gemm_param_main.c.primary = &flip_d_weight_a[kw*C_t*F_t];
//                 backweight_kernel_main( &gemm_param_main );
//             }
//             last_block = wb;
//         }

//         if (W_t % XS_TILE_WBACKWARD != 0){

//             trans_param_edge.in.primary  = &grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD];
//             trans_param_edge.out.primary = &grad_edgetrans[n*F_t*edge_W_t];
//             trans_edgekernel_grad( &trans_param_edge );

//             for(int kw = 0; kw < WW_t; kw++) {
//                 gemm_param_edge.a.primary = &grad_edgetrans[n*F_t*edge_W_t];
//                 gemm_param_edge.b.primary = &input_a[n*C_t*Win_t + (last_block + XS_TILE_WBACKWARD) + kw*dial];
//                 gemm_param_edge.c.primary = &flip_d_weight_a[kw*F_t*C_t];
//                 backweight_kernel_edge( &gemm_param_edge );
//             }
//         }
//     }


//     /* jited tranpose to permute the array dimensions
//         Overall Convert (WW_t, C_t, F_t) -----> (F_t, C_t, WW_t)*/
//     libxsmm_blasint per_m1 = F_t;
//     libxsmm_blasint per_n1 = C_t;
//     libxsmm_blasint ldi_1 = F_t;
//     libxsmm_blasint ldo_1 = C_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( per_m1, per_n1, ldi_1, ldo_1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_permute_1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_permute_1 == NULL) {
//         fprintf( stderr, "JIT unary TPP for trans_permute_1 (NORM_TO_NORMT) in backward weight pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Convert (WW_t, C_t, F_t) -----> (WW_t, F_t, C_t) */
//     #pragma omp parallel for
//     for(int kw = 0; kw < WW_t; kw++){                           /* permute last two dimensions */
//         libxsmm_meltw_unary_param trans_param_permute_1;
//         trans_param_permute_1.in.primary  = &flip_d_weight_a[kw*C_t*F_t];
//         trans_param_permute_1.out.primary = &flip_weight_a[kw*C_t*F_t];
//         trans_permute_1( &trans_param_permute_1 );
//     }


//     libxsmm_blasint per_m2 = F_t*C_t;
//     libxsmm_blasint per_n2 = WW_t;
//     libxsmm_blasint ldi_2 = F_t*C_t;
//     libxsmm_blasint ldo_2 = WW_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( per_m2, per_n2, ldi_2, ldo_2, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_permute_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_permute_2 == NULL) {
//         fprintf( stderr, "JIT unary TPP for trans_permute_2 (NORM_TO_NORMT) in backward weight pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
//     libxsmm_meltw_unary_param trans_param_permute_2;
//     trans_param_permute_2.in.primary  = flip_weight_a;
//     trans_param_permute_2.out.primary = d_weight_a;
//     trans_permute_2( &trans_param_permute_2 );


//     return {d_input, d_weight};         /* return data gradiant and weight gradiant */
// }
RECORD_FUNCTION("Conv1dOpti_forward_libxsmm", std::vector<c10::IValue>({input, weight}));    // For recording time

    int N_t = input.size(0);                                 /* Batch */
    int C_t = input.size(1);                                 /* Channel */
    int Win_t = input.size(2);                               /* input width */

    int F_t = weight.size(0);                                /* Number of filters */
    int WW_t = weight.size(2);                               /* filter width */

    int dial = dilation;                                     /* dilation parameter */
    int pad_size = ((WW_t- 1))*dial;                         /* Total padding size */
    int W_t = Win_t - pad_size;                              /* output width */

    auto Y = input.new_empty({N_t,F_t,W_t});                   /* output */

    DECL_VLA_PTR_PT(T, input_a, [C_t][Win_t], input);
    DECL_VLA_PTR_PT(T, weight_a, [C_t][WW_t], weight);
    DECL_VLA_PTR_PT(T, Y_a, [F_t][W_t], Y);

    auto flip_weight = weight.new_empty({WW_t,F_t,C_t});        /* Array to store permuted weight tensor (width, filters, channels) */
    DECL_VLA_PTR_PT(T, flip_weight_a, [F_t][C_t], flip_weight);

    /* tranpose to permute the array dimensions
    Overall convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)*/

    auto trans_permute_tpp = SCOPEIT(XformExtTPP<T>(F_t*C_t, WW_t, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    trans_permute_tpp(&weight_a[0][0][0], &flip_weight_a[0][0][0]);

    int lda = C_t;                      /* Input channels (16) */
    int ldb = Win_t;                    /* Input width    (60400) */
    int ldc = W_t;                      /* Output width   (60000) */
    unsigned long long l_br = WW_t;     /* Number of batches in brGEMM (= width of kernel = 51) */

    int tile_multiple = (W_t/XS_TILE_FORWARD)*XS_TILE_FORWARD;                                          /* Number of blocks/Tiles in the output width */

    /* JIT eltwise TPPs for initialization... */
    int tpp_m1 = XS_TILE_FORWARD;                      /* columns */
    int tpp_m2 = W_t - tile_multiple;                  /* columns */
    int tpp_n = F_t;                                   /* rows */
    int tpp_k = C_t;
    int ld_zero = W_t;

    auto main_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(tpp_n, tpp_m1, tpp_k, F_t*C_t, dial, lda, ldb, ldc, 1.0, 0, 1)));
    auto edge_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(tpp_n, tpp_m2, tpp_k, F_t*C_t, dial, lda, ldb, ldc, 1.0, 0, 1)));

    auto main_zero_tpp = SCOPEIT(SetZeroTPP<T>(tpp_n, tpp_m1, ld_zero), EW_ZERO);
    auto edge_zero_tpp = SCOPEIT(SetZeroTPP<T>(tpp_n, tpp_m2, ld_zero), EW_ZERO);
    /* Main compute loop */
    {
    RECORD_SCOPE(forward_loop, {Y_a, input_a, flip_weight_a});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for
            for(int n = 0; n < N_t; n++) {                               /* Loop for batches */
                int last_block = 0;
                for(int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb += XS_TILE_FORWARD) {    /* width blocking loop (Main case) */
                    
                    main_zero_tpp(&Y_a[n][0][wb]);
                    main_brgemm_tpp(&flip_weight_a[0][0][0], &input_a[n][0][wb], &Y_a[n][0][wb], l_br);

                    last_block = wb;
                }

                if (W_t % XS_TILE_FORWARD != 0){                        /* Edge Case */

                    edge_zero_tpp(&Y_a[n][0][last_block + XS_TILE_FORWARD]);
                    edge_brgemm_tpp(&flip_weight_a[0][0][0], &input_a[n][0][last_block + XS_TILE_FORWARD], &Y_a[n][0][last_block + XS_TILE_FORWARD], l_br);
                }
            }
        }
    }

    return Y;


// at::Tensor Conv1dOpti_forward_libxsmm(at::Tensor& input, at::Tensor& weight, int dilation){

//     /* RECORD_FUNCTION("Conv1dOpti_forward_libxsmm", std::vector<c10::IValue>({input, weight}));    // For recording time   */

//     int64_t N_t = input.size(0);                                 /* Batch */
//     int64_t C_t = input.size(1);                                 /* Channel */
//     int64_t Win_t = input.size(2);                               /* input width */

//     int64_t F_t = weight.size(0);                                /* Number of filters */
//     int64_t WW_t = weight.size(2);                               /* filter width */

//     int64_t dial = dilation;                                     /* dilation parameter */
//     int64_t pad_size = ((WW_t- 1))*dial;                         /* Total padding size */
//     int64_t W_t = Win_t - pad_size;                              /* output width */

//     auto Y = input.new_empty({N_t,F_t,W_t});                     /* New tensor for output */

//     float* input_a = input.data_ptr<float>();                    /* Get pointers for accessing the tensors */
//     float* weight_a = weight.data_ptr<float>();
//     float* Y_a = Y.data_ptr<float>();

//     auto flip_weight = weight.new_empty({WW_t,F_t,C_t});        /* Array to store permuted weight tensor (width, filters, channels) */
//     float* flip_weight_a = flip_weight.data_ptr<float>();

//     /* jited tranpose to permute the array dimensions
//         Overall convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)*/

//     libxsmm_blasint per_m = WW_t;
//     libxsmm_blasint per_n = F_t*C_t;
//     libxsmm_blasint per_ldi = WW_t;
//     libxsmm_blasint per_ldo = F_t*C_t;

//     libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( per_m, per_n, per_ldi, per_ldo, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_permute_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_permute_kernel == NULL) {
//         fprintf( stderr, "JIT unary TPP for trans_permute_kernel (normal to normal transform) isn't working in the forward pass. Bailing...!\n");
//         exit(-1);
//     }
//     libxsmm_meltw_unary_param trans_permute_param;
//     trans_permute_param.in.primary  = weight_a;
//     trans_permute_param.out.primary = flip_weight_a;
//     trans_permute_kernel( &trans_permute_param);

//     int lda = C_t;                                              /* Input channels (15) */
//     int ldb = Win_t;                                            /* Input width (60400) */
//     int ldc = W_t;                                              /* Output width (60000)*/
//     unsigned long long l_br = WW_t;

//     int tile_multiple = (W_t/XS_TILE_FORWARD)*XS_TILE_FORWARD;

//     /* Dispatch SGEMM kernels for the normal case and the edge case*/
//     /* setting update GEMM struct */
//     libxsmm_gemm_flags l_flags;
//     libxsmm_gemm_prefetch_type l_prefetch;
//     libxsmm_gemm_shape l_shape;
//     libxsmm_gemm_batch_reduce_config l_brconfig;

//     l_flags = LIBXSMM_GEMM_FLAG_NONE;
//     l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

//     l_shape = libxsmm_create_gemm_shape(XS_TILE_FORWARD, F_t, C_t, ldb, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
//     l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
//     l_brconfig.br_stride_a_hint = dial*sizeof(float);
//     l_brconfig.br_stride_b_hint = F_t*C_t*sizeof(float);
//     libxsmm_gemmfunction brgemm_kernel_main = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

//     l_shape = libxsmm_create_gemm_shape(W_t - tile_multiple, F_t, C_t, ldb, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
//     l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
//     l_brconfig.br_stride_a_hint = dial*sizeof(float);
//     l_brconfig.br_stride_b_hint = F_t*C_t*sizeof(float);
//     libxsmm_gemmfunction brgemm_kernel_edge = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

//     /* JIT eltwise TPPs for initialization... */
//     libxsmm_blasint tpp_m1 = XS_TILE_FORWARD;                      /* columns */
//     libxsmm_blasint tpp_m2 = W_t - tile_multiple;                  /* columns */
//     libxsmm_blasint tpp_n = F_t;                                   /* rows */
//     libxsmm_blasint ld_zero = W_t;

//     libxsmm_meltw_unary_type unary_type;
//     unary_type = LIBXSMM_MELTW_TYPE_UNARY_XOR;
//     libxsmm_meltw_unary_flags unary_flags;
//     unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary zero_kernel_main = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );
//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, tpp_n, tpp_m2, ld_zero, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary zero_kernel_edge = libxsmm_dispatch_meltw_unary_v2( unary_type, unary_shape, unary_flags );

//     if ((zero_kernel_main == NULL) || (zero_kernel_edge == NULL)) {
//         fprintf( stderr, "JIT UNARY kernel for initilizing zeros failed in forward pass. Bailing...!\n");
//         exit(-1);
//     }

//     /* Main compute loop */
//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++) {                               /* Loop for batches */
//         int last_block = 0;
//         libxsmm_meltw_unary_param zero_param_main, zero_param_edge;
//         libxsmm_gemm_param gemm_param_main, gemm_param_edge;

//         for(int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb += XS_TILE_FORWARD) {    /* width blocking loop (Main case) */

//             zero_param_main.out.primary = &Y_a[n*F_t*W_t + wb];       /* Initialization */
//             zero_kernel_main( &zero_param_main );

//             gemm_param_main.a.primary = &input_a[n*C_t*Win_t + 0*Win_t + wb];
//             gemm_param_main.b.primary = &flip_weight_a[0];
//             gemm_param_main.c.primary = &Y_a[n*F_t*W_t + 0*W_t + wb];
//             gemm_param_main.op.tertiary = &l_br;
//             brgemm_kernel_main( &gemm_param_main );

//             last_block = wb;
//         }

//         if (W_t % XS_TILE_FORWARD != 0){                        /* Edge Case */

//             zero_param_edge.out.primary = &Y_a[n*F_t*W_t + last_block + XS_TILE_FORWARD];     /* Initialization */
//             zero_kernel_edge( &zero_param_edge );

//             gemm_param_edge.a.primary = &input_a[n*C_t*Win_t + 0*Win_t + last_block + XS_TILE_FORWARD];
//             gemm_param_edge.b.primary = &flip_weight_a[0];
//             gemm_param_edge.c.primary = &Y_a[n*F_t*W_t + 0*W_t + last_block + XS_TILE_FORWARD];
//             gemm_param_edge.op.tertiary = &l_br;
//             brgemm_kernel_edge( &gemm_param_edge );
//         }
//     }

//     return Y;           /* Return output array */
// }

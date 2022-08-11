

    RECORD_FUNCTION("Conv1dOpti_backward_bf16", std::vector<c10::IValue>({grad, input, weight}));        // For recording time

    int64_t N_t = input.size(0);                    /* Batch */
    int64_t C_t = input.size(1);                    /* Channel */
    int64_t Win_t = input.size(2);                  /* input width */

    int64_t F_t = weight.size(0);                   /* Number of filters */
    int64_t WW_t = weight.size(2);                  /* filter width */

    int64_t dial = dilation;                        /* dilation parameter */
    int64_t pad_size = ((WW_t- 1))*dial;            /* Total padding size */
    int64_t W_t = Win_t - pad_size;                 /* output width */

    auto d_input = input.new_empty({N_t,C_t,Win_t});            /* declare data gradiant tensor */
    auto d_weight = weight.new_empty({F_t,C_t,WW_t});           /* declare weight gradiant tensor */

    DECL_VLA_PTR_PT(T, input_a, [C_t][Win_t], input);
    DECL_VLA_PTR_PT(T, grad_a, [F_t][W_t], grad);
    DECL_VLA_PTR_PT(T, d_input_a, [C_t][Win_t], d_input);
    // DECL_VLA_PTR_PT(T, d_weight_a, [], d_weight);
    // DECL_VLA_PTR_PT(T, weight_a, [], weight);
    T* d_weight_a = d_weight.data_ptr<T>();
    T* weight_a = weight.data_ptr<T>();


    /* Backward Data part of the code */

    auto flip_weight_tensor = weight.new_empty({WW_t,C_t,F_t});                             /* Weight tensor with permuted dimension (width, channels, filters) */
    // DECL_VLA_PTR_PT(T, flip_weight_a, [], flip_weight_tensor);                                 /* Get pointer */
    T* flip_weight_a = flip_weight_tensor.data_ptr<T>();


    auto weight_buffer = weight.new_empty({F_t,C_t,WW_t});                  /* Tensor weight buffer */
    // DECL_VLA_PTR_PT(T, weight_buffer_a, [], weight_buffer);                 /* Get pointer */
    T* weight_buffer_a = weight_buffer.data_ptr<T>();

    #pragma omp parallel for
    for(int i = 0; i < F_t*C_t; i++){
        for(int kw = 0; kw < WW_t; kw++){                                   /* reverse copy */
            flip_weight_a[i*WW_t + kw] = weight_a[i*WW_t + WW_t - kw - 1];
        }
    }

    /* jited tranpose to permute the array dimensions
        Overall convert (F_t, C_t, WW_t) -----> (WW_t, C_t, F_t)*/
    int flip_m1 = WW_t;
    int flip_n1 = F_t*C_t;

    auto trans_unary_flip_1 = SCOPEIT(XformExtTPP<T>(flip_n1, flip_m1, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    trans_unary_flip_1(&flip_weight_a[0], &weight_buffer_a[0]);

    int flip_m2 = C_t;
    int flip_n2 = F_t;

    auto trans_unary_flip_2 = SCOPEIT(XformExtTPP<T>(flip_n2, flip_m2, XformTPP::XFORM_XPOSE_TPP), XPOSE);

    /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
    #pragma omp parallel for
    for(int kw = 0; kw < WW_t; kw++){                                     /* permute last two dimensions */
        trans_unary_flip_2(&weight_buffer_a[kw*C_t*F_t], &flip_weight_a[kw*C_t*F_t]);
    }

    int64_t Wpad_t = W_t + 2*(WW_t - 1)*dial;                             /* For padding gradiant on both sides */
    int64_t tile_multiple = (Win_t/XS_TILE_DBACKWARD)*XS_TILE_DBACKWARD;  /* Number of blocks/tiles in Input */

    int lda = F_t;                                                        /* Number of Filters (16) */
    int ldc = Win_t;                                                      /* Input width (60400) */
    unsigned long long l_br = WW_t;                                       /* Number of batches in brGEMM (= width of kernel = 51) */

    int pad_tile_multiple = 2 * (((WW_t - 1)*dial)/XS_TILE_DBACKWARD + 1) * XS_TILE_DBACKWARD;       /* Padded block/tile (896) */
    auto grad_shortpad_tensor = grad.new_empty({N_t,F_t,2*pad_tile_multiple});
    DECL_VLA_PTR_PT(T, grad_a_shortpad, [F_t][2*pad_tile_multiple], grad_shortpad_tensor);                                          /* Get pointer */

    int ldb_shortpad = 2*pad_tile_multiple;                               /* grad padded short buffer (1792) */

    int short_width = ((XS_TILE_DBACKWARD + (WW_t-1)*dial)/XS_TILE_DBACKWARD + 1)*XS_TILE_DBACKWARD;    /* Width of buffer   (512) */
    auto grad_shortvnni_backdata = grad.new_empty({N_t,F_t,short_width});                                 /* Buffer for storing VNNI transform */
    DECL_VLA_PTR_PT(T, grad_a_shortvnni, [F_t][short_width], grad_shortvnni_backdata);                                      /* Get pointer */

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    auto backdata_shortkernel_main = SCOPEITGEMM((BrgemmTPP<T,T>(C_t, XS_TILE_DBACKWARD, F_t, C_t*F_t, 2*dial, lda, short_width, ldc, 1.0, 0, 1)));
    auto backdata_shortkernel_edge = SCOPEITGEMM((BrgemmTPP<T,T>(C_t, (Win_t - tile_multiple), F_t, C_t*F_t, 2*dial, lda, short_width, ldc, 1.0, 0, 1)));

    /* Virtual copy kernels */
    int virtual_m1 = pad_tile_multiple - ((WW_t - 1)*dial);     /* columns */
    int virtual_m2 = ((WW_t - 1)*dial);                         /* columns */
    int virtual_n = F_t;                                        /* rows */
    int ldi_virtual = W_t;
    int ldo_virtual = 2*pad_tile_multiple;

    if (ldi_virtual < virtual_m1){                                          /* corner case when width's are very small */
        virtual_m1 = ldi_virtual;
        auto all_zero_backdata = SCOPEIT(SetZeroTPP<T>(virtual_n, ldo_virtual, ldo_virtual), EW_ZERO);
        #pragma omp parallel for
        for(int n = 0; n < N_t; n++){
            all_zero_backdata(&grad_a_shortpad[n][0][0]);
        }
    }

    auto virtual_copy = SCOPEIT(CpyTPP<T>(virtual_n, virtual_m1, ldi_virtual, ldo_virtual), EW_COPY);
    auto virtual_copy_zero = SCOPEIT(SetZeroTPP<T>(virtual_n, virtual_m2, ldo_virtual), EW_ZERO);

    {
    RECORD_SCOPE(virtul_copy_loop, {grad_a, grad_a_shortpad});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for
            for(int n = 0; n < N_t; n++){                         /* Loops for storing the edge portion of gradinant array into grad_a_shortpad */

                virtual_copy_zero(&grad_a_shortpad[n][0][0]);

                virtual_copy(&grad_a[n][0][0], &grad_a_shortpad[n][0][(WW_t - 1)*dial]);
                virtual_copy(&grad_a[n][0][W_t - virtual_m1], &grad_a_shortpad[n][0][ldo_virtual - virtual_m1 - ((WW_t - 1)*dial)]);

                virtual_copy_zero(&grad_a_shortpad[n][0][ldo_virtual - ((WW_t - 1)*dial)]);
            }
        }
    }

    /* JIT eltwise TPPs for initialization ... */
    int tpp_m1 = XS_TILE_DBACKWARD;                      /* columns */
    int tpp_m2 = Win_t - tile_multiple;                  /* columns */
    int tpp_n = C_t;                                     /* rows */
    int ld_zero = Win_t;

    auto copy_kernel_main = SCOPEIT(SetZeroTPP<T>(tpp_n, tpp_m1, ld_zero), EW_ZERO);
    auto copy_kernel_edge = SCOPEIT(SetZeroTPP<T>(tpp_n, tpp_m2, ld_zero), EW_ZERO);


    /* use jited VNNI */
    int ldi_1 = W_t;
    int ldi_2 = ldb_shortpad;                            /* (1792) */
    int ldo = short_width;                               /* (512) */

    tpp_m1 = (XS_TILE_DBACKWARD + dial*(WW_t-1));
    tpp_m2 = (XS_TILE_DBACKWARD + dial*(WW_t-1));

    auto trans_shortvnni_kernel_1 = SCOPEIT(XformExtTPP<T>(F_t, tpp_m1, F_t, tpp_m1, ldi_1, ldo, XformTPP::XFORM_N2V_TPP), VNNI);
    auto trans_shortvnni_kernel_2 = SCOPEIT(XformExtTPP<T>(F_t, tpp_m2, F_t, tpp_m2, ldi_2, ldo, XformTPP::XFORM_N2V_TPP), VNNI);

    /* Main backward data pass loop */
    {
    RECORD_SCOPE(backward_data_loop_bf16, {grad_a, grad_a_shortpad});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for
            for(int n = 0; n < N_t; n++) {
                int last_block=0;

                for(int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

                    copy_kernel_main(&d_input_a[n][0][wb]);
                    
                    if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD){
                        /* Normal case (Take VNNI transform of a portion of grad_a array ) */

                        /* VNNI transform */
                        trans_shortvnni_kernel_1(&grad_a[n][0][wb - (WW_t-1)*dial], &grad_a_shortvnni[n][0][0]);

                        /* brGEMM */
                        backdata_shortkernel_main(&flip_weight_a[0], &grad_a_shortvnni[n][0][0], &d_input_a[n][0][wb], l_br);
                    }
                    else if (wb < (WW_t-1)*dial){
                        /* Right side case (Take VNNI transform of grad_a_shortpad array) */

                        /* VNNI transform */
                        trans_shortvnni_kernel_2(&grad_a_shortpad[n][0][wb], &grad_a_shortvnni[n][0][0]);

                        /* brGEMM */
                        backdata_shortkernel_main(&flip_weight_a[0], &grad_a_shortvnni[n][0][0], &d_input_a[n][0][wb], l_br); 
                    }
                    else{
                        /* Left side case (Take VNNI transform of grad_a_shortpad array) */

                        /* VNNI transform */
                        trans_shortvnni_kernel_2(&grad_a_shortpad[n][0][wb - Wpad_t + 2*pad_tile_multiple], &grad_a_shortvnni[n][0][0]);

                        /* brGEMM */
                        backdata_shortkernel_main(&flip_weight_a[0], &grad_a_shortvnni[n][0][0], &d_input_a[n][0][wb], l_br);
                    }
                    last_block = wb;
                }

                if (Win_t % XS_TILE_DBACKWARD != 0){                                /* Edge case */

                    /* Right side case (Take VNNI transform of grad_a_shortpad array) */
                    copy_kernel_edge(&d_input_a[n][0][last_block + XS_TILE_DBACKWARD]);

                    /* VNNI transform */
                    trans_shortvnni_kernel_2(&grad_a_shortpad[n][0][last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple], &grad_a_shortvnni[n][0][0]);

                    /* brGEMM */
                    backdata_shortkernel_edge(&flip_weight_a[0], &grad_a_shortvnni[n][0][0], &d_input_a[n][0][last_block + XS_TILE_DBACKWARD], l_br);
                }
            }
        }
    }



    /* -------------------------------  Backward Weight part of the code ---------------------------------- */


    // float* flip_d_weight_a = (float*) libxsmm_aligned_malloc( F_t*C_t*WW_t*sizeof(float), 64 );             /* Array for permuted weight gradiant */
    auto flip_d_weight = weight.new_empty({F_t,C_t,WW_t}, at::kFloat);
    float* flip_d_weight_a = flip_d_weight.data_ptr<float>();

    for(int w = 0; w < F_t*C_t*WW_t; w++){          /* Initialize array */
        flip_d_weight_a[w] = 0.0f;
    }

    /*  lda = W_t;                                  // Already defined variables */
    /* ldb = Win_t; */
    /* ldc = C_t; */
    l_br = WW_t;                                    /* Number of batches in brGEMM (= width of kernel = 51) */
    tile_multiple = (W_t/XS_TILE_WBACKWARD)*XS_TILE_WBACKWARD;


    /* Blocking on grad_a */
    int lda_g = Win_t;
    int ldb_trans_g = F_t;
    int ldc_g = F_t;

    int M_g = W_t;
    int N_g = F_t;
    int short_W_t = XS_TILE_WBACKWARD;
    int edge_W_t = W_t - tile_multiple;

    auto grad_shortvnni_backweight = grad.new_empty({N_t,F_t,short_W_t});                            /* Short buffer for storing VNNI transform */
    DECL_VLA_PTR_PT(T, grad_shortvnni, [F_t][short_W_t], grad_shortvnni_backweight);   

    auto grad_edgevnni_backweight = grad.new_empty({N_t,F_t,edge_W_t});                              /* Short buffer for storing VNNI transform in edge case */
    DECL_VLA_PTR_PT(T, grad_edgevnni, [F_t][edge_W_t], grad_edgevnni_backweight); 

    /* use jited tranpose */
    auto trans_shortkernel_grad = SCOPEIT(XformExtTPP<T>(N_g, short_W_t, short_W_t, N_g, M_g, N_g, XformTPP::XFORM_XPOSE_N2V_TPP), XPOSE);
    auto trans_edgekernel_grad = SCOPEIT(XformExtTPP<T>(N_g, edge_W_t, edge_W_t, N_g, M_g, N_g, XformTPP::XFORM_XPOSE_N2V_TPP), XPOSE);

    /* Dispatch brGEMM kernels for the normal case and the edge case*/
    auto backweight_kernel_main = SCOPEITGEMM((BrgemmTPP<T, float>(C_t, F_t, XS_TILE_WBACKWARD, 0, 0, lda_g, ldb_trans_g, ldc_g, 1.0, 0, 1)));
    auto backweight_kernel_edge = SCOPEITGEMM((BrgemmTPP<T, float>(C_t, F_t, (W_t - tile_multiple), 0, 0, lda_g, ldb_trans_g, ldc_g, 1.0, 0, 1)));

    /* Main compute loop for backward weight pass */
    {
    RECORD_SCOPE(backward_weight_loop_bf16, {grad_a, grad_a_shortpad});
        {
            RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
            #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])
            for(int n = 0; n < N_t; n++) {
                int last_block = 0;

                for(int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {            /* Main Case */

                    /* Take transpose assumping FP32 (This will do both transpose and VNNI transform for BF16) */
                    trans_shortkernel_grad(&grad_a[n][0][wb], &grad_shortvnni[n][0][0]);

                    for(int kw = 0; kw < WW_t; kw++) {
                        backweight_kernel_main(&input_a[n][0][wb + kw*dial], &grad_shortvnni[n][0][0], &flip_d_weight_a[kw*C_t*F_t], 1);
                    }
                    last_block = wb;
                }

                if (W_t % XS_TILE_WBACKWARD != 0){              /* Edge Case */

                    trans_edgekernel_grad(&grad_a[n][0][last_block + XS_TILE_WBACKWARD], &grad_edgevnni[n][0][0]);

                    for(int kw = 0; kw < WW_t; kw++) {

                        backweight_kernel_edge(&input_a[n][0][last_block + XS_TILE_WBACKWARD + kw*dial], &grad_edgevnni[n][0][0], &flip_d_weight_a[kw*C_t*F_t], 1);
                    }
                }
            }
        }
    }


    auto flip_d_weight_tensor = weight.new_empty({WW_t,C_t,F_t});
    // DECL_VLA_PTR_PT(T, flip_d_weight_bf16, [], flip_d_weight_tensor); 
    T* flip_d_weight_bf16 = flip_d_weight_tensor.data_ptr<T>();


    /* JIT eltwise TPPs for FP32 to BF16 conversion... */
    int cvt_m = 1;
    int cvt_n = F_t*C_t*WW_t;

    // auto eltwise_kernel = SCOPEIT(ConvertTPP<float, T>(cvt_n, cvt_m, cvt_m, cvt_m), EW_ZERO);
    // eltwise_kernel(flip_d_weight_a, flip_d_weight_bf16);
    libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( cvt_m, cvt_n, cvt_m, cvt_m, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
    libxsmm_meltwfunction_unary eltwise_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    if ( eltwise_kernel == NULL ) {
        fprintf( stderr, "JIT unary TPP to convert FP32 to BF16 failed. Bailing...!\n");
        exit(-1);
    }

    libxsmm_meltw_unary_param eltwise_params;
    eltwise_params.in.primary = flip_d_weight_a;
    eltwise_params.out.primary = flip_d_weight_bf16;
    eltwise_kernel(&eltwise_params);


    /* jited tranpose to permute the array dimensions
        Overall Convert (WW_t, C_t, F_t) -----> (F_t, C_t, WW_t)*/
    int per_m1 = F_t;
    int per_n1 = C_t;

    auto trans_permute_1 = SCOPEIT(XformExtTPP<T>(per_n1, per_m1, XformTPP::XFORM_XPOSE_TPP), XPOSE);

    /* Convert (WW_t, C_t, F_t) -----> (WW_t, F_t, C_t) */
    #pragma omp parallel for
    for(int kw = 0; kw < WW_t; kw++){                   /* permute last two dimensions */

        trans_permute_1(&flip_d_weight_bf16[kw*C_t*F_t], &flip_weight_a[kw*C_t*F_t]);
    }

    int per_m2 = F_t*C_t;
    int per_n2 = WW_t;
    
    auto trans_permute_2 = SCOPEIT(XformExtTPP<T>(per_n2, per_m2, XformTPP::XFORM_XPOSE_TPP), XPOSE);
    /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
    trans_permute_2(&flip_weight_a[0], &d_weight_a[0]);

    return {d_input, d_weight};


// std::tuple<at::Tensor, at::Tensor> Conv1dOpti_backward_bf16_libxsmm(at::Tensor& grad, at::Tensor& input, at::Tensor& weight, int dilation){

//     /* RECORD_FUNCTION("Conv1dOpti_backward_bf16", std::vector<c10::IValue>({grad, input, weight}));        // For recording time */

//     int64_t N_t = input.size(0);                    /* Batch */
//     int64_t C_t = input.size(1);                    /* Channel */
//     int64_t Win_t = input.size(2);                  /* input width */

//     int64_t F_t = weight.size(0);                   /* Number of filters */
//     int64_t WW_t = weight.size(2);                  /* filter width */

//     int64_t dial = dilation;                        /* dilation parameter */
//     int64_t pad_size = ((WW_t- 1))*dial;            /* Total padding size */
//     int64_t W_t = Win_t - pad_size;                 /* output width */

//     auto d_input = input.new_empty({N_t,C_t,Win_t});            /* declare data gradiant tensor */
//     auto d_weight = weight.new_empty({F_t,C_t,WW_t});           /* declare weight gradiant tensor */

//     libxsmm_bfloat16* input_a = (libxsmm_bfloat16*) input.data_ptr<at::BFloat16>();         /* Get BFloat16 data pointers for accessing tensors */
//     libxsmm_bfloat16* weight_a = (libxsmm_bfloat16*) weight.data_ptr<at::BFloat16>();
//     libxsmm_bfloat16* grad_a = (libxsmm_bfloat16*) grad.data_ptr<at::BFloat16>();
//     libxsmm_bfloat16* d_input_a = (libxsmm_bfloat16*) d_input.data_ptr<at::BFloat16>();
//     libxsmm_bfloat16* d_weight_a = (libxsmm_bfloat16*) d_weight.data_ptr<at::BFloat16>();

//     /* Backward Data part of the code */

//     auto flip_weight_tensor = weight.new_empty({WW_t,C_t,F_t});                             /* Weight tensor with permuted dimension (width, channels, filters) */
//     libxsmm_bfloat16* flip_weight_a = (libxsmm_bfloat16*) flip_weight_tensor.data_ptr<at::BFloat16>();   /* Get pointer */


//     auto weight_buffer = weight.new_empty({F_t,C_t,WW_t});                  /* Tensor weight buffer */
//     libxsmm_bfloat16* weight_buffer_a = (libxsmm_bfloat16*) weight_buffer.data_ptr<at::BFloat16>();

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

//     libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape( flip_m1, flip_n1, flip_ldi_1, flip_ldo_1, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary trans_flip_1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_flip_1 == NULL) {
//         fprintf( stderr, "JIT unary TPP for trans_flip_1 (NORM_TO_NORM Transform) in backward data failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t) */
//     libxsmm_meltw_unary_param trans_param_flip_1;
//     trans_param_flip_1.in.primary  = flip_weight_a;
//     trans_param_flip_1.out.primary = weight_buffer_a;
//     trans_flip_1( &trans_param_flip_1 );

//     libxsmm_blasint flip_m2 = C_t;
//     libxsmm_blasint flip_n2 = F_t;
//     libxsmm_blasint flip_ldi_2 = C_t;
//     libxsmm_blasint flip_ldo_2 = F_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( flip_m2, flip_n2, flip_ldi_2, flip_ldo_2, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary trans_flip_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_flip_2 == NULL) {
//         fprintf( stderr, "JIT unary TPP for trans_flip_2 (NORM_TO_NORM Transform) in backward data failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
//     #pragma omp parallel for
//     for(int kw = 0; kw < WW_t; kw++){                                     /* permute last two dimensions */
//         libxsmm_meltw_unary_param trans_param_flip_2;
//         trans_param_flip_2.in.primary  = &weight_buffer_a[kw*C_t*F_t];
//         trans_param_flip_2.out.primary = &flip_weight_a[kw*C_t*F_t];
//         trans_flip_2( &trans_param_flip_2 );
//     }

//     int64_t Wpad_t = W_t + 2*(WW_t - 1)*dial;                             /* For padding gradiant on both sides */
//     int64_t tile_multiple = (Win_t/XS_TILE_DBACKWARD)*XS_TILE_DBACKWARD;  /* Number of blocks/tiles in Input */

//     int lda = F_t;                                                        /* Number of Filters (16) */
//     /* int ldb_orig = W_t;                                                //    Output width (60000) */
//     /* int ldb = Wpad_t;                                                  //    Extra padded grad input case 60800 */
//     int ldc = Win_t;                                                      /* Input width (60400) */
//     unsigned long long l_br = WW_t;                                       /* Number of batches in brGEMM (= width of kernel = 51) */

//     int pad_tile_multiple = 2 * (((WW_t - 1)*dial)/XS_TILE_DBACKWARD + 1) * XS_TILE_DBACKWARD;       /* Padded block/tile (896) */
//     int ldb_shortpad = 2*pad_tile_multiple;                               /* grad padded short buffer (1792) */

//     auto grad_shortpad_tensor = grad.new_empty({N_t,F_t,2*pad_tile_multiple});
//     libxsmm_bfloat16* grad_a_shortpad = (libxsmm_bfloat16*) grad_shortpad_tensor.data_ptr<at::BFloat16>();   /* short buffer for padded gradiant */

//     /* Virtual copy kernels */
//     libxsmm_blasint virtual_m1 = pad_tile_multiple - ((WW_t - 1)*dial);     /* columns */
//     libxsmm_blasint virtual_m2 = ((WW_t - 1)*dial);                         /* columns */
//     libxsmm_blasint virtual_n = F_t;                                        /* rows */
//     libxsmm_blasint ldi_virtual = W_t;
//     libxsmm_blasint ldo_virtual = 2*pad_tile_multiple;

//     if (ldi_virtual < virtual_m1){                                          /* corner case when width's are very small */
//         virtual_m1 = ldi_virtual;
//         unary_shape = libxsmm_create_meltw_unary_shape( ldo_virtual, virtual_n, ldo_virtual, ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//         libxsmm_meltwfunction_unary all_zero_backdata_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//         if ( all_zero_backdata_bf16 == NULL) {
//             fprintf( stderr, "JIT unary TPP for all_zero_backdata_bf16 kernel in backward data pass failed. Bailing...!\n");
//             exit(-1);
//         }
//         #pragma omp parallel for
//         for(int n = 0; n < N_t; n++){
//             libxsmm_meltw_unary_param all_zero_params;
//             all_zero_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];     /* Initialize the entire array when widths are small */
//             all_zero_backdata_bf16(&all_zero_params);
//         }
//     }

//     unary_shape = libxsmm_create_meltw_unary_shape( virtual_m1, virtual_n, ldi_virtual, ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary virtual_copy_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     unary_shape = libxsmm_create_meltw_unary_shape( virtual_m2, virtual_n, virtual_m2, ldo_virtual, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary virtual_copy_zero_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

//     if ((virtual_copy_bf16 == NULL) || (virtual_copy_zero_bf16 == NULL)) {
//         fprintf( stderr, "JIT unary TPP for virtual_copy_bf16 kernel in backward data pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++){                         /* Loops for storing the edge portion of gradinant array into grad_a_shortpad */

//         libxsmm_meltw_unary_param vcopy_params;           /* Copy parameter variable for holding the pointer */
//         libxsmm_meltw_unary_param vcopy_params_zero;

//         vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual];                                        /* copy zeros */
//         virtual_copy_zero_bf16(&vcopy_params_zero);

//         vcopy_params.in.primary = &grad_a[n*F_t*W_t];                                                               /* copy after zeros from start of the grad array */
//         vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ((WW_t - 1)*dial)];
//         virtual_copy_bf16(&vcopy_params);

//         vcopy_params.in.primary = &grad_a[n*F_t*W_t + W_t - virtual_m1];                                            /* copy from the end of the grad array */
//         vcopy_params.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - virtual_m1 - ((WW_t - 1)*dial)];
//         virtual_copy_bf16(&vcopy_params);

//         vcopy_params_zero.out.primary = &grad_a_shortpad[n*F_t*ldo_virtual + ldo_virtual - ((WW_t - 1)*dial)];     /* copy zeros */
//         virtual_copy_zero_bf16(&vcopy_params_zero);
//     }

// /*
// #else
//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++){                   // loop to store the edges for gradiant array into grad_a_shortpad buffer
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

//     int short_width = ((XS_TILE_DBACKWARD + (WW_t-1)*dial)/XS_TILE_DBACKWARD + 1)*XS_TILE_DBACKWARD;    /* Width of buffer   (512) */

//     auto grad_shortvnni_backdata = grad.new_empty({N_t,F_t,short_width});                                 /* Buffer for storing VNNI transform */
//     libxsmm_bfloat16* grad_a_shortvnni = (libxsmm_bfloat16*) grad_shortvnni_backdata.data_ptr<at::BFloat16>();

//     /* Dispatch brGEMM kernels for the normal case and the edge case*/
//     libxsmm_gemm_flags l_flags;
//     libxsmm_gemm_prefetch_type l_prefetch;
//     libxsmm_gemm_shape l_shape;
//     libxsmm_gemm_batch_reduce_config l_brconfig;

//     l_flags = LIBXSMM_GEMM_FLAG_VNNI_A;
//     l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

//     l_shape = libxsmm_create_gemm_shape(XS_TILE_DBACKWARD, C_t, F_t, short_width, lda, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
//     l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
//     l_brconfig.br_stride_a_hint = 2*dial*sizeof(libxsmm_bfloat16);
//     l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(libxsmm_bfloat16);
//     libxsmm_gemmfunction backdata_shortkernel_main = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

//     l_shape = libxsmm_create_gemm_shape(Win_t - tile_multiple, C_t, F_t, short_width, lda, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
//     l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
//     l_brconfig.br_stride_a_hint = 2*dial*sizeof(libxsmm_bfloat16);
//     l_brconfig.br_stride_b_hint = C_t*F_t*sizeof(libxsmm_bfloat16);
//     libxsmm_gemmfunction backdata_shortkernel_edge = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch, l_brconfig);

//     if ((backdata_shortkernel_main == NULL) | (backdata_shortkernel_edge == NULL)) {
//         fprintf( stderr, "JIT for backdata_shortkernel_main in backward data pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* JIT eltwise TPPs for initialization ... */
//     libxsmm_blasint tpp_m1 = XS_TILE_DBACKWARD;                      /* columns */
//     libxsmm_blasint tpp_m2 = Win_t - tile_multiple;                  /* columns */
//     libxsmm_blasint tpp_n = C_t;                                     /* rows */
//     libxsmm_blasint ld_zero = Win_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, tpp_n, tpp_m1, ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary copy_kernel_main_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, tpp_n, tpp_m2, ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary copy_kernel_edge_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

//     if ((copy_kernel_main_bf16 == NULL) || (copy_kernel_edge_bf16  == NULL)) {
//         fprintf( stderr, "JIT for copy_kernel_main_bf16 in backward data pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* use jited VNNI */
//     libxsmm_blasint ldi_1 = W_t;
//     libxsmm_blasint ldi_2 = ldb_shortpad;                            /* (1792) */
//     libxsmm_blasint ldo = short_width;                               /* (512) */

//     libxsmm_meltw_unary_type trans_vnni_type;
//     if ( F_t % 2 == 1 ) {
//         trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD;
//     } else {
//         trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
//     }

//     tpp_m1 = (XS_TILE_DBACKWARD + dial*(WW_t-1));
//     tpp_m2 = (XS_TILE_DBACKWARD + dial*(WW_t-1));

//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, F_t, ldi_1, ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary trans_shortvnni_kernel_1 = libxsmm_dispatch_meltw_unary_v2( trans_vnni_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, F_t, ldi_2, ldo, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary trans_shortvnni_kernel_2 = libxsmm_dispatch_meltw_unary_v2( trans_vnni_type, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

//     if ((trans_shortvnni_kernel_1 == NULL) || (trans_shortvnni_kernel_2 == NULL)) {
//         fprintf( stderr, "JIT unary TPP for trans_shortvnni_kernel_1 (NORM_TO_VNN transfor) in backward data pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Main backward data pass loop */
//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++) {
//         int last_block=0;

//         libxsmm_meltw_unary_param copy_params_main, copy_params_edge;                 /* Copy parameter variable for holding the pointer */
//         libxsmm_meltw_unary_param trans_param_1, trans_param_2;
//         libxsmm_gemm_param gemm_param_main, gemm_param_edge;

//         for(int wb = 0; wb < Win_t - XS_TILE_DBACKWARD + 1; wb += XS_TILE_DBACKWARD) {

//             copy_params_main.out.primary = &d_input_a[n*C_t*Win_t + wb];             /* Initialization */
//             copy_kernel_main_bf16(&copy_params_main);

//             if (wb >= (WW_t-1)*dial && wb < Win_t - (WW_t-1)*dial - XS_TILE_DBACKWARD){
//                 /* Normal case (Take VNNI transform of a portion of grad_a array ) */

//                 /* VNNI transform */
//                 trans_param_1.in.primary  = &grad_a[n*F_t*W_t + 0*W_t + wb - (WW_t-1)*dial];
//                 trans_param_1.out.primary = &grad_a_shortvnni[n*F_t*short_width];
//                 trans_shortvnni_kernel_1( &trans_param_1 );

//                 /* brGEMM */
//                 gemm_param_main.a.primary = &grad_a_shortvnni[n*F_t*short_width];
//                 gemm_param_main.b.primary = &flip_weight_a[0];
//                 gemm_param_main.c.primary = &d_input_a[n*C_t*Win_t + 0*Win_t + wb];
//                 gemm_param_main.op.tertiary = &l_br;
//                 backdata_shortkernel_main( &gemm_param_main );
//             }
//             else if (wb < (WW_t-1)*dial){
//                 /* Right side case (Take VNNI transform of grad_a_shortpad array) */

//                 /* VNNI transform */
//                 trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb];
//                 trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
//                 trans_shortvnni_kernel_2( &trans_param_2 );

//                 /* brGEMM */
//                 gemm_param_main.a.primary = &grad_a_shortvnni[n*F_t*short_width];
//                 gemm_param_main.b.primary = &flip_weight_a[0];
//                 gemm_param_main.c.primary = &d_input_a[n*C_t*Win_t + 0*Win_t + wb];
//                 gemm_param_main.op.tertiary = &l_br;
//                 backdata_shortkernel_main( &gemm_param_main );
//             }
//             else{
//                 /* Left side case (Take VNNI transform of grad_a_shortpad array) */

//                 /* VNNI transform */
//                 trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + wb - Wpad_t + 2*pad_tile_multiple];
//                 trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
//                 trans_shortvnni_kernel_2( &trans_param_2 );

//                 /* brGEMM */
//                 gemm_param_main.a.primary = &grad_a_shortvnni[n*F_t*short_width];
//                 gemm_param_main.b.primary = &flip_weight_a[0];
//                 gemm_param_main.c.primary = &d_input_a[n*C_t*Win_t + 0*Win_t + wb];
//                 gemm_param_main.op.tertiary = &l_br;
//                 backdata_shortkernel_main( &gemm_param_main );
//             }
//             last_block = wb;
//         }

//         if (Win_t % XS_TILE_DBACKWARD != 0){                                /* Edge case */

//             /* Right side case (Take VNNI transform of grad_a_shortpad array) */

//             copy_params_edge.out.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];             /* Initialization */
//             copy_kernel_edge_bf16(&copy_params_edge);

//             /* VNNI transform */
//             trans_param_2.in.primary  = &grad_a_shortpad[n*F_t*2*pad_tile_multiple + last_block + XS_TILE_DBACKWARD - Wpad_t + 2*pad_tile_multiple];
//             trans_param_2.out.primary = &grad_a_shortvnni[n*F_t*short_width];
//             trans_shortvnni_kernel_2( &trans_param_2 );

//             /* brGEMM */
//             gemm_param_edge.a.primary = &grad_a_shortvnni[n*F_t*short_width];
//             gemm_param_edge.b.primary = &flip_weight_a[0];
//             gemm_param_edge.c.primary = &d_input_a[n*C_t*Win_t + last_block + XS_TILE_DBACKWARD];
//             gemm_param_edge.op.tertiary = &l_br;
//             backdata_shortkernel_edge( &gemm_param_edge );
//         }
//     }



//     /* -------------------------------  Backward Weight part of the code ---------------------------------- */


//     float* flip_d_weight_a = (float*) libxsmm_aligned_malloc( F_t*C_t*WW_t*sizeof(float), 64 );             /* Array for permuted weight gradiant */

//     for(int w = 0; w < F_t*C_t*WW_t; w++){          /* Initialize array */
//         flip_d_weight_a[w] = 0.0f;
//     }

//     /*  lda = W_t;                                  // Already defined variables */
//     /* ldb = Win_t; */
//     /* ldc = C_t; */
//     l_br = WW_t;                                    /* Number of batches in brGEMM (= width of kernel = 51) */
//     tile_multiple = (W_t/XS_TILE_WBACKWARD)*XS_TILE_WBACKWARD;


//     /* Blocking on grad_a */
//     int lda_g = Win_t;
//     int ldb_trans_g = F_t;
//     int ldc_g = F_t;

//     libxsmm_blasint M_g = W_t/2;
//     libxsmm_blasint N_g = F_t;
//     libxsmm_blasint short_W_t = XS_TILE_WBACKWARD;
//     libxsmm_blasint edge_W_t = W_t - tile_multiple;

//     auto grad_shortvnni_backweight = grad.new_empty({N_t,F_t,short_W_t});                            /* Short buffer for storing VNNI transform */
//     libxsmm_bfloat16* grad_shortvnni = (libxsmm_bfloat16*) grad_shortvnni_backweight.data_ptr<at::BFloat16>();

//     auto grad_edgevnni_backweight = grad.new_empty({N_t,F_t,edge_W_t});                              /* Short buffer for storing VNNI transform in edge case */
//     libxsmm_bfloat16* grad_edgevnni = (libxsmm_bfloat16*) grad_edgevnni_backweight.data_ptr<at::BFloat16>();

//     /* use jited tranpose */
//     unary_shape = libxsmm_create_meltw_unary_shape( short_W_t/2, N_g, M_g, N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_shortkernel_grad = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     unary_shape = libxsmm_create_meltw_unary_shape( edge_W_t/2, N_g, M_g, N_g, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32 );
//     libxsmm_meltwfunction_unary trans_edgekernel_grad = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

//     if ((trans_shortkernel_grad == NULL) || (trans_edgekernel_grad == NULL)) {
//         fprintf( stderr, "JIT unary TPP for trans_shortkernel_grad (NORM_TO_NORM Transform) in backward data failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Dispatch brGEMM kernels for the normal case and the edge case*/
//     l_flags = LIBXSMM_GEMM_FLAG_VNNI_A;
//     l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

//     l_shape = libxsmm_create_gemm_shape(F_t, C_t, XS_TILE_WBACKWARD, ldb_trans_g, lda_g, ldc_g, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16);
//     libxsmm_gemmfunction backweight_kernel_main = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch);

//     l_shape = libxsmm_create_gemm_shape(F_t, C_t, W_t - tile_multiple, ldb_trans_g, lda_g, ldc_g, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16);
//     libxsmm_gemmfunction backweight_kernel_edge = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch);

//     if ((backweight_kernel_main == NULL) || (backweight_kernel_edge == NULL)) {
//         fprintf( stderr, "JIT for backweight_kernel_main failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Main compute loop for backward weight pass */
//     #pragma omp parallel for reduction(+: flip_d_weight_a[:F_t*C_t*WW_t])
//     for(int n = 0; n < N_t; n++) {
//         int last_block = 0;
//         libxsmm_meltw_unary_param trans_param_short, trans_param_edge;
//         libxsmm_gemm_param gemm_param_main, gemm_param_edge;

//         for(int wb = 0; wb < W_t - XS_TILE_WBACKWARD + 1; wb += XS_TILE_WBACKWARD) {            /* Main Case */

//             /* Take transpose assumping FP32 (This will do both transpose and VNNI transform for BF16) */
//             trans_param_short.in.primary  = &grad_a[n*F_t*W_t + wb];
//             trans_param_short.out.primary = &grad_shortvnni[n*F_t*short_W_t];
//             trans_shortkernel_grad( &trans_param_short );

//             for(int kw = 0; kw < WW_t; kw++) {
//                 gemm_param_main.a.primary = &grad_shortvnni[n*F_t*short_W_t];
//                 gemm_param_main.b.primary = &input_a[n*C_t*Win_t + wb + kw*dial];
//                 gemm_param_main.c.primary = &flip_d_weight_a[kw*C_t*F_t];
//                 backweight_kernel_main( &gemm_param_main );
//             }
//             last_block = wb;
//         }

//         if (W_t % XS_TILE_WBACKWARD != 0){              /* Edge Case */

//             trans_param_edge.in.primary  = &grad_a[n*F_t*W_t + last_block + XS_TILE_WBACKWARD];
//             trans_param_edge.out.primary = &grad_edgevnni[n*F_t*edge_W_t];
//             trans_edgekernel_grad( &trans_param_edge );

//             for(int kw = 0; kw < WW_t; kw++) {
//                 gemm_param_edge.a.primary = &grad_edgevnni[n*F_t*edge_W_t];
//                 gemm_param_edge.b.primary = &input_a[n*C_t*Win_t + (last_block + XS_TILE_WBACKWARD) + kw*dial];
//                 gemm_param_edge.c.primary = &flip_d_weight_a[kw*F_t*C_t];
//                 backweight_kernel_edge( &gemm_param_edge );
//             }
//         }
//     }


//     auto flip_d_weight_tensor = weight.new_empty({WW_t,C_t,F_t});
//     libxsmm_bfloat16* flip_d_weight_bf16 = (libxsmm_bfloat16*) flip_d_weight_tensor.data_ptr<at::BFloat16>();


//     /* JIT eltwise TPPs for FP32 to BF16 conversion... */
//     libxsmm_blasint cvt_m = 1;
//     libxsmm_blasint cvt_n = F_t*C_t*WW_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( cvt_m, cvt_n, cvt_m, cvt_m, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary eltwise_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( eltwise_kernel == NULL ) {
//         fprintf( stderr, "JIT unary TPP to convert FP32 to BF16 failed. Bailing...!\n");
//         exit(-1);
//     }

//     libxsmm_meltw_unary_param eltwise_params;
//     eltwise_params.in.primary = flip_d_weight_a;
//     eltwise_params.out.primary = flip_d_weight_bf16;
//     eltwise_kernel(&eltwise_params);


//     /* jited tranpose to permute the array dimensions
//         Overall Convert (WW_t, C_t, F_t) -----> (F_t, C_t, WW_t)*/
//     libxsmm_blasint per_m1 = F_t;
//     libxsmm_blasint per_n1 = C_t;
//     libxsmm_blasint ldi_per_1 = F_t;
//     libxsmm_blasint ldo_per_1 = C_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( per_m1, per_n1, ldi_per_1, ldo_per_1, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary trans_permute_1 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_permute_1 == NULL) {
//         fprintf( stderr, "JIT unary TPP trans_permute_1 (NORM_TO_NORM Transform) in backward weight failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Convert (WW_t, C_t, F_t) -----> (WW_t, F_t, C_t) */
//     #pragma omp parallel for
//     for(int kw = 0; kw < WW_t; kw++){                   /* permute last two dimensions */
//         libxsmm_meltw_unary_param trans_param_permute_1;
//         trans_param_permute_1.in.primary  = &flip_d_weight_bf16[kw*C_t*F_t];
//         trans_param_permute_1.out.primary = &flip_weight_a[kw*C_t*F_t];
//         trans_permute_1( &trans_param_permute_1 );
//     }

//     libxsmm_blasint per_m2 = F_t*C_t;
//     libxsmm_blasint per_n2 = WW_t;
//     libxsmm_blasint ldi_per_2 = F_t*C_t;
//     libxsmm_blasint ldo_per_2 = WW_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( per_m2, per_n2, ldi_per_2, ldo_per_2, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary trans_permute_2 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
//     if ( trans_permute_2 == NULL) {
//         fprintf( stderr, "JIT unary TPP trans_permute_2 (NORM_TO_NORM Transform) in backward weight failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Convert (WW_t, F_t, C_t) -----> (F_t, C_t, WW_t) */
//     libxsmm_meltw_unary_param trans_param_permute_2;
//     trans_param_permute_2.in.primary  = flip_weight_a;
//     trans_param_permute_2.out.primary = d_weight_a;
//     trans_permute_2( &trans_param_permute_2 );

//     libxsmm_free(flip_d_weight_a);

//     return {d_input, d_weight};
// }
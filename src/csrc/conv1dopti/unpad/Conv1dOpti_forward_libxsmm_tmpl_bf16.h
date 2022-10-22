
RECORD_FUNCTION(
    "Conv1dOpti_forward_bf16",
    std::vector<c10::IValue>({input, weight})); // For recording time

int64_t N_t = input.size(0); /* Batch */
int64_t C_t = input.size(1); /* Channel */
int64_t Win_t = input.size(2); /* input width */

int64_t F_t = weight.size(0); /* Number of filters */
int64_t WW_t = weight.size(2); /* filter width */

int64_t dial = dilation; /* dilation parameter */
int64_t pad_size = ((WW_t - 1)) * dial; /* Total padding size */
int64_t W_t = Win_t - pad_size; /* output width */

auto Y = input.new_empty({N_t, F_t, W_t}); /* New tensor for output */

auto input_a = GetVLAPtr<T>(input, {C_t, Win_t});
auto weight_a = GetVLAPtr<T>(weight, {C_t, WW_t});
auto Y_a = GetVLAPtr<T>(Y, {F_t, W_t});

auto flip_weight =
    weight.new_empty({WW_t, F_t, C_t}); /* Array to store permuted weight tensor
                                           (width, filters, channels) */
auto flip_weight_a = GetVLAPtr<T>(flip_weight, {F_t, C_t});

/* tranpose to permute the array dimensions
Overall convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)*/
auto trans_permute_tpp =
    SCOPEIT(XformExtTPP<T>(F_t * C_t, WW_t, XformTPP::XFORM_XPOSE_TPP), XPOSE);
trans_permute_tpp(&weight_a[0][0][0], &flip_weight_a[0][0][0]);

int64_t lda = C_t; /* Input channels (16) */
// int ldb = Win_t;                    /* Input width    (60400) */
int64_t ldc = W_t; /* Output width   (60000) */
unsigned long long l_br =
    WW_t; /* Number of batches in brGEMM (= width of kernel = 51) */

int64_t tile_multiple = (W_t / XS_TILE_FORWARD) *
    XS_TILE_FORWARD; /* Number of blocks/Tiles in the output width */

int64_t main_width =
    ((XS_TILE_FORWARD + (WW_t - 1) * dial) / XS_TILE_FORWARD + 1) *
    XS_TILE_FORWARD; /* width of main buffer */
auto input_mainvnni = input.new_empty(
    {N_t, C_t, main_width}); /* VNNI transformed array of the main buffer */
auto input_mainvnni_a = GetVLAPtr<T>(input_mainvnni, {C_t, main_width});

int64_t edge_width =
    (((W_t - tile_multiple) + (WW_t - 1) * dial) / XS_TILE_FORWARD + 1) *
    XS_TILE_FORWARD; /* width of buffer in the edge case (last block) */
auto input_edgevnni = input.new_empty(
    {N_t,
     C_t,
     edge_width}); /* VNNI VNNI transformed array of the edge buffer */
auto input_edgevnni_a = GetVLAPtr<T>(input_edgevnni, {C_t, edge_width});

/* JIT eltwise TPPs for initialization... */
int64_t tpp_m1 = XS_TILE_FORWARD; /* columns */
int64_t tpp_m2 = (W_t - tile_multiple); /* columns */
int64_t tpp_n = F_t; /* rows */
int64_t tpp_k = C_t;
int64_t ld_zero = W_t;

auto main_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
    tpp_n,
    tpp_m1,
    tpp_k,
    F_t* C_t,
    2 * dial,
    lda,
    main_width,
    ldc,
    1.0,
    0,
    1)));
auto edge_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
    tpp_n,
    tpp_m2,
    tpp_k,
    F_t* C_t,
    2 * dial,
    lda,
    edge_width,
    ldc,
    1.0,
    0,
    1)));

auto main_zero_tpp = SCOPEIT(SetZeroTPP<T>(tpp_n, tpp_m1, ld_zero), EW_ZERO);
auto edge_zero_tpp = SCOPEIT(SetZeroTPP<T>(tpp_n, tpp_m2, ld_zero), EW_ZERO);

// /* use jited VNNI */
int64_t ldi = Win_t;
int64_t ldo_main = main_width;
int64_t ldo_edge = edge_width;
tpp_m1 = (XS_TILE_FORWARD + dial * (WW_t - 1));
tpp_m2 = (W_t - tile_multiple + dial * (WW_t - 1));

auto mainvnni_trans_tpp = SCOPEIT(
    XformExtTPP<
        T>(C_t, tpp_m1, C_t, tpp_m1, ldi, ldo_main, XformTPP::XFORM_N2V_TPP),
    VNNI);
auto edgevnni_trans_tpp = SCOPEIT(
    XformExtTPP<
        T>(C_t, tpp_m2, C_t, tpp_m2, ldi, ldo_edge, XformTPP::XFORM_N2V_TPP),
    VNNI);

/* Main compute loop */
{
  RECORD_SCOPE(forward_loop_bf16, {Y, input, flip_weight});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < N_t; n++) { /* Loop for batches */
      int last_block = 0;

      for (int wb = 0; wb < W_t - XS_TILE_FORWARD + 1;
           wb += XS_TILE_FORWARD) { /* width blocking loop (Main case) */

        /* Main case */
        main_zero_tpp(&Y_a[n][0][wb]);

        /* VNNI transform */
        mainvnni_trans_tpp(&input_a[n][0][wb], &input_mainvnni_a[n][0][0]);

        /* brGEMM */
        main_brgemm_tpp(
            &flip_weight_a[0][0][0],
            &input_mainvnni_a[n][0][0],
            &Y_a[n][0][wb],
            l_br);
        last_block = wb; /* Store value for last block */
      }

      if (W_t % XS_TILE_FORWARD != 0) { /* Edge case */

        edge_zero_tpp(&Y_a[n][0][last_block + XS_TILE_FORWARD]);

        /* VNNI transform */
        edgevnni_trans_tpp(
            &input_a[n][0][last_block + XS_TILE_FORWARD],
            &input_edgevnni_a[n][0][0]);

        /* brGEMM */
        edge_brgemm_tpp(
            &flip_weight_a[0][0][0],
            &input_edgevnni_a[n][0][0],
            &Y_a[n][0][last_block + XS_TILE_FORWARD],
            l_br);
      }
    }
  }
}

return Y; /* Return output tensor */

// old code
// at::Tensor Conv1dOpti_forward_bf16_libxsmm(at::Tensor& input, at::Tensor&
// weight, int dilation){

//     /* RECORD_FUNCTION("Conv1dOpti_forward_bf16",
//     std::vector<c10::IValue>({input, weight}));        // For recording time
//     */

//     int64_t N_t = input.size(0);                    /* Batch */
//     int64_t C_t = input.size(1);                    /* Channel */
//     int64_t Win_t = input.size(2);                  /* input width */

//     int64_t F_t = weight.size(0);                   /* Number of filters */
//     int64_t WW_t = weight.size(2);                  /* filter width */

//     int64_t dial = dilation;                        /* dilation parameter */
//     int64_t pad_size = ((WW_t- 1))*dial;            /* Total padding size */
//     int64_t W_t = Win_t - pad_size;                 /* output width */

//     auto Y = input.new_empty({N_t,F_t,W_t});        /* New tensor for output
//     */

//     libxsmm_bfloat16* input_a = (libxsmm_bfloat16*)
//     input.data_ptr<at::BFloat16>();                /* Get BFloat16 data
//     pointers for accessing tensors */ libxsmm_bfloat16* weight_a =
//     (libxsmm_bfloat16*) weight.data_ptr<at::BFloat16>(); libxsmm_bfloat16*
//     Y_a = (libxsmm_bfloat16*) Y.data_ptr<at::BFloat16>();

//     auto flip_weight = weight.new_empty({WW_t,F_t,C_t}); /* Weight tensor
//     with permuted dimension (width, filters, channels) */ libxsmm_bfloat16*
//     flip_weight_a = (libxsmm_bfloat16*) flip_weight.data_ptr<at::BFloat16>();
//     /* Get BFloat16 data pointers for accessing the tensor */

//     /* jited tranpose to permute the array dimensions
//         Overall convert (F_t, C_t, WW_t) -----> (WW_t, F_t, C_t)*/
//     libxsmm_blasint per_m = WW_t;
//     libxsmm_blasint per_n = F_t*C_t;
//     libxsmm_blasint per_ldi = WW_t;
//     libxsmm_blasint per_ldo = F_t*C_t;

//     libxsmm_meltw_unary_shape unary_shape = libxsmm_create_meltw_unary_shape(
//     per_m, per_n, per_ldi, per_ldo, LIBXSMM_DATATYPE_BF16,
//     LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary trans_permute_kernel_bf16 =
//     libxsmm_dispatch_meltw_unary_v2(
//     LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, unary_shape,
//     LIBXSMM_MELTW_FLAG_UNARY_NONE ); if ( trans_permute_kernel_bf16 == NULL)
//     {
//         fprintf( stderr, "JIT unary TPP for trans_permute_kernel
//         (NORM_TO_NORM transform) in forward pass failed. Bailing...!\n");
//         exit(-1);
//     }
//     libxsmm_meltw_unary_param trans_permute_param;
//     trans_permute_param.in.primary  = weight_a;
//     trans_permute_param.out.primary = flip_weight_a;
//     trans_permute_kernel_bf16( &trans_permute_param);

//     int lda = C_t;                      /* Input channels (16) */
//     /*int ldb = Win_t;                     Input width    (60400) */
//     int ldc = W_t;                      /* Output width   (60000) */
//     unsigned long long l_br = WW_t;     /* Number of batches in brGEMM (=
//     width of kernel = 51) */

//     int tile_multiple = (W_t/XS_TILE_FORWARD)*XS_TILE_FORWARD; /* Number of
//     blocks/Tiles in the output width */

//     int main_width = ((XS_TILE_FORWARD + (WW_t-1)*dial)/XS_TILE_FORWARD +
//     1)*XS_TILE_FORWARD;          /* width of main buffer */ auto
//     input_mainvnni = input.new_empty({N_t,C_t,main_width}); /* VNNI
//     transformed array of the main buffer */ libxsmm_bfloat16*
//     input_a_mainvnni = (libxsmm_bfloat16*)
//     input_mainvnni.data_ptr<at::BFloat16>();  /* Get pointer */

//     int edge_width = (((W_t - tile_multiple) + (WW_t-1)*dial)/XS_TILE_FORWARD
//     + 1)*XS_TILE_FORWARD;     /* width of buffer in the edge case (last
//     block) */ auto input_edgevnni = input.new_empty({N_t,C_t,edge_width}); /*
//     VNNI VNNI transformed array of the edge buffer */ libxsmm_bfloat16*
//     input_a_edgevnni = (libxsmm_bfloat16*)
//     input_edgevnni.data_ptr<at::BFloat16>();   /* Get pointer */

//     /* Dispatch brGEMM kernels for the normal case and the edge case*/
//     libxsmm_gemm_flags l_flags;
//     libxsmm_gemm_prefetch_type l_prefetch;
//     libxsmm_gemm_shape l_shape;
//     libxsmm_gemm_batch_reduce_config l_brconfig;

//     l_flags = LIBXSMM_GEMM_FLAG_VNNI_A;
//     l_prefetch = LIBXSMM_GEMM_PREFETCH_NONE;

//     l_shape = libxsmm_create_gemm_shape(XS_TILE_FORWARD, F_t, C_t,
//     main_width, lda, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
//     LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16); l_brconfig.br_type =
//     LIBXSMM_GEMM_BATCH_REDUCE_STRIDE; l_brconfig.br_stride_a_hint =
//     dial*2*sizeof(libxsmm_bfloat16); l_brconfig.br_stride_b_hint =
//     F_t*C_t*sizeof(libxsmm_bfloat16); libxsmm_gemmfunction
//     brgemm_kernel_main_bf16 = libxsmm_dispatch_brgemm_v2(l_shape, l_flags,
//     l_prefetch, l_brconfig);

//     l_shape = libxsmm_create_gemm_shape(W_t - tile_multiple, F_t, C_t,
//     edge_width, lda, ldc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
//     LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16); l_brconfig.br_type =
//     LIBXSMM_GEMM_BATCH_REDUCE_STRIDE; l_brconfig.br_stride_a_hint =
//     dial*2*sizeof(libxsmm_bfloat16); l_brconfig.br_stride_b_hint =
//     F_t*C_t*sizeof(libxsmm_bfloat16); libxsmm_gemmfunction
//     brgemm_kernel_edge_bf16 = libxsmm_dispatch_brgemm_v2(l_shape, l_flags,
//     l_prefetch, l_brconfig);

//     /* JIT eltwise TPPs for initialization ... */
//     libxsmm_blasint tpp_m1 = XS_TILE_FORWARD;                      /* columns
//     */ libxsmm_blasint tpp_m2 = W_t - tile_multiple;                  /*
//     columns */ libxsmm_blasint tpp_n = F_t; /* rows */ libxsmm_blasint
//     ld_zero = W_t;

//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, tpp_n, tpp_m1,
//     ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
//     LIBXSMM_DATATYPE_BF16 ); libxsmm_meltwfunction_unary
//     copy_kernel_forward_main_bf16 = libxsmm_dispatch_meltw_unary_v2(
//     LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE
//     ); unary_shape = libxsmm_create_meltw_unary_shape( tpp_m2, tpp_n, tpp_m2,
//     ld_zero, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
//     LIBXSMM_DATATYPE_BF16 ); libxsmm_meltwfunction_unary
//     copy_kernel_forward_edge_bf16 = libxsmm_dispatch_meltw_unary_v2(
//     LIBXSMM_MELTW_TYPE_UNARY_XOR, unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE
//     );

//     if ((copy_kernel_forward_main_bf16 == NULL) ||
//     (copy_kernel_forward_edge_bf16 == NULL)) {
//         fprintf( stderr, "JIT unary TPP for copy_kernel_forward_main_bf16 in
//         forward pass failed. Bailing...!\n"); exit(-1);
//     }

//     /* use jited VNNI */
//     libxsmm_blasint ldi = Win_t;
//     libxsmm_blasint ldo_main = main_width;
//     libxsmm_blasint ldo_edge = edge_width;

//     libxsmm_meltw_unary_type trans_vnni_type;
//     if ( C_t % 2 == 1 ) {
//         trans_vnni_type =
//         LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD;
//     } else {
//         trans_vnni_type = LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2;
//     }
//     tpp_m1 = (XS_TILE_FORWARD + dial*(WW_t-1));
//     tpp_m2 = (W_t - tile_multiple + dial*(WW_t-1));

//     unary_shape = libxsmm_create_meltw_unary_shape( tpp_m1, C_t, ldi,
//     ldo_main, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
//     LIBXSMM_DATATYPE_BF16 ); libxsmm_meltwfunction_unary
//     trans_mainvnni_kernel = libxsmm_dispatch_meltw_unary_v2( trans_vnni_type,
//     unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE ); unary_shape =
//     libxsmm_create_meltw_unary_shape( tpp_m2, C_t, ldi, ldo_edge,
//     LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16 );
//     libxsmm_meltwfunction_unary trans_edgevnni_kernel =
//     libxsmm_dispatch_meltw_unary_v2( trans_vnni_type, unary_shape,
//     LIBXSMM_MELTW_FLAG_UNARY_NONE );

//     if ( (trans_mainvnni_kernel == NULL) || (trans_edgevnni_kernel == NULL))
//     {
//         fprintf( stderr, "JIT unary TPP for trans_mainvnni_kernel
//         (NORM_TO_VNNI transform) in forward pass failed. Bailing...!\n");
//         exit(-1);
//     }

//     /* Main compute loop */
//     #pragma omp parallel for
//     for(int n = 0; n < N_t; n++) { /* Loop for batches */
//         int last_block = 0;
//         libxsmm_meltw_unary_param copy_params_main, copy_params_edge; /* Copy
//         parameter variable for holding the pointer */
//         libxsmm_meltw_unary_param trans_param_main, trans_param_edge;
//         libxsmm_gemm_param gemm_param_main, gemm_param_edge;

//         for(int wb = 0; wb < W_t - XS_TILE_FORWARD + 1; wb +=
//         XS_TILE_FORWARD) {        /* width blocking loop (Main case) */

//             copy_params_main.out.primary = &Y_a[n*F_t*W_t + wb]; /*
//             Initialization of output array */
//             copy_kernel_forward_main_bf16(&copy_params_main);

//             /* VNNI transform */
//             trans_param_main.in.primary  = &input_a[n*C_t*Win_t + 0*Win_t +
//             wb]; trans_param_main.out.primary =
//             &input_a_mainvnni[n*C_t*main_width]; trans_mainvnni_kernel(
//             &trans_param_main );

//             /* brGEMM */
//             gemm_param_main.a.primary = &input_a_mainvnni[n*C_t*main_width];
//             gemm_param_main.b.primary = &flip_weight_a[0];
//             gemm_param_main.c.primary = &Y_a[n*F_t*W_t + 0*W_t + wb];
//             gemm_param_main.op.tertiary = &l_br;
//             brgemm_kernel_main_bf16( &gemm_param_main );

//             last_block = wb; /* Store value for last block */
//         }

//         if (W_t % XS_TILE_FORWARD != 0){ /* Edge case */

//             copy_params_edge.out.primary = &Y_a[n*F_t*W_t + last_block +
//             XS_TILE_FORWARD];                 /* Initialization of output
//             array */ copy_kernel_forward_edge_bf16(&copy_params_edge);

//             /* VNNI transform */
//             trans_param_edge.in.primary  = &input_a[n*C_t*Win_t + 0*Win_t +
//             (last_block + XS_TILE_FORWARD)]; trans_param_edge.out.primary =
//             &input_a_edgevnni[n*C_t*edge_width]; trans_edgevnni_kernel(
//             &trans_param_edge );

//             /* brGEMM */
//             gemm_param_edge.a.primary = &input_a_edgevnni[n*C_t*edge_width];
//             gemm_param_edge.b.primary = &flip_weight_a[0];
//             gemm_param_edge.c.primary = &Y_a[n*F_t*W_t + 0*W_t + (last_block
//             + XS_TILE_FORWARD)]; gemm_param_edge.op.tertiary = &l_br;
//             brgemm_kernel_edge_bf16( &gemm_param_edge );
//         }
//     }

//     return Y;              /* Return output tensor */
// }
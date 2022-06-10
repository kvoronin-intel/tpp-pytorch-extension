RECORD_FUNCTION("conv_bwd", std::vector<c10::IValue>());

// ( grad_output, input, weight) = inputs

auto t_GO = inputs[0]; // [N][Kb][H][W][bk]
auto t_I  = inputs[1]; // [N][Cb][H][W][bc]
auto t_W  = inputs[2];
//auto t_scratch = inputs[3];

auto sizes = t_I.sizes();

int R = cfg.R;
int S = cfg.S;
int ifh = cfg.H;
int ifw = cfg.W;
int ofh = cfg.ofh;
int ofw = cfg.ofw;
int ifhp = cfg.ifhp;
int ifwp = cfg.ifwp;
int ofhp = cfg.ofhp;
int ofwp = cfg.ofwp;
int bk = cfg.bk;
int bc = cfg.bc;
int bn = cfg.N;
int stride_h = cfg.u;
int stride_w = cfg.v;
int Cb = cfg.blocksifm;
int Kb = cfg.blocksofm;

int pad_h_in = cfg.pad_h_in;
int pad_w_in = cfg.pad_w_in;
int pad_h_out = cfg.pad_h_out;
int pad_w_out = cfg.pad_w_out;

int nThreads = cfg.threads;

int C = Cb * bc;
int K = Kb * bk;

const long N  = sizes[0];

std::vector<long> output_size{N, Kb, ofhp, ofwp, bk};

std::vector<long> weight_tr_size{Cb, Kb, R, S, bk, bc};

/*
std::cout << "t_I sizes = " << t_I.sizes() << std::endl;
std::cout << "output_size = " << output_size << std::endl;
std::cout << "R = " << R << " S = " << S << std::endl;
std::cout << "stride_h = " << stride_h << " stride_w = " << stride_w << std::endl;
std::cout << "pad_h_in = " << pad_h_in << " pad_w_in = " << pad_w_in << std::endl;
std::cout << "Cb Kb bc Kb bk = " << Cb << " " << Kb << " " << bc << " " << Kb << " " << bk << std::endl;
//std::cout << "weight_tr_size = " << weight_tr_size << std::endl;
*/

auto t_grad_input  = at::empty(t_I.sizes(), torch::TensorOptions().dtype(t_I.dtype()));
auto t_grad_weight = at::empty(t_W.sizes(), torch::TensorOptions().dtype(t_W.dtype()));
auto t_WT          = at::empty(weight_tr_size, torch::TensorOptions().dtype(t_W.dtype()));

//------------------------------------

  long  pad_h = pad_h_out;
  long  pad_w = pad_w_out;

  /* Some algorithmic knobs  */
  /* Uses parallelism in the MB dimension for f32 precision */
  long use_mb_par_f32 = 1; //1; FIXME back

  /* Fuse bf16 necessary transposes */
  long bf16_use_nchw_format = 1;//1; // FIXME back!
  long bf16_use_chwn_format = 1;
  long bf16_fuse_upd_transposes = 1;//1; FIXME back!

  /* Control variants for chwn format */
  long bf16_acc_nw = 1;
  long par_over_h_pixels = 1;
  long use_private_trans = 0;

  /* Control variants for nchw format */
  long pack_input_upfront = 0;
  long compute_pixels = 0;
  long remainder_pixels = 0;
  long upd_remaining_pixels = 0;
  long accum_length_pixels = 0;
  long max_init_offset = 0;
  long input_compute_pad = 0;
  long input_pixels = 0;
  long output_pixels = 0;
  long pixel_blocking = 0;
  long n_used_pixels = 0;
  long use_intermediate_f32_wt_tensor = 0;
  long use_hybrid_imgfm_parallelization = 0; //0; FIXME back
  long n_img_teams = 7;
  long n_ofm_teams = 4;
  long weight_copies = 0;
  long multiple_target = 2;
  long max_compute_offset_input = 0;
  long use_f32_wt_reduction_and_external_wt_vnni = 0; //0; FIXME back
  long compute_full_wt_output_block = 0; // 0; FIXME back

  bf16_use_chwn_format = (bf16_use_nchw_format > 0) ? 0 : 1;
  use_private_trans = bf16_fuse_upd_transposes;

//  long bn = N; // declared earlier
//------------------------------------

long long int running_scratch_size_in_bytes = 0;

std::vector<long> scratch_size{nThreads, C, K, R, S};

running_scratch_size_in_bytes += nThreads * C * K * R * S * sizeof(T);

long long int private_tr_input_offset = 0, private_tr_output_offset = 0,
              tr_input_offset = 0, tr_output_offset = 0,
              scratch_float_offset = 0, scratch_bf16_weight_offset = 0,
              input_mylinearized_pixels_offset = 0, output_mylinearized_pixels_offset = 0;

if (sizeof(T) == 2) {
  //t_private_tr_input  = at::empty({nThreads, N, ifhp, ifwp, C}, torch::TensorOptions().dtype(at::kBFloat16));
  //t_private_tr_output = at::empty({nThreads, N, ofhp, ofwp, K}, torch::TensorOptions().dtype(at::kBFloat16));
  private_tr_input_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += nThreads * N * ifhp * ifwp * C * sizeof(T);
  private_tr_output_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += nThreads * N * ofhp * ofwp * K * sizeof(T);
}

if (sizeof(T) == 2) {
  //t_tr_input  = at::empty({N, ifhp, ifwp, C}, torch::TensorOptions().dtype(at::kBFloat16));
  //t_tr_output = at::empty({N, ofhp, ofwp, K}, torch::TensorOptions().dtype(at::kBFloat16));
  tr_input_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += N * ifhp * ifwp * C * sizeof(T);
  tr_output_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += N * ofhp * ofwp * K * sizeof(T);
}

if (sizeof(T) == 2) {
  //t_scratch_float       = at::empty({nThreads, C, K, R, S}, torch::TensorOptions().dtype(at::kFloat));
  //t_scratch_bf16_weight = at::empty({C, K, R, S},           torch::TensorOptions().dtype(at::kBFloat16));
  scratch_float_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += nThreads * C * K * R * S * sizeof(float);
  scratch_bf16_weight_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes +=            C * K * R * S * sizeof(T);
}

if (sizeof(T) == 2) {
    if (bf16_use_nchw_format > 0) {
     if (R == 1 && S == 1 && (stride_w != 1 || stride_h != 1)) {
        pack_input_upfront = 1;
      } else {
        pack_input_upfront = 0;
      }
      compute_pixels = ofw * ofh + 2 * pad_w * (ofh-1);
      remainder_pixels = (compute_pixels % multiple_target == 0) ? 0 : (compute_pixels/multiple_target+1)*multiple_target - compute_pixels;
      accum_length_pixels = compute_pixels + remainder_pixels;
      max_init_offset = 2 * pad_h * ifwp + 2 * pad_w;
      max_compute_offset_input = max_init_offset + accum_length_pixels;
      input_compute_pad = (max_compute_offset_input > ifwp*ifhp) ? max_compute_offset_input - ifwp*ifhp : 0;
      input_pixels = ifwp*ifhp+ input_compute_pad;
      if (pack_input_upfront) {
        input_pixels = accum_length_pixels;
      }
      output_pixels = accum_length_pixels;
      pixel_blocking = accum_length_pixels;
      n_used_pixels = accum_length_pixels;
      use_intermediate_f32_wt_tensor = (pixel_blocking == n_used_pixels) ? 0 : 1;
      float beta = (use_intermediate_f32_wt_tensor) ? (float)1.0 : (float)0.0;
      if (use_hybrid_imgfm_parallelization == 0) {
        ;
      } else {
        ;
      }
      upd_remaining_pixels = output_pixels - ((compute_pixels+1)/2)*2;
    }
  //input_linearized_pixels  = (DType*)libxsmm_aligned_malloc( N*input_pixels*C*sizeof(DType), 2097152);
  input_mylinearized_pixels_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += N * input_pixels * C * sizeof(T);
  //output_linearized_pixels = (DType*)libxsmm_aligned_malloc( N*output_pixels*K*sizeof(DType), 2097152);
  output_mylinearized_pixels_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += N * output_pixels * K * sizeof(T);
}

auto max_scratch_size_in_bytes = running_scratch_size_in_bytes;

auto t_scratch_experimental = at::empty({max_scratch_size_in_bytes}, torch::TensorOptions().dtype(at::kByte));

//return std::vector<at::Tensor>({t_grad_input, t_grad_weight});

{ /* main dummy scope */

  long fm_blocking = (bk % 16 == 0) ? 16 : bk;
  long reduce_work = Kb * C * R * S * (bk/fm_blocking);
  long reduce_chunk_size = (reduce_work + nThreads - 1)/nThreads;
  long reduce_work_tripcount = (reduce_work + reduce_chunk_size - 1) / reduce_chunk_size;

  long chunk0 = reduce_chunk_size * fm_blocking;
  long chunk1 = K * C * R * S  - (reduce_work_tripcount-1) * chunk0;
  chunk1 = (chunk1 <= 0) ? chunk0 : chunk1;

  int gemm_m = 0, gemm_n = 0, gemm_k = 0;

  char bf16_conv_spec_string[256];
  char fp32_conv_spec_string[256];

  //sprintf(fp32_conv_spec_string, "Abcdefg");
  if (use_mb_par_f32 == 0)
    sprintf(fp32_conv_spec_string, "abcdefg"); // specifically for checking the case use_mb_par_f32 = 0
  else
    sprintf(fp32_conv_spec_string, "Abcdefg");
  sprintf(bf16_conv_spec_string, "Abcdef");
  //sprintf(bf16_conv_spec_string, "A{C:7}bC{R:4}def");//specifically to test the code path with col_id = ind[9]

  if (use_hybrid_imgfm_parallelization > 0) {
    bf16_fuse_upd_transposes = 0;
    weight_copies = n_img_teams;
  } else {
    weight_copies = nThreads;
  }

  int trans_tracker_size = 0;
  std::unique_ptr<int[]> trans_tracker;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>)         gemm_as_brgemm_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>)               zero_tpp;
  SCOPEIT_DECL(ReduceAddColExtTPP<T,T>)     wt_reduce0_T_tpp, wt_reduce1_T_tpp;

  SCOPEITGEMM_DECL(BrgemmTPP<T, float>)     brgemm_acc_pixel_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>)               zero_bf16_tpp;
  SCOPEIT_DECL(SetZeroTPP<float>)           zero_float_tpp;
  SCOPEIT_DECL(ConvertTPP<float, T>)        fp32bf16_cvt_tpp;
  SCOPEIT_DECL(XformExtTPP<T>)              trans_xform_tpp, vnni_xform_tpp, wt_vnni_xform_tpp;
  SCOPEIT_DECL(ReduceAddColExtTPP<float,T>) wt_reduce0_float_tpp, wt_reduce1_float_tpp;

  /* Should only be used for bfloat16 */
  SCOPEIT_DECL(XformExtTPP<T>)              vnni_output_compute_pixels_bf16_xform_tpp, transpose_input_pixels_bf16_xform_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>)               vnni_output_zero_remaining_pixels_bf16_tpp;
  SCOPEITGEMM_DECL(BrgemmTPP<T, float>)     gemm_kernel_non_hybrid_as_brgemm_tpp;
  SCOPEITGEMM_DECL(BrgemmTPP<T, T>)         gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp;
  //SCOPEITGEMM_DECL(GemmTPP<T, T>)           gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp;
  SCOPEIT_DECL(ReduceAddColExtTPP<T,T>)     wt_reduce0_bf16bf16_tpp, wt_reduce1_bf16bf16_tpp;
  SCOPEIT_DECL(ReduceAddColExtTPP<float,T>) wt_reduce0_f32bf16_tpp, wt_reduce1_f32bf16_tpp;
  SCOPEITGEMM_DECL(BrgemmTPP<T, float>)     brgemm_kernel_hybrid_tpp;
  SCOPEITGEMM_DECL(BrgemmTPP<T, T>)         brgemm_kernel_hybrid_zerobeta_cvnni_tpp;

  if (sizeof(T) == 4) {
    gemm_n = bc;
    gemm_m = bk;
    gemm_k = ofw;

    //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, dtype, dtype, dtype);
    //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);

    //l_unary_shape.m         = chunk0;
    //l_unary_shape.n         = nThreads;
    //l_unary_shape.ldi       = K * C * R * S ;
    //l_unary_shape.ldo       = chunk0;
    //wt_reduce_kernel0_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce0_T_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(nThreads, chunk0, K*C*R*S, chunk0)), EW_RED);

    //l_unary_shape.m         = chunk1;
    //l_unary_shape.ldo       = chunk1;
    //wt_reduce_kernel1_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce1_T_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(nThreads, chunk1, K*C*R*S, chunk1)), EW_RED);

    //auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    gemm_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* irrelevant strides */ 1, 1, bc*stride_w, bk, bk, 1.0, 1 /*a_trans*/, 0)));//, BRGEMM);

  } else { /* bfloat16 goes here */

    gemm_n = bc;
    gemm_m = bk;
    gemm_k = bn;

    //auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, bn, C*ifhp*ifwp, bn, dtype, dtype, dtype);
    //trans_xform_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    //std::cout << "trans_xform_tpp " << std::endl;
    trans_xform_tpp = SCOPEIT(XformExtTPP<T>(bn, bc, bc, bn, C*ifhp*ifwp, bn, XformTPP::XFORM_XPOSE_TPP, false), XPOSE); /* assuming row-major-ness */
    // FIXME: Should it be bn, bc for the out_rows, out_cols?

    //tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, K*ofhp*ofwp, bk, dtype, dtype, dtype);
    //vnni_xform_kernel =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    //std::cout << "vnni_xform_tpp " << std::endl;
    vnni_xform_tpp = SCOPEIT(XformExtTPP<T>(bn, bk, bn, bk, K*ofhp*ofwp, bk, XformTPP::XFORM_N2V_TPP, false), XPOSE); /* assuming row-major-ness */
    /* Tr input is : Cb ifhp ifwp bc bn  */
    /* Tr output is: Kb ofhp ofwp bn/2 bk 2  */

    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bn, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    //tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    //tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
    //gemm_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* irrelevant strides */ 1, 1, bn, bk, bk, 1.0, 0 /*a_trans*/, 0)));//, BRGEMM);

    //std::cout << "brgemm_acc_pixel_tpp " << std::endl;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bn*bk*sizeof(DType), stride_w*bc*bn*sizeof(DType), 0 );
    //brgemm_kernel_acc_pixel.gemm  = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm_acc_pixel_tpp = SCOPEITGEMM((BrgemmTPP<T,float>(gemm_n, gemm_m, gemm_k, stride_w*bc*bn, bn*bk, bn, bk, bk, 1.0, 0 /*a_trans*/, 0)));//, BRGEMM);

    //std::cout << "zero_bf16_tpp " << std::endl;
    //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
    //zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    zero_bf16_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);

    //l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    zero_float_tpp = SCOPEIT(SetZeroTPP<float>(bk*gemm_n), EW_ZERO);

    //l_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bk, LIBXSMM_DATATYPE_F32, dtype, LIBXSMM_DATATYPE_F32);
    //fp32bf16_cvt_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    fp32bf16_cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bc, bk, bk, bk)), EW_ZERO);

    //l_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bk, dtype, dtype, dtype);
    //wt_vnni_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    //std::cout << "wt_vnni_xform_tpp " << std::endl;
    wt_vnni_xform_tpp = SCOPEIT(XformExtTPP<T>(bc, bk, bk, bk, XformTPP::XFORM_N2V_TPP, false), XPOSE); /* assuming row-major-ness */

    //l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, nThreads, K * C *R * S, chunk0, LIBXSMM_DATATYPE_F32, dtype, LIBXSMM_DATATYPE_F32);
    //wt_reduce_kernel0_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce0_float_tpp = SCOPEIT((ReduceAddColExtTPP<float,T>(nThreads, chunk0, K*C*R*S, chunk0)), EW_RED);

    //l_unary_shape.m         = chunk1;
    //l_unary_shape.ldo       = chunk1;
    //wt_reduce_kernel1_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ; 
    wt_reduce1_float_tpp = SCOPEIT((ReduceAddColExtTPP<float,T>(nThreads, chunk1, K*C*R*S, chunk1)), EW_RED);
    //printf("l_unary_shape m n ldi ldo = %d %d %d %d \n", l_unary_shape.m, l_unary_shape.n, l_unary_shape.ldi, l_unary_shape.ldo);

    trans_tracker_size = Cb + Kb + 64 - 64%(Cb+Kb);
    //int *trans_tracker = (int*)libxsmm_aligned_malloc( nThreads*trans_tracker_size*sizeof(int), 2097152);
    trans_tracker = std::make_unique<int[]>(nThreads * (Cb + Kb + 64 - 64%(Cb+Kb)));


    //l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, weight_copies, K * C *R * S, chunk0, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32);
    //wt_reduce_kernel0_bf16bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce0_bf16bf16_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(weight_copies, chunk0, K*C*R*S, chunk0)), EW_RED);
    //l_unary_shape.m         = chunk1;
    //l_unary_shape.ldo       = chunk1;
    //wt_reduce_kernel1_bf16bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce1_bf16bf16_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(weight_copies, chunk1, K*C*R*S, chunk1)), EW_RED);

    //l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, weight_copies, K * C *R * S, chunk0, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32);
    //wt_reduce_kernel0_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce0_f32bf16_tpp = SCOPEIT((ReduceAddColExtTPP<float,T>(weight_copies, chunk0, K*C*R*S, chunk0)), EW_RED);
    //l_unary_shape.m         = chunk1;
    //l_unary_shape.ldo       = chunk1;
    //wt_reduce_kernel1_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce1_f32bf16_tpp = SCOPEIT((ReduceAddColExtTPP<float,T>(weight_copies, chunk1, K*C*R*S, chunk1)), EW_RED);

    if (bf16_use_nchw_format > 0) {
      if (use_hybrid_imgfm_parallelization == 0) {

        //auto new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
        //auto new_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
        //auto new_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
        //if (use_intermediate_f32_wt_tensor == 0) {
        //  new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        //}
        //gemm_kernel_non_hybrid.gemm = libxsmm_dispatch_gemm_v2( new_shape, new_flags, new_prefetch_flags );

        //printf("for gemm_kernel_non_hybrid as brgemm extension\n");
        if (use_intermediate_f32_wt_tensor == 0) {
          //new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
          gemm_kernel_non_hybrid_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,float>(bc, bk, pixel_blocking, /* irrelevant strides */ 1, 1, input_pixels, bk, bk, 0.0 /* beta */, 0 /*a_trans*/, 0)));//, BRGEMM);
        } else
          gemm_kernel_non_hybrid_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,float>(bk, bc, pixel_blocking, /* irrelevant strides */ 1, 1, input_pixels, bk, bk, 1.0 /* beta */, 0 /*a_trans*/, 0)));//, BRGEMM);

        //auto new_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
        //if (use_intermediate_f32_wt_tensor == 0) {
        //  new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        //}

        //new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, dtype, dtype);
        //new_flags |=  LIBXSMM_GEMM_FLAG_BETA_0  | LIBXSMM_GEMM_FLAG_VNNI_C;
        //gemm_kernel_non_hybrid_zerobeta_cvnni.gemm      = libxsmm_dispatch_gemm_v2( new_shape, new_flags, new_prefetch_flags );

        //printf("Attempting to create gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp\n");
        gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(bc, bk, pixel_blocking, /* irrelevant strides */ 1, 1, input_pixels, bk, bk, 0.0 /* beta */, 0 /*a_trans*/, 1 /*c_vnni */, 0)));//, BRGEMM);
        //gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp = SCOPEITGEMM((GemmTPP<T,T>(bc, bk, pixel_blocking, input_pixels, bk, bk, 0.0 /* beta */, 0 /*a_trans*/, 0, 1 /* b_vnni */, 1 /*c_vnni*/)));//, BRGEMM);
        //printf("Success\n");

      } else { /* for use_hybrid_imgfm_parallelization == 0 */
/*
        auto new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
        auto new_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
        auto new_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
        if (use_intermediate_f32_wt_tensor == 0) {
          new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        }
        auto new_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 );
        brgemm_kernel_hybrid.gemm   = libxsmm_dispatch_brgemm_v2( new_shape, new_flags, new_prefetch_flags, new_brconfig );
*/
        long stride_a = K * output_pixels; // will be multiplied by element size inside the wrapper
        long stride_b = C * input_pixels;

        if (use_intermediate_f32_wt_tensor == 0) {
          //new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
          brgemm_kernel_hybrid_tpp = SCOPEITGEMM((BrgemmTPP<T,float>(bc, bk, pixel_blocking, stride_b, stride_a, input_pixels, bk, bk, 0.0 /* beta */, 0 /*a_trans*/, 0)));//, BRGEMM);
        } else
          brgemm_kernel_hybrid_tpp = SCOPEITGEMM((BrgemmTPP<T,float>(bk, bc, pixel_blocking, stride_b, stride_a, input_pixels, bk, bk, 1.0 /* beta */, 0 /*a_trans*/, 0)));//, BRGEMM);
/*
        new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, dtype, dtype);
        new_flags |=  LIBXSMM_GEMM_FLAG_BETA_0  | LIBXSMM_GEMM_FLAG_VNNI_C;
        brgemm_kernel_hybrid_zerobeta_cvnni.gemm   = libxsmm_dispatch_brgemm_v2( new_shape, new_flags, new_prefetch_flags, new_brconfig );
*/
        brgemm_kernel_hybrid_zerobeta_cvnni_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(bc, bk, pixel_blocking, stride_b, stride_a, input_pixels, bk, bk, 0.0 /* beta */, 0 /*a_trans*/, 1 /*c_vnni */, 0)));//, BRGEMM);

      } /* else-if for use_hybrid_imgfm_parallelization == 0 */

      //auto new_tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, ifwp, bc, input_pixels, dtype, dtype, dtype);
      //transpose_input_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      transpose_input_pixels_bf16_xform_tpp = SCOPEIT(XformExtTPP<T>(ifwp, bc, bc, ifwp, bc, input_pixels, XformTPP::XFORM_XPOSE_TPP, false), XPOSE); /* assuming row-major-ness */
      //new_tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, compute_pixels, bk, bk, dtype, dtype, dtype);
      if ((ofhp * ofwp) % 2 == 0) {
        //vnni_output_compute_pixels_bf16 =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
        vnni_output_compute_pixels_bf16_xform_tpp = SCOPEIT(XformExtTPP<T>(compute_pixels, bk, compute_pixels, bk, bk, bk, XformTPP::XFORM_N2V_TPP, false), XPOSE); /* assuming row-major-ness */
      } else {
        //printf("Xform TPP wrapper for LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD has not been implemented \n");
        //exit(-1);
        vnni_output_compute_pixels_bf16_xform_tpp = SCOPEIT(XformExtTPP<T>(compute_pixels, bk, compute_pixels, bk, bk, bk, XformTPP::XFORM_N2V_PAD_TPP, false), XPOSE); /* assuming row-major-ness */
        //vnni_output_compute_pixels_bf16 =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      }
      //upd_remaining_pixels = output_pixels - ((compute_pixels+1)/2)*2;
      //auto zero_unary_shape = libxsmm_create_meltw_unary_shape(bk*upd_remaining_pixels, 1, bk*upd_remaining_pixels, bk*upd_remaining_pixels, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
      //vnni_output_zero_remaining_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, zero_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
      vnni_output_zero_remaining_pixels_bf16_tpp = SCOPEIT(SetZeroTPP<T>(bk*upd_remaining_pixels), EW_ZERO);

    } else { /* for  bf16_use_nchw_format > 0 */
    } /* else-if for bf16_use_nchw_format > 0 */

  } /* if-else over T */

  // JIT requested nested loop specs
  long n_step = 1;
  long c_step = 1;
  long k_step = 1;
  long h_step = 1;
  long w_step = ofw;
  long r_step = 1;
  long s_step = 1;
  long tr_step = 1;

  // Aux steps for linearized algo loops
  long _n_step = 1;
  long _k_step = 1;
  long _c_step = 1;
  long _r_step = 1;
  long _s_step = 1;

/*
  printf("bf16_use_nchw_format     = %d \n", bf16_use_nchw_format);
  printf("bf16_fuse_upd_transposes = %d \n", bf16_fuse_upd_transposes);
  printf("    bf16_acc_nw          = %d \n", bf16_acc_nw);
  printf("    par_over_h_pixels    = %d \n", par_over_h_pixels);
  printf("    pack_input_upfront   = %d \n", pack_input_upfront);
  printf("    use_intermediate_f32_wt_tensor = %d \n",  use_intermediate_f32_wt_tensor);
  printf("      use_hybrid_imgfm_parallelization = %d \n", use_hybrid_imgfm_parallelization);
  printf("      n_img_teams                     = %d \n", n_img_teams);
  printf("      n_ofm_teams                     = %d \n", n_ofm_teams);
  printf("      use_f32_wt_reduction_and_external_wt_vnni = %d \n", use_f32_wt_reduction_and_external_wt_vnni);
  printf("      compute_full_wt_output_block    = %d \n", compute_full_wt_output_block);
*/
  //std::cout << "debug: fm_blocking reduce_work reduce_work_tripcount chunk0 chunk1 = " << fm_blocking << " " <<  reduce_work << " " << reduce_work_tripcount << " " << chunk0 << " " << chunk1 << std::endl;

  //std::cout << "debug: N = nThreads? n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " = " << nThreads << " " << n_step << " " << Cb << " " << c_step << " "
  //                                                                                              << Kb << " " << k_step << " " << ofh << " " << h_step << " "
  //                                                                                              << ofw << " " << w_step << " " << R << " " << r_step << " "
  //                                                                                              << S << " " << s_step << " " << std::endl;

  auto zero_wt_loop = ThreadedLoop<5>({
      LoopSpecs{0, nThreads, 1, false},// true},
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      "Abcde");

  auto conv_bwd_upd_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step},//, true},
      LoopSpecs{0, Cb, c_step},//, true},
      LoopSpecs{0, Kb, k_step},//, true},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},//, true},
      LoopSpecs{0, S, s_step}},//, true}},
      fp32_conv_spec_string);

  auto tr_input_loop = ThreadedLoop<3>({
      LoopSpecs{0, Cb, tr_step},
      LoopSpecs{0, ifhp, tr_step},
      LoopSpecs{0, ifwp, tr_step}},
      "ABC"); // FIXME back to ABC

  auto tr_output_loop  = ThreadedLoop<3>({
      LoopSpecs{0, Kb, tr_step},
      LoopSpecs{0, ofhp, tr_step},
      LoopSpecs{0, ofwp, tr_step}},
      "ABC"); // FIXME back to ABC

  if (sizeof(T) == 2) {
    w_step = 1;
  }

  if (bf16_acc_nw == 1) {
    w_step = ofw;
    h_step = 1;
  }

  auto conv_loop_bf16 = ThreadedLoop<6>({
      LoopSpecs{0, Cb, c_step, false},//true},
      LoopSpecs{0, Kb, k_step, false},//true},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step, false},//true},
      LoopSpecs{0, S, s_step, false}},//true}},
      bf16_conv_spec_string);

  auto reduce_wt_loop = ThreadedLoop<1>({
      LoopSpecs{0, reduce_work_tripcount, 1, false}},//true}},
      "A");

  auto vnni_wt_loop = ThreadedLoop<4>({
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      "ABCD");

  char nchw_format_loop_spec[256];
  auto tr_input_nchw_loop = ThreadedLoop<2>({
      LoopSpecs{0, N, _n_step},
      LoopSpecs{0, Cb, _c_step}},
      "Ab");

  auto tr_output_nchw_loop = ThreadedLoop<2>({
      LoopSpecs{0, N, _n_step},
      LoopSpecs{0, Kb, _k_step}},
      "Ab");

  if (use_hybrid_imgfm_parallelization == 0) {
    //sprintf(nchw_format_loop_spec, "Abcdef");
  } else {
    if (compute_full_wt_output_block > 0) {
      _n_step = N;
    } else {
      _n_step = N/n_img_teams;
    }
  }

  auto conv_loop_bf16_nchw = ThreadedLoop<6>({
      LoopSpecs{0, N, _n_step, true},
      LoopSpecs{0, Cb, _c_step, true},
      LoopSpecs{0, Kb, _k_step, true},
      LoopSpecs{0, n_used_pixels, pixel_blocking},
      LoopSpecs{0, R, _r_step, true},
      LoopSpecs{0, S, _s_step, true}},
      bf16_conv_spec_string);

//  std::cout << "gemm_n gemm_m gemm_k for bwd_upd = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;
//  std::cout << "bn bk bc = " << bn << " " << bk << " " << bc << std::endl;
//  std::cout << "bf16_upfront_trans = " << bf16_upfront_trans << std::endl;
//  std::cout << "use_private_trans = " << use_private_trans << std::endl;
//  std::cout << "use_mb_par = " << use_mb_par << std::endl;
//  std::cout << "par_over_h_pixels = " << par_over_h_pixels << std::endl;

  {
    RECORD_SCOPE(conv_bwd_upd, {});
    {
      if (sizeof(T) == 4) {
        if (use_mb_par_f32 == 0) {
          //printf("Case use_mb_par_f32 == 0 is untested so far!\n"); exit(-1);

          conv_bwd_upd_loop(
            [&](int* ind) {
              int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

              DECL_VLA_PTR_PT_EXT(T,    gradout,   [Kb][ofhp][ofwp][bk],   t_GO, (pad_h_out * ofwp * bk + pad_w_out * bk));
              DECL_VLA_PTR_PT    (T,     inp,      [Cb][ifhp][ifwp][bc],   t_I);
              DECL_VLA_PTR_PT    (T,    weight,    [Cb][R][S][bc][bk],     t_grad_weight);

              if (i_n == 0 && i_w == 0 && i_h == 0) {
                zero_tpp(weight[i_k][i_c][i_r][i_s][0]);
              }
              gemm_as_brgemm_tpp(inp    [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                                 gradout[i_n][i_k][i_h]                 [i_w],
                                 weight [i_k][i_c][i_r][i_s][0],
                                 1, /* brcount */
                                 true);
            },
            [&]() {},
            [&]() {});

        } else { /* else for if (use_mb_par == 0) */
          //printf("Case else for use_mb_par_f32 == 0 is untested so far!\n"); exit(-1);

          zero_wt_loop(
            [&](int* ind) {
              int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];

              DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);

              zero_tpp(scratch[i_n][i_k][i_c][i_r][i_s][0]);
            },
            [&]() {},
            [&]() {});

          conv_bwd_upd_loop(
            [&](int* ind) {
              int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];
              int tid = omp_get_thread_num();

              DECL_VLA_PTR_PT_EXT(T,    gradout,   [Kb][ofhp][ofwp][bk],   t_GO, (pad_h_out * ofwp * bk + pad_w_out * bk));
              DECL_VLA_PTR_PT    (T,    inp,       [Cb][ifhp][ifwp][bc],   t_I);
              DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);

              gemm_as_brgemm_tpp(inp    [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                                 gradout[i_n][i_k][i_h]                 [i_w],
                                 scratch[tid][i_k][i_c][i_r][i_s][0],
                                 1, /* brcount */
                                 true);
            },
            [&]() {},
            [&]() {});

          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];

              DECL_VLA_PTR_PT    (T,    weight2d,  [chunk0],               t_grad_weight);
              DECL_VLA_PTR_PT_EXT_CAST(T,    unsigned char, scratch2d, [chunk0],               t_scratch_experimental, 0);
              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce0_T_tpp( scratch2d[i_n], weight2d[i_n] );
              } else {
                wt_reduce1_T_tpp( scratch2d[i_n], weight2d[i_n] );
              }
            },
            [&]() {},
            [&]() {});
        }

      } else { /* T = bfloat16 goes into else */
        if (bf16_use_nchw_format > 0) {
          //printf("Case bf16_use_nchw_format > 0 is untested so far!\n"); exit(-1);
          if (bf16_fuse_upd_transposes == 0) {
            //printf("Case bf16_fuse_upd_transposes == 0 is untested so far!\n"); exit(-1);
            tr_input_nchw_loop(
              [&](int* ind) {
                int i_n = ind[0], i_c = ind[1];
                DECL_VLA_PTR_PT         (T,                input,                      [Cb][ifhp][ifwp][bc],   t_I);
                DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char, input_mylinearized_pixels,  [Cb][bc][input_pixels], t_scratch_experimental, input_mylinearized_pixels_offset);
                for (int ij = 0; ij < ifhp; ij++) {
                  transpose_input_pixels_bf16_xform_tpp(input[i_n][i_c][ij][0], &input_mylinearized_pixels[i_n][i_c][0][ij*ifwp]);
                }
              },
              [&]() {},
              [&]() {});

            tr_output_nchw_loop(
              [&](int* ind) {
                int i_n = ind[0], i_k = ind[1];
                DECL_VLA_PTR_PT         (T,                gradout,                    [Kb][ofhp][ofwp][bk],   t_GO);
                DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char, output_mylinearized_pixels, [Kb][output_pixels][bk], t_scratch_experimental, output_mylinearized_pixels_offset);

                vnni_output_compute_pixels_bf16_xform_tpp(gradout[i_n][i_k][pad_h][pad_w], output_mylinearized_pixels[i_n][i_k][0]);
                if (upd_remaining_pixels > 0) {
                  printf("Case upd_remaining_pixels > 0 is untested so far!\n"); exit(-1);
                  vnni_output_zero_remaining_pixels_bf16_tpp(output_mylinearized_pixels[i_n][i_k][(compute_pixels+1)/2]);
                }
              },
              [&]() {},
              [&]() {});
          } /* for if bf16_fuse_upd_transposes == 0 */

          if (use_hybrid_imgfm_parallelization == 0) {
            //printf("Case use_hybrid_imgfm_parallelization == 0 is untested so far!\n"); exit(-1);

        conv_loop_bf16_nchw(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], pix = ind[3], i_r = ind[4], i_s = ind[5];
            libxsmm_gemm_param gemm_param;
            libxsmm_meltw_unary_param unary_param;
            int tid = omp_get_thread_num();

            DECL_VLA_PTR_PT         (T,                    gradout,   [Kb][ofhp][ofwp][bk],   t_GO);
            DECL_VLA_PTR_PT         (T,                    input,     [Cb][ifhp][ifwp][bc],   t_I);

            DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, scratch,       [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);
            DECL_VLA_PTR_PT_EXT_CAST(float, unsigned char, scratch_float, [Kb][Cb][R][S][bc][bk], t_scratch_experimental, scratch_float_offset);

            DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, output_mylinearized_pixels, [Kb][output_pixels][bk], t_scratch_experimental, output_mylinearized_pixels_offset);
            DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, input_mylinearized_pixels,  [Cb][bc][input_pixels],  t_scratch_experimental, input_mylinearized_pixels_offset);

            if (bf16_fuse_upd_transposes == 1 && pix == 0 && i_c == 0 && i_r == 0 && i_s == 0) {
              //printf("Case bf16_fuse_upd_transposes == 1 + bunch of conditions is untested so far!\n"); exit(-1);
              vnni_output_compute_pixels_bf16_xform_tpp(gradout[i_n][i_k][pad_h][pad_w], output_mylinearized_pixels[i_n][i_k][0]);
              if (upd_remaining_pixels > 0) {
                printf("Case upd_remaining_pixels > 0 is untested so far!\n");
                exit(-1);
                vnni_output_zero_remaining_pixels_bf16_tpp(output_mylinearized_pixels[i_n][i_k][(compute_pixels+1)/2]);
              }
            }

            if (bf16_fuse_upd_transposes == 1 && pix == 0 && i_k == 0 && i_r == 0 && i_s == 0) {
              //printf("Case bf16_fuse_upd_transposes == 1 + bunch of conditions is untested so far!\n"); exit(-1);
              for (int ij = 0; ij < ifhp; ij++) {
                transpose_input_pixels_bf16_xform_tpp(input[i_n][i_c][ij][0], &input_mylinearized_pixels[i_n][i_c][0][ij*ifwp]);
              }
            }

            if (use_f32_wt_reduction_and_external_wt_vnni > 0) {
              if (pix == 0) {
                zero_float_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0]);
              }
              gemm_kernel_non_hybrid_as_brgemm_tpp(&input_mylinearized_pixels[i_n][i_c][0][pix + i_r * ifwp + i_s],
                                                    output_mylinearized_pixels[i_n][i_k][pix],
                                                    scratch_float[tid][i_k][i_c][i_r][i_s][0],
                                                    1 /* brcount */,
                                                    true);
            } else {
              //printf("Case else for use_f32_wt_reduction_and_external_wt_vnni > 0 is untested so far!\n"); exit(-1);
              gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp(&input_mylinearized_pixels[i_n][i_c][0][pix + i_r * ifwp + i_s],
                                                                  output_mylinearized_pixels[i_n][i_k][pix],
                                                                  scratch[tid][i_k][i_c][i_r][i_s][0],
                                                                  1 /* brcount */,
                                                                  true);
            }
          },
          [&]() {if (sizeof(T) == 2) gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp.config();},
          [&]() {if (sizeof(T) == 2) gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp.release();});

        if (use_f32_wt_reduction_and_external_wt_vnni > 0) {
            //printf("Case use_f32_wt_reduction_and_external_wt_vnni > 0 is untested so far!\n"); exit(-1);

          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float_2d,        [chunk0], t_scratch_experimental,              scratch_float_offset);
              DECL_VLA_PTR_PT_EXT_CAST    (T,     unsigned char, scratch_bf16_weight_2d,  [chunk0], t_scratch_experimental,              scratch_bf16_weight_offset);

              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce0_f32bf16_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
              } else {
                wt_reduce1_f32bf16_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
              }
            },
            [&]() {},
            [&]() {});

          vnni_wt_loop(
            [&](int* ind) {
              int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];
              DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, scratch_bf16_weight, [Cb][R][S][bc][bk], t_scratch_experimental, scratch_bf16_weight_offset);
              DECL_VLA_PTR_PT         (T,                    filter,              [Cb][R][S][bc][bk], t_grad_weight);

              wt_vnni_xform_tpp(scratch_bf16_weight[i_k][i_c][i_r][i_s][0], filter[i_k][i_c][i_r][i_s][0] );
            },
            [&]() {},
            [&]() {});

        } else {
            //printf("Case else for use_f32_wt_reduction_and_external_wt_vnni > 0 is untested so far!\n");
            //exit(-1);
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              DECL_VLA_PTR_PT         (T,                   weight2d,  [chunk0], t_grad_weight);
              DECL_VLA_PTR_PT_EXT_CAST(T,    unsigned char, scratch2d, [chunk0], t_scratch_experimental, 0);

              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce0_bf16bf16_tpp( scratch2d[i_n], weight2d[i_n] );
              } else {
                wt_reduce1_bf16bf16_tpp( scratch2d[i_n], weight2d[i_n] );
              }
            },
            [&]() {},
            [&]() {});
        }

          } else { /* for if use_hybrid_imgfm_parallelization == 0 */
//            printf("Case else for use_hybrid_imgfm_parallelization == 0 is untested so far!\n"); exit(-1);

              DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, output_mylinearized_pixels_dbg, [Kb][output_pixels][bk], t_scratch_experimental, output_mylinearized_pixels_offset);
              DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, input_mylinearized_pixels_dbg,  [Cb][bc][input_pixels],  t_scratch_experimental, input_mylinearized_pixels_offset);

       conv_loop_bf16_nchw(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], pix = ind[3], i_r = ind[4], i_s = ind[5];

            DECL_VLA_PTR_PT         (T,                    filter,    [Cb][R][S][bc][bk],     t_grad_weight);

            DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, scratch,       [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);
            DECL_VLA_PTR_PT_EXT_CAST(float, unsigned char, scratch_float, [Kb][Cb][R][S][bc][bk], t_scratch_experimental, scratch_float_offset);

            DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, output_mylinearized_pixels, [Kb][output_pixels][bk], t_scratch_experimental, output_mylinearized_pixels_offset);
            DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, input_mylinearized_pixels,  [Cb][bc][input_pixels],  t_scratch_experimental, input_mylinearized_pixels_offset);

            int my_col_id;
            unsigned long long brcount = _n_step;

            if (compute_full_wt_output_block == 0) {
              //printf("Case compute_full_wt_output_block == 0 for hybrid is not tested \n"); exit(-1);
              my_col_id = ind[9];
              if (use_f32_wt_reduction_and_external_wt_vnni > 0) {
                if (pix == 0) {
                  zero_float_tpp(scratch_float[my_col_id][i_k][i_c][i_r][i_s][0]);
                }
                brgemm_kernel_hybrid_tpp(&input_mylinearized_pixels[i_n][i_c][0][pix + i_r * ifwp + i_s],
                                          output_mylinearized_pixels[i_n][i_k][pix],
                                          scratch_float[my_col_id][i_k][i_c][i_r][i_s][0],
                                          brcount /* brcount */,
                                          true);
              } else {
                brgemm_kernel_hybrid_zerobeta_cvnni_tpp(&input_mylinearized_pixels[i_n][i_c][0][pix + i_r * ifwp + i_s],
                                                         output_mylinearized_pixels[i_n][i_k][pix],
                                                         scratch[my_col_id][i_k][i_c][i_r][i_s][0],
                                                         brcount /* brcount */,
                                                         true);
              } /* if-else for use_f32_wt_reduction_and_external_wt_vnni > 0 */
            } else {
              //printf("Case else for compute_full_wt_output_block == 0 for hybrid is not tested \n"); exit(-1);
              brgemm_kernel_hybrid_zerobeta_cvnni_tpp(&input_mylinearized_pixels[i_n][i_c][0][pix + i_r * ifwp + i_s],
                                                       output_mylinearized_pixels[i_n][i_k][pix],
                                                       filter[i_k][i_c][i_r][i_s][0],
                                                       brcount /* brcount */,
                                                       true);
            }
          },
          [&]() {if (sizeof(T) == 2) brgemm_kernel_hybrid_zerobeta_cvnni_tpp.config();},
          [&]() {if (sizeof(T) == 2) brgemm_kernel_hybrid_zerobeta_cvnni_tpp.release();});

        if (use_f32_wt_reduction_and_external_wt_vnni > 0) {
          //printf("Case use_f32_wt_reduction_and_external_wt_vnni > 0 for hybrid is not tested \n"); exit(-1);
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float_2d,        [chunk0], t_scratch_experimental,              scratch_float_offset);
              DECL_VLA_PTR_PT_EXT_CAST    (T,     unsigned char, scratch_bf16_weight_2d,  [chunk0], t_scratch_experimental,              scratch_bf16_weight_offset);

              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce0_f32bf16_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
              } else {
                wt_reduce1_f32bf16_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
              }
            },
            [&]() {},
            [&]() {});

          vnni_wt_loop(
            [&](int* ind) {
              int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];
              DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, scratch_bf16_weight, [Cb][R][S][bc][bk], t_scratch_experimental, scratch_bf16_weight_offset);
              DECL_VLA_PTR_PT         (T,                    filter,              [Cb][R][S][bc][bk], t_grad_weight);

              wt_vnni_xform_tpp(scratch_bf16_weight[i_k][i_c][i_r][i_s][0], filter[i_k][i_c][i_r][i_s][0] );
            },
            [&]() {},
            [&]() {});
        } else if (compute_full_wt_output_block == 0) {
          //printf("Case compute_full_wt_output_block == 0 for hybrid is not tested \n"); exit(-1);
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              DECL_VLA_PTR_PT         (T,                   weight2d,  [chunk0], t_grad_weight);
              DECL_VLA_PTR_PT_EXT_CAST(T,    unsigned char, scratch2d, [chunk0], t_scratch_experimental, 0);
              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce0_bf16bf16_tpp( scratch2d[i_n], weight2d[i_n] );
              } else {
                wt_reduce1_bf16bf16_tpp( scratch2d[i_n], weight2d[i_n] );
              }
            },
            [&]() {},
            [&]() {});
        }

          } /* if-else for use_hybrid_imgfm_parallelization */
        } else if (bf16_use_chwn_format > 0) { /* for if bf16_use_nchw_format > 0 */
          //printf("Case bf16_use_chwn_format > 0 is untested so far!\n"); exit(-1);
          if (use_private_trans == 0) {
            //printf("Case use_private_trans == 0 is untested so far!\n"); exit(-1);
            tr_input_loop(
              [&](int* ind) {
                int i_c = ind[0], i_h = ind[1], i_w = ind[2];

                DECL_VLA_PTR_PT    (T,    input,          [Cb]  [ifhp][ifwp][bc],   t_I);
                DECL_VLA_PTR_PT_EXT_CAST(T,  unsigned char,  tr_input,       [ifhp][ifwp][bc]  [bn],   t_scratch_experimental, tr_input_offset);

                trans_xform_tpp(input[0][i_c][i_h][i_w], tr_input[i_c][i_h][i_w][0]);
              },
              [&]() {},
              [&]() {});

            tr_output_loop(
              [&](int* ind) {
                int i_k = ind[0], i_h = ind[1], i_w = ind[2];

                DECL_VLA_PTR_PT    (T,    output,         [Kb]  [ofhp][ofwp][bk],   t_GO);
                DECL_VLA_PTR_PT_EXT_CAST (T, unsigned char,    tr_output,      [ofhp][ofwp][bn]  [bk],   t_scratch_experimental, tr_output_offset);

                vnni_xform_tpp(output[0][i_k][i_h][i_w], tr_output[i_k][i_h][i_w][0]);
              },
              [&]() {},
              [&]() {});
          } else { /* dummy else for if use_private_trans == 0 */
            //printf("Case else use_private_trans == 0 is untested so far!\n"); exit(-1);
          }

          if (par_over_h_pixels > 0) {
            zero_wt_loop(
              [&](int* ind) {
                int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];
                DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float, [Kb][Cb][R][S][bc][bk], t_scratch_experimental, scratch_float_offset);

                zero_float_tpp(scratch_float[i_n][i_k][i_c][i_r][i_s][0]);
              },
              [&]() {},
              [&]() {});
          }

          if (use_private_trans > 0) {
            memset(trans_tracker.get(), 0, trans_tracker_size*nThreads*sizeof(int));
          }

          conv_loop_bf16(
            [&](int* ind) {
              int i_c = ind[0], i_k = ind[1], i_h = ind[2], i_w = ind[3], i_r = ind[4], i_s = ind[5];
              int tid = omp_get_thread_num();

              DECL_VLA_PTR_PT    (T,     filter,        [Cb][R][S][bc][bk],       t_grad_weight);
              DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float,     [Kb][Cb][R][S][bc][bk], t_scratch_experimental, scratch_float_offset);

              if (i_h == 0 && i_w == 0 && par_over_h_pixels == 0) {
                zero_float_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0]);
              }

              if (use_private_trans > 0) {

                DECL_VLA_PTR_PT    (T,     input,             [Cb][ifhp][ifwp][bc],   t_I);
                DECL_VLA_PTR_PT    (T,     output,            [Kb][ofhp][ofwp][bk],   t_GO);

                DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,     private_tr_input,  [ifhp][ifwp][bc][bn],   t_scratch_experimental, private_tr_input_offset  + tid*(N*ifhp*ifwp*C) * sizeof(T));
                DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,     private_tr_output, [ofhp][ofwp][bn][bk],   t_scratch_experimental, private_tr_output_offset + tid*(N*ofhp*ofwp*K) * sizeof(T));

                int *inp_loc = (int*) trans_tracker.get() + tid * trans_tracker_size + i_c;
                int *out_loc = (int*) trans_tracker.get() + tid * trans_tracker_size + Cb + i_k;

                int is_inp_trans = *inp_loc;
                int is_out_trans = *out_loc;

                if (is_inp_trans == 0) {
                  for (int _ih = 0; _ih < ifhp; _ih++) {
                    for (int _iw = 0; _iw < ifwp; _iw++) {
                      trans_xform_tpp(input[0][i_c][_ih][_iw], private_tr_input[i_c][_ih][_iw][0]);
                    }
                  }
                  *inp_loc = 1;
                }

                if (is_out_trans == 0) {
                  for (int _ih = 0; _ih < ofhp; _ih++) {
                    for (int _iw = 0; _iw < ofwp; _iw++) {
                      vnni_xform_tpp(output[0][i_k][_ih][_iw], private_tr_output[i_k][_ih][_iw][0]);
                    }
                  }
                  *out_loc = 1;
                }

                brgemm_acc_pixel_tpp(private_tr_input  [i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][0],
                                     private_tr_output [i_k][i_h + pad_h_out]     [i_w + pad_w_out]     [0],
                                     scratch_float     [tid][i_k][i_c][i_r][i_s][0],
                                     w_step * h_step, /* brcount */
                                     true);
              } else { /* for if use_private_trans > 0 */

                DECL_VLA_PTR_PT_EXT_CAST    (T, unsigned char,     tr_input,      [ifhp][ifwp][bc]  [bn],   t_scratch_experimental, tr_input_offset);
                DECL_VLA_PTR_PT_EXT_CAST    (T, unsigned char,     tr_output,     [ofhp][ofwp][bn]  [bk],   t_scratch_experimental, tr_output_offset);

                brgemm_acc_pixel_tpp(tr_input     [i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][0],
                                     tr_output    [i_k][i_h + pad_h_out]     [i_w + pad_w_out]     [0],
                                     scratch_float[tid][i_k][i_c][i_r][i_s][0],
                                     w_step * h_step, /* brcount */
                                     true);
              } /* for if-else use_private_trans > 0 */

              if ((i_h == ofh - h_step) && (i_w == ofw - w_step) && (par_over_h_pixels == 0)) {
                fp32bf16_cvt_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0], (T*)(scratch_float[tid][i_k][i_c][i_r][i_s][0]));

                wt_vnni_xform_tpp((T*)(scratch_float[tid][i_k][i_c][i_r][i_s][0]), filter[i_k][i_c][i_r][i_s][0] );
              }

            },
            [&]() {if (sizeof(T) == 2) brgemm_acc_pixel_tpp.config();},
            [&]() {if (sizeof(T) == 2) brgemm_acc_pixel_tpp.release();});
          /* end of gemm_loop_bf16 definition */

          if (par_over_h_pixels > 0) {

            reduce_wt_loop(
              [&](int* ind) {
                int i_n = ind[0];

                DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float_2d,        [chunk0], t_scratch_experimental,              scratch_float_offset);
                DECL_VLA_PTR_PT_EXT_CAST    (T,     unsigned char, scratch_bf16_weight_2d,  [chunk0], t_scratch_experimental,              scratch_bf16_weight_offset);

                if (i_n < reduce_work_tripcount - 1) {
                  wt_reduce0_float_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
                } else {
                  wt_reduce1_float_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
                }
              },
              [&]() {},
              [&]() {});

            vnni_wt_loop(
              [&](int* ind) {
                int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];

                DECL_VLA_PTR_PT_EXT_CAST    (T,     unsigned char, scratch_bf16_weight,  [Cb][R][S][bc][bk], t_scratch_experimental,              scratch_bf16_weight_offset);
                DECL_VLA_PTR_PT    (T,     filter,              [Cb][R][S][bc][bk],     t_grad_weight);

                wt_vnni_xform_tpp(scratch_bf16_weight[i_k][i_c][i_r][i_s][0], filter[i_k][i_c][i_r][i_s][0] );
              },
              [&]() {},
              [&]() {});

          } /* for if par_over_h_pixels > 0 */
        } /* if-else for bf16_use_nchw_format/bf16_use_chwn_format */
      } /* if-else over T */
    } /* end of the scope with recorded parallel for */
  } /* end of the conv_bwd_upd scope */

//return std::vector<at::Tensor>({t_grad_input, t_grad_weight});

  long Kb_step = Kb;

  long avoid_rim_fmas = 0;
  long non_1x1_with_strides = 0;
  if (ofh <= 7 && ofw <=7 && R == 3 && S == 3 && stride_w == 1 && stride_h == 1) {
    avoid_rim_fmas = 1;
  }

  if ((R != 1 && stride_h != 1) ||
      (S != 1 && stride_w != 1)) {
    non_1x1_with_strides = 1;
  }

  //auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bc, dtype, dtype, dtype);
  //if (dtype == LIBXSMM_DATATYPE_F32) {
  //  wt_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  //} else {
  //  wt_trans_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI2_TO_VNNI2T, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
  //}

  SCOPEIT_DECL(XformExtTPP<T>) wt_trans_tpp;

  if (sizeof(T) == 4)
    wt_trans_tpp = SCOPEIT(XformExtTPP<T>(bc, bk, bk, bc, XformTPP::XFORM_XPOSE_TPP, false), XPOSE); /* assuming row-major-ness */
  else
    wt_trans_tpp = SCOPEIT(XformExtTPP<T>(bc, bk, bk, bc, XformTPP::XFORM_XPOSE_V2V_TPP, false), XPOSE); /* assuming row-major-ness */

  n_step = 1;
  c_step = 1;
  k_step = Kb_step;
  h_step = 1;
  w_step = ofw;
  r_step = R;
  s_step = S;

  if ((avoid_rim_fmas == 1) || (non_1x1_with_strides == 1)) {
    r_step = 1;
    s_step = 1;
  }

  gemm_n = ofw;
  gemm_m = bc;
  gemm_k = bk;

  //std::cout << "gemm_n gemm_m gemm_k for bwd_d = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;
  //std::cout << "avoid_rim_fmas, non_1x1_with_strides = " << avoid_rim_fmas << " " << non_1x1_with_strides << std::endl;

  std::unique_ptr<unsigned long long[]> A_offsets, B_offsets;

  decltype(zero_tpp) zero_rim_tpp, zero_all_pixels_tpp, zero_bc_tpp;
  decltype(gemm_as_brgemm_tpp) brgemm_tpp, brgemm2_tpp;

  //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bc*ifwp, 1, bc*ifwp, bc*ifwp, dtype, dtype, dtype);
  //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  zero_rim_tpp = SCOPEIT(SetZeroTPP<T>(bc*ifwp), EW_ZERO);
  //l_unary_shape = libxsmm_create_meltw_unary_shape(bc*ifwp*ifhp, 1, bc*ifwp*ifhp, bc*ifwp*ifhp, dtype, dtype, dtype);
  //zero_kernel_all_pixels = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
  zero_all_pixels_tpp = SCOPEIT(SetZeroTPP<T>(bc*ifwp*ifhp), EW_ZERO);

  if ((R == 1 && S == 1) ||
      (avoid_rim_fmas == 1) ||
      (non_1x1_with_strides == 1)) {
    //auto gemm_n = ofw;
    //auto gemm_m = bc;
    //auto gemm_k = bk;

    //l_unary_shape = libxsmm_create_meltw_unary_shape(bc, 1, bc, bc, dtype, dtype, dtype);
    //zero_kernel_bc = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);  
    zero_bc_tpp = SCOPEIT(SetZeroTPP<T>(bc), EW_ZERO);

    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bk*ofhp*ofwp*sizeof(DType), Kb_step );
    //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, bk*ofhp*ofwp, R*S*bc*bk, bk, bc, bc*stride_w, 1.0, 0, 0)));//, BRGEMM);
    //l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n-1, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    //brgemm_kernel2.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm2_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-1, gemm_m, gemm_k, bk*ofhp*ofwp, R*S*bc*bk, bk, bc, bc*stride_w, 1.0, 0, 0)));//, BRGEMM);

  } else {

    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bc, bk, stride_w*bc, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_OFFSET, 0, 0, 0 );
    //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* no strides due to reduce_offset */ bk, bc, bc*stride_w, 1.0, 0, 0)));//, BRGEMM);

    A_offsets = std::make_unique<unsigned long long[]>(Kb * R * S);
    B_offsets = std::make_unique<unsigned long long[]>(Kb * R * S);

    // Prepare offset array
    unsigned long long i = 0;
    for (long ifm = 0; ifm < Kb_step; ifm++) {
      for (long kj = 0; kj < R; kj++) {
        for (long ki = 0; ki < S; ki++) {
          A_offsets[i] = (ifm * R * S * bc * bk +
              kj * S * bc * bk +
              ki * bc * bk) * sizeof(T);
          B_offsets[i] = (ifm * ofhp * ofwp * bk +
              kj * ofwp * bk +
              ki * bk) * sizeof(T);
          i++;
        }
      }
    } /* outer loop for filling the offsets */
  } /* if-else over the datatype T */

  auto wt_trans_loop = ThreadedLoop<4>({
      LoopSpecs{0, Kb, 1, false}, // true},
      LoopSpecs{0, Cb, 1, false},//, true},
      LoopSpecs{0, R, 1, false},//, true},
      LoopSpecs{0, S, 1, false}},//, true}},
      "ABCD");

  /* FIXME: Fix this! */
  char loop_specs_str[256] = "Abcdefg";

  auto conv_bwd_d_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, false},// true},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, Kb, k_step, false},//, true},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      loop_specs_str);


  {
    RECORD_SCOPE(conv_bwd_d, {});
    {

      wt_trans_loop(
        [&](int* ind) {
          int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];

          DECL_VLA_PTR_PT    (T,    weight,    [Cb][R][S][bc][bk], t_W);
          DECL_VLA_PTR_PT    (T,    weight_tr, [Kb][R][S][bk][bc], t_WT);

          wt_trans_tpp(weight[i_k][i_c][i_r][i_s][0], weight_tr[i_c][i_k][R-1-i_r][S-1-i_s][0]);
        },
        [&]() {},
        [&]() {});

      conv_bwd_d_loop(
        [&](int* ind) {
          int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

          DECL_VLA_PTR_PT    (T,    gradout,     [Kb][ofhp][ofwp][bk],   t_GO);
          DECL_VLA_PTR_PT_EXT(T,    gradout_off, [Kb][ofhp][ofwp][bk],   t_GO, (pad_h_out * ofwp * bk + pad_w_out * bk));
          DECL_VLA_PTR_PT    (T,    dinp,        [Cb][ifhp][ifwp][bc],   t_grad_input);
          DECL_VLA_PTR_PT_EXT(T,    dinp_off,    [Cb][ifhp][ifwp][bc],   t_grad_input, (pad_h_in * ifwp * bc + pad_w_in * bc));
          DECL_VLA_PTR_PT    (T,    weight_tr,   [Kb][R][S][bk][bc],     t_WT);

          if (avoid_rim_fmas == 0) {
            if (non_1x1_with_strides == 0) {
              if (i_k == 0 && i_r == 0 && i_s == 0) {
                if (stride_h != 1) {
                  if (i_w == 0 && i_h == 0) {
                    zero_all_pixels_tpp(dinp[i_n][i_c][0][0]);
                  }
                } else {
                  zero_rim_tpp(dinp_off[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
                }
              }

              brgemm_tpp(gradout  [i_n][i_k][i_h][i_w],
                         weight_tr[i_c][i_k][i_r][i_s][0],
                         dinp_off [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         B_offsets.get(), A_offsets.get(),
                         Kb_step * r_step * s_step,
                         true);
            } else { /* for non_1x1_with_strides == 0 */
              if (i_k == 0 && i_r == 0 && i_s == 0) {
                if (stride_h != 1) {
                  if (i_w == 0 && i_h == 0) {
                    zero_all_pixels_tpp(dinp[i_n][i_c][0][0]);
                  }
                } else {
                  zero_rim_tpp(dinp[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
                }
              }

              brgemm_tpp(gradout_off[i_n][i_k][i_h][i_w],
                         weight_tr  [i_c][i_k][R-1-i_r][S-1-i_s][0],
                         dinp       [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         Kb_step * r_step * s_step,
                         true);

              /* Zero Rim..  */
              if (i_r == R-1 && i_s == S-1 && i_h == ofh-h_step && i_w == ofw-w_step && i_k == Kb - Kb_step) {
                for (int ij = 0; ij < ifhp; ij++) {
                  for (int ii = 0; ii < ifwp; ii++) {
                    if ((ij < pad_h_in || ij >= ifh + pad_h_in) ||
                        (ii < pad_w_in || ii >= ifw + pad_w_in)) {
                      zero_bc_tpp(dinp[i_n][i_c][ij][ii]);
                    }
                  }
                }
              }
            } /* else-if for non_1x1_with_strides == 0 */
          } else { /* avoid_rim_fmas == 0 */
            if (i_k == 0 && i_r == 0 && i_s == 0) {
              zero_rim_tpp(dinp_off[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
            }
            if (i_r == 0 && i_h == 0) {
              /* Do no FLOPS  */
            } else if (i_r == R-1 && i_h == ofh-1 ) {
              /* Do no FLOPS  */
            } else if ( i_w == 0 && i_s == 0 ) {
              brgemm2_tpp(gradout  [i_n][i_k][i_h + i_r][i_w + i_s + 1],
                          weight_tr[i_c][i_k][i_r][i_s][0],
                          dinp_off [i_n][i_c][i_h][i_w + 1],
                          Kb_step,
                          true);
            } else if ( i_w + w_step == ofw  && i_s == S-1) {
              brgemm2_tpp(gradout  [i_n][i_k][i_h + i_r][i_w + i_s],
                          weight_tr[i_c][i_k][i_r][i_s][0],
                          dinp_off [i_n][i_c][i_h][i_w],
                          Kb_step,
                          true);
            } else {
              brgemm_tpp(gradout  [i_n][i_k][i_h + i_r][i_w + i_s],
                         weight_tr[i_c][i_k][i_r][i_s][0],
                         dinp_off [i_n][i_c][i_h][i_w],
                         Kb_step,
                         true);
            }
          } /* else-if for avoid_rim_fmas */
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});

    } /* end of the scope with recorded parallel for */
  } /* end of the conv_bwd_d scope */

//#endif

} /* end of the dummy scope */


//auto t_dummy     = at::empty({0},  torch::TensorOptions().dtype(at::kFloat));
//return std::vector<at::Tensor>({t_dummy, t_grad_weight});
return std::vector<at::Tensor>({t_grad_input, t_grad_weight});

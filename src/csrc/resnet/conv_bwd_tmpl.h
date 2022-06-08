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

//std::cout << "t_I sizes = " << t_I.sizes() << std::endl;
//std::cout << "output_size = " << output_size << std::endl;
//std::cout << "CP Cb bc Kb bk = " << CP << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;
//std::cout << "weight_tr_size = " << weight_tr_size << std::endl;

auto t_grad_input  = at::empty(t_I.sizes(), torch::TensorOptions().dtype(t_I.dtype()));
auto t_grad_weight = at::empty(t_W.sizes(), torch::TensorOptions().dtype(t_W.dtype()));
auto t_WT          = at::empty(weight_tr_size, torch::TensorOptions().dtype(t_W.dtype()));

#if 1
//------------------------------------

  long  pad_h = pad_h_out;
  long  pad_w = pad_w_out;

  /* Some algorithmic knobs  */
  /* Uses parallelism in the MB dimension for f32 precision */
  long use_mb_par_f32 = 1;

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
  long use_hybrid_imgfm_parallelization = 0;
  long n_img_teams = 7;
  long n_ofm_teams = 4;
  long weight_copies = 0;
  long multiple_target = 2;
  long max_compute_offset_input = 0;
  long use_f32_wt_reduction_and_external_wt_vnni = 0; //0; FIXME back
  long compute_full_wt_output_block = 0;

  bf16_use_chwn_format = (bf16_use_nchw_format > 0) ? 0 : 1;
  use_private_trans = bf16_fuse_upd_transposes;

//  long bn = N; // declared earlier
//------------------------------------
#endif


//  DType *scratch_libxsmm = (DType*)libxsmm_aligned_malloc( nThreads*C*K*R*S*sizeof(DType), 2097152);
long long int running_scratch_size_in_bytes = 0;

std::vector<long> scratch_size{nThreads, C, K, R, S};

#if 0
at::Tensor t_scratch;
if (sizeof(T) == 4)
  t_scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kFloat));
else /* bfloat16 */
  t_scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kBFloat16)); /* Hopefully, not a problem */
#endif

running_scratch_size_in_bytes += nThreads * C * K * R * S * sizeof(T);

long long int private_tr_input_offset = 0, private_tr_output_offset = 0,
              tr_input_offset = 0, tr_output_offset = 0,
              scratch_float_offset = 0, scratch_bf16_weight_offset = 0,
              input_mylinearized_pixels_offset = 0, output_mylinearized_pixels_offset = 0;
//at::Tensor t_private_tr_input, t_private_tr_output;
if (sizeof(T) == 2) {
#if 0
  t_private_tr_input  = at::empty({nThreads, N, ifhp, ifwp, C}, torch::TensorOptions().dtype(at::kBFloat16));
  t_private_tr_output = at::empty({nThreads, N, ofhp, ofwp, K}, torch::TensorOptions().dtype(at::kBFloat16));
#endif
  //DType **private_tr_input_libxsmm  = (DType**)libxsmm_aligned_malloc( nThreads*sizeof(DType*), 2097152);
  //DType **private_tr_output_libxsmm = (DType**)libxsmm_aligned_malloc( nThreads*sizeof(DType*), 2097152);
  //for (int thr = 0; thr < nThreads; thr++) {
  //  private_tr_input_libxsmm[thr] = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  //  private_tr_output_libxsmm[thr] = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  //}
  private_tr_input_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += nThreads * N * ifhp * ifwp * C * sizeof(T);
  private_tr_output_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += nThreads * N * ofhp * ofwp * K * sizeof(T);
}

//at::Tensor t_tr_input, t_tr_output;
if (sizeof(T) == 2) {
#if 0
  //DType *tr_input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  //DType *tr_output_libxsmm = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  t_tr_input  = at::empty({N, ifhp, ifwp, C}, torch::TensorOptions().dtype(at::kBFloat16));
  t_tr_output = at::empty({N, ofhp, ofwp, K}, torch::TensorOptions().dtype(at::kBFloat16));
#endif
  tr_input_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += N * ifhp * ifwp * C * sizeof(T);
  tr_output_offset = running_scratch_size_in_bytes;
  running_scratch_size_in_bytes += N * ofhp * ofwp * K * sizeof(T);
}

//at::Tensor t_scratch_float, t_scratch_bf16_weight;
if (sizeof(T) == 2) {
#if 0
  //float *scratch_libxsmm = (float*)libxsmm_aligned_malloc( nThreads*C*K*R*S*sizeof(float), 2097152);
  //libxsmm_bfloat16 *scratch_libxsmm_bf16_weights = (libxsmm_bfloat16*)libxsmm_aligned_malloc(C*K*R*S*sizeof(libxsmm_bfloat16), 2097152);
  t_scratch_float       = at::empty({nThreads, C, K, R, S}, torch::TensorOptions().dtype(at::kFloat));
  t_scratch_bf16_weight = at::empty({C, K, R, S},           torch::TensorOptions().dtype(at::kBFloat16));
#endif
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

//if (t_scratch.numel() <= 1) {

//return std::vector<at::Tensor>({t_grad_input, t_grad_weight});

{ /* main dummy scope */


#if 1
#else
  long use_mb_par = 1;
#endif

  long fm_blocking = (bk % 16 == 0) ? 16 : bk;
  long reduce_work = Kb * C * R * S * (bk/fm_blocking);
  long reduce_chunk_size = (reduce_work + nThreads - 1)/nThreads;
  long reduce_work_tripcount = (reduce_work + reduce_chunk_size - 1) / reduce_chunk_size;

  long chunk0 = reduce_chunk_size * fm_blocking;
  long chunk1 = K * C * R * S  - (reduce_work_tripcount-1) * chunk0;
  chunk1 = (chunk1 <= 0) ? chunk0 : chunk1;

  int gemm_m = 0, gemm_n = 0, gemm_k = 0;

#if 1
  char bf16_conv_spec_string[256];
  char fp32_conv_spec_string[256];

  sprintf(fp32_conv_spec_string, "Abcdefg");
  sprintf(bf16_conv_spec_string, "Abcdef");
/*
  if (sizeof(DType) == 4) {
    sprintf(fp32_conv_spec_string, "%s", loop_specs_str);
    sprintf(bf16_conv_spec_string, "Abcdef");
  } else {
    sprintf(fp32_conv_spec_string, "Abcdefg");
    sprintf(bf16_conv_spec_string, "%s", loop_specs_str);
  }
*/

  if (use_hybrid_imgfm_parallelization > 0) {
    bf16_fuse_upd_transposes = 0;
    weight_copies = n_img_teams;
  } else {
    weight_copies = nThreads;
  }
#else
  /* Some algorithmic decisions  */
  long bf16_upfront_trans = 1;
  long bf16_acc_nw = 1;
  long par_over_h_pixels = 1;
  long use_private_trans = 0;

  char bf16_conv_spec_string[256];
  int use_h_par_bf16 = 0;
  const char* const env_h_par_str = getenv("USE_H_PAR_BF16");
  if (0 == env_h_par_str) {
    use_h_par_bf16 = 0;
  } else {
    use_h_par_bf16 = atoi(env_h_par_str);
  }

  if (use_h_par_bf16 > 0) {
    sprintf(bf16_conv_spec_string,"ABEFCd");
  } else {
    //sprintf(bf16_conv_spec_string,"A{C:7}B{R:8}EFcd"); hardcoded for 56 threads
    sprintf(bf16_conv_spec_string,"ABEFcd");
  }
  par_over_h_pixels = use_h_par_bf16;
#endif

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

#if 1

  // TPP kernels that may be used
  libxsmm_meltwfunction_unary zero_kernel;
  libxsmm_meltwfunction_unary zero_kernel_bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_f32;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_f32;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_f32bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_f32bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_bf16bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_bf16bf16;
  libxsmm_meltwfunction_unary trans_xform_kernel;
  libxsmm_meltwfunction_unary vnni_xform_kernel;
  libxsmm_meltwfunction_unary fp32bf16_cvt_kernel;
  libxsmm_meltwfunction_unary wt_vnni_kernel;
  libxsmm_meltwfunction_unary vnni_output_compute_pixels_bf16;
  libxsmm_meltwfunction_unary vnni_output_zero_remaining_pixels_bf16;
  libxsmm_meltwfunction_unary transpose_input_pixels_bf16;

  libxsmm_xmmfunction tileconfig_kernel;
  libxsmm_xmmfunction tilerelease_kernel;
  libxsmm_xmmfunction gemm_kernel;
  libxsmm_xmmfunction brgemm_kernel_acc_pixel;
  libxsmm_xmmfunction gemm_kernel_non_hybrid;
  libxsmm_xmmfunction gemm_kernel_non_hybrid_zerobeta_cvnni;
  libxsmm_xmmfunction brgemm_kernel_hybrid;
  libxsmm_xmmfunction brgemm_kernel_hybrid_zerobeta_cvnni;



  typedef T DType;

  DType *input_linearized_pixels = NULL, *output_linearized_pixels = NULL;
{
  
  // Setup basic GEMM flags
  auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto l_tc_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto l_tr_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto dtype      = (sizeof(DType) == 2) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;

  auto l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, weight_copies, K * C *R * S, chunk0, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
  wt_reduce_kernel0_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
  l_unary_shape.m         = chunk1;
  l_unary_shape.ldo       = chunk1;
  wt_reduce_kernel1_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;

  l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, weight_copies, K * C *R * S, chunk0, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32);
  wt_reduce_kernel0_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
  l_unary_shape.m         = chunk1;
  l_unary_shape.ldo       = chunk1;
  wt_reduce_kernel1_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;

  l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, weight_copies, K * C *R * S, chunk0, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_F32);
  wt_reduce_kernel0_bf16bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
  l_unary_shape.m         = chunk1;
  l_unary_shape.ldo       = chunk1;
  wt_reduce_kernel1_bf16bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;

  // Configure zero TPP kernels
  l_unary_shape = libxsmm_create_meltw_unary_shape(bk*bc, 1, bk*bc, bk*bc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
  zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  l_unary_shape = libxsmm_create_meltw_unary_shape(bk*bc, 1, bk*bc, bk*bc, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
  zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  // Generate XForm TPP kernels
  auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, bn, C*ifhp*ifwp, bn, dtype, dtype, dtype);
  trans_xform_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, K*ofhp*ofwp, bk, dtype, dtype, dtype);
  vnni_xform_kernel =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );

  tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bk, dtype, dtype, dtype);
  wt_vnni_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  // Generate f32->bf16 cvt TPP kernel
  l_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bk, LIBXSMM_DATATYPE_F32, dtype, LIBXSMM_DATATYPE_F32);
  fp32bf16_cvt_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  if (sizeof(DType) == 4) {
    auto gemm_n = bc;
    auto gemm_m = bk;
    auto gemm_k = ofw;
    auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
  } else {
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
        auto new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
        auto new_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
        auto new_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
        if (use_intermediate_f32_wt_tensor == 0) {
          new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        }
        printf("for gemm_kernel_non_hybrid\n");
        printf("new_shape = %d %d %d %d %d %d\n", new_shape.m, new_shape.n, new_shape.k, new_shape.lda, new_shape.ldb, new_shape.ldc);
        printf("new flags = %d \n", new_flags);
        gemm_kernel_non_hybrid.gemm = libxsmm_dispatch_gemm_v2( new_shape, new_flags, new_prefetch_flags );
        
        new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, dtype, dtype);
        new_flags |=  LIBXSMM_GEMM_FLAG_BETA_0  | LIBXSMM_GEMM_FLAG_VNNI_C;
//        new_flags |=  LIBXSMM_GEMM_FLAG_BETA_0;
        printf("new_shape = %d %d %d %d %d %d\n", new_shape.m, new_shape.n, new_shape.k, new_shape.lda, new_shape.ldb, new_shape.ldc);
        printf("new flags = %d \n", new_flags);
        gemm_kernel_non_hybrid_zerobeta_cvnni.gemm      = libxsmm_dispatch_gemm_v2( new_shape, new_flags, new_prefetch_flags );
        tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( new_shape, l_tc_flags, new_prefetch_flags );
        tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( new_shape, l_tr_flags, new_prefetch_flags );
      } else {
        long stride_a = K * output_pixels * sizeof(DType);
        long stride_b = C * input_pixels * sizeof(DType);
        auto new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
        auto new_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
        auto new_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
        if (use_intermediate_f32_wt_tensor == 0) {
          new_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
        }
        auto new_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, stride_a, stride_b, 0 );
        brgemm_kernel_hybrid.gemm   = libxsmm_dispatch_brgemm_v2( new_shape, new_flags, new_prefetch_flags, new_brconfig );

        new_shape = libxsmm_create_gemm_shape( bk, bc, pixel_blocking, bk, input_pixels, bk, dtype, dtype, dtype, dtype);
        new_flags |=  LIBXSMM_GEMM_FLAG_BETA_0  | LIBXSMM_GEMM_FLAG_VNNI_C;
        brgemm_kernel_hybrid_zerobeta_cvnni.gemm   = libxsmm_dispatch_brgemm_v2( new_shape, new_flags, new_prefetch_flags, new_brconfig );

        tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( new_shape, l_tc_flags, new_prefetch_flags );
        tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( new_shape, l_tr_flags, new_prefetch_flags );
      }
      input_linearized_pixels  = (DType*)libxsmm_aligned_malloc( N*input_pixels*C*sizeof(DType), 2097152);
      output_linearized_pixels = (DType*)libxsmm_aligned_malloc( N*output_pixels*K*sizeof(DType), 2097152);
      auto new_tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, ifwp, bc, input_pixels, dtype, dtype, dtype);
      transpose_input_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      new_tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, compute_pixels, bk, bk, dtype, dtype, dtype);
      if ((ofhp * ofwp) % 2 == 0) {
        vnni_output_compute_pixels_bf16 =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      } else {
        vnni_output_compute_pixels_bf16 =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      }
      upd_remaining_pixels = output_pixels - ((compute_pixels+1)/2)*2;
      auto zero_unary_shape = libxsmm_create_meltw_unary_shape(bk*upd_remaining_pixels, 1, bk*upd_remaining_pixels, bk*upd_remaining_pixels, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
      vnni_output_zero_remaining_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, zero_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    } else {
      auto gemm_n = bc;
      auto gemm_m = bk;
      auto gemm_k = bn;
      auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bn, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
      auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
      auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bn*bk*sizeof(DType), stride_w*bc*bn*sizeof(DType), 0 );
      tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
      tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
      gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
      brgemm_kernel_acc_pixel.gemm  = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    }
  }
}

printf("Set the libxsmm kernels\n");
#endif

//#define DEBUGGING

#ifdef DEBUGGING
  libxsmm_xmmfunction tileconfig_kernel;
  libxsmm_xmmfunction tilerelease_kernel;
  libxsmm_xmmfunction gemm_kernel;
  libxsmm_xmmfunction brgemm_kernel_acc_pixel;
  libxsmm_meltwfunction_unary zero_kernel;
  libxsmm_meltwfunction_unary zero_kernel_bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_f32;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_f32;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_f32bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_f32bf16;
  libxsmm_meltwfunction_unary trans_xform_kernel;
  libxsmm_meltwfunction_unary vnni_xform_kernel;
  libxsmm_meltwfunction_unary fp32bf16_cvt_kernel;
  libxsmm_meltwfunction_unary wt_vnni_kernel;

  {
    typedef T DType;

  auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto l_tc_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto l_tr_flags = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) : LIBXSMM_GEMM_FLAGS('N', 'T');
  auto dtype      = (sizeof(DType) == 2) ? LIBXSMM_DATATYPE_BF16 : LIBXSMM_DATATYPE_F32;


    auto gemm_n = bc;
    auto gemm_m = bk;
    auto gemm_k = bn;
    auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bn, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, bn, C*ifhp*ifwp, bn, dtype, dtype, dtype);
    //printf("trans_xform_kernel:\n");
    //printf("tr_unary_shape m n ldi ldo = %d %d %d %d \n", tr_unary_shape.m, tr_unary_shape.n, tr_unary_shape.ldi, tr_unary_shape.ldo);
    trans_xform_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, K*ofhp*ofwp, bk, dtype, dtype, dtype);
    //printf("vnni_xform_kernel:\n");
    //printf("tr_unary_shape m n ldi ldo = %d %d %d %d \n", tr_unary_shape.m, tr_unary_shape.n, tr_unary_shape.ldi, tr_unary_shape.ldo);
    vnni_xform_kernel =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    /* Tr input is : Cb ifhp ifwp bc bn  */
    /* Tr output is: Kb ofhp ofwp bn/2 bk 2  */
    gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );

    //printf("brgemm_kernel_acc_pixel:\n");
    //printf("l_shape       m n k lda ldb ldc = %d %d %d %d %d %d\n", l_shape.m, l_shape.n, l_shape.k, l_shape.lda, l_shape.ldb, l_shape.ldc);
    //printf("l_flags = %d \n", l_flags);

    auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bn*bk*sizeof(DType), stride_w*bc*bn*sizeof(DType), 0 );
    brgemm_kernel_acc_pixel.gemm  = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );

    auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
    zero_kernel_bf16 = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

    l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

    l_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bk, LIBXSMM_DATATYPE_F32, dtype, LIBXSMM_DATATYPE_F32);
    fp32bf16_cvt_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    l_unary_shape = libxsmm_create_meltw_unary_shape(bk, bc, bk, bk, dtype, dtype, dtype);
    wt_vnni_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    //printf("wt_vnni_kernel:\n");
    //printf("l_unary_shape m n ldi ldo = %d %d %d %d \n", l_unary_shape.m, l_unary_shape.n, l_unary_shape.ldi, l_unary_shape.ldo);

    l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, nThreads, K * C *R * S, chunk0, LIBXSMM_DATATYPE_F32, dtype, LIBXSMM_DATATYPE_F32);
    wt_reduce_kernel0_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    l_unary_shape.m         = chunk1;
    l_unary_shape.ldo       = chunk1;
    wt_reduce_kernel1_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ; 
    //printf("l_unary_shape m n ldi ldo = %d %d %d %d \n", l_unary_shape.m, l_unary_shape.n, l_unary_shape.ldi, l_unary_shape.ldo);
  }
#endif


  if (sizeof(T) == 4) {
//#if 0
    gemm_n = bc;
    gemm_m = bk;
    gemm_k = ofw;

    //std::cout << "gemm_n gemm_m gemm_k for bwd_upd = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;

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

//#endif
  } else { /* bfloat16 goes here */

    //printf("Untested so far!\n");
    //exit(-1);

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

        printf("for gemm_kernel_non_hybrid as brgemm extension\n");
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
      } /* else-if for use_hybrid_imgfm_parallelization == 0 */

      printf("transpose_input_pixels_bf16_xform_tpp\n");
      //auto new_tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, ifwp, bc, input_pixels, dtype, dtype, dtype);
      //transpose_input_pixels_bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      transpose_input_pixels_bf16_xform_tpp = SCOPEIT(XformExtTPP<T>(ifwp, bc, bc, ifwp, bc, input_pixels, XformTPP::XFORM_XPOSE_TPP, false), XPOSE); /* assuming row-major-ness */
      //new_tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, compute_pixels, bk, bk, dtype, dtype, dtype);
      printf("vnni_output_compute_pixels_bf16_xform_tpp\n");
      if ((ofhp * ofwp) % 2 == 0) {
        //vnni_output_compute_pixels_bf16 =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
        vnni_output_compute_pixels_bf16_xform_tpp = SCOPEIT(XformExtTPP<T>(compute_pixels, bk, compute_pixels, bk, bk, bk, XformTPP::XFORM_N2V_TPP, false), XPOSE); /* assuming row-major-ness */
      } else {
        //printf("Xform TPP wrapper for LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD has not been implemented \n");
        //exit(-1);
        vnni_output_compute_pixels_bf16_xform_tpp = SCOPEIT(XformExtTPP<T>(compute_pixels, bk, compute_pixels, bk, bk, bk, XformTPP::XFORM_N2V_PAD_TPP, false), XPOSE); /* assuming row-major-ness */
        //vnni_output_compute_pixels_bf16 =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2_PAD, new_tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
      }
      upd_remaining_pixels = output_pixels - ((compute_pixels+1)/2)*2;
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
#if 1
  // Aux steps for linearized algo loops
  long _n_step = 1;
  long _k_step = 1;
  long _c_step = 1;
  long _r_step = 1;
  long _s_step = 1;
#endif

  //std::cout << "debug: fm_blocking reduce_work reduce_work_tripcount chunk0 chunk1 = " << fm_blocking << " " <<  reduce_work << " " << reduce_work_tripcount << " " << chunk0 << " " << chunk1 << std::endl;

  //std::cout << "debug: N = nThreads? n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " = " << nThreads << " " << n_step << " " << Cb << " " << c_step << " "
  //                                                                                              << Kb << " " << k_step << " " << ofh << " " << h_step << " "
  //                                                                                              << ofw << " " << w_step << " " << R << " " << r_step << " "
  //                                                                                              << S << " " << s_step << " " << std::endl;

  // FIXME: Is not necessary?
  //DECL_VLA_PTR_PT    (T,    tmp_scratch,   [Kb][Cb][R][S][bc][bk], t_scratch);
  //memset((T*)(tmp_scratch[0][0][0][0][0][0]), 0, nThreads*C*K*R*S*sizeof(T));

  auto zero_wt_loop = ThreadedLoop<5>({
      LoopSpecs{0, nThreads, 1, false},// true}, 
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      "Abcde");
//#if 0

  /* FIXME: Fix this! */
  char loop_specs_str[256] = "Abcdefg";

  auto conv_bwd_upd_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step},//, true},
      LoopSpecs{0, Cb, c_step},//, true},
      LoopSpecs{0, Kb, k_step},//, true},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},//, true},
      LoopSpecs{0, S, s_step}},//, true}},
#if 1
      fp32_conv_spec_string);
#else
      loop_specs_str);
#endif

//#endif

  auto tr_input_loop = ThreadedLoop<3>({
      LoopSpecs{0, Cb, tr_step},
      LoopSpecs{0, ifhp, tr_step},
      LoopSpecs{0, ifwp, tr_step}},
      "ABC"); //!!! FIXME back to ABC

  auto tr_output_loop  = ThreadedLoop<3>({
      LoopSpecs{0, Kb, tr_step},
      LoopSpecs{0, ofhp, tr_step},
      LoopSpecs{0, ofwp, tr_step}},
      "ABC"); //!!! FIXME back to ABC

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
//#if 0
        if (use_mb_par_f32 == 0) {
          printf("Case use_mb_par_f32 == 0 is untested so far!\n");
          exit(-1);

          conv_bwd_upd_loop(
            [&](int* ind) {
              int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

              //DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
              DECL_VLA_PTR_PT_EXT(T,    gradout,   [Kb][ofhp][ofwp][bk],   t_GO, (pad_h_out * ofwp * bk + pad_w_out * bk));
              //DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
              DECL_VLA_PTR_PT    (T,     inp,      [Cb][ifhp][ifwp][bc],   t_I);
              //DType *filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
              DECL_VLA_PTR_PT    (T,    weight,    [Cb][R][S][bc][bk],     t_grad_weight);

              //libxsmm_gemm_param gemm_param;
              //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              if (i_n == 0 && i_w == 0 && i_h == 0) {
                //libxsmm_meltw_unary_param zero_param;
                //zero_param.out.primary = (void*)gemm_param.c.primary;
                //zero_kernel( &zero_param );
                zero_bf16_tpp(weight[i_k][i_c][i_r][i_s][0]);
              }
              //gemm_kernel.gemm( &gemm_param );
              gemm_as_brgemm_tpp(inp    [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                                 gradout[i_n][i_k][i_h]                 [i_w],
                                 weight [i_k][i_c][i_r][i_s][0],
                                 1, /* brcount */
                                 true);
            },
            [&]() {if (sizeof(T) == 2) gemm_as_brgemm_tpp.config();},
            [&]() {if (sizeof(T) == 2) gemm_as_brgemm_tpp.release();});

        } else { /* else for if (use_mb_par == 0) */

          printf("Case else for use_mb_par_f32 == 0 is untested so far!\n");
          exit(-1);

          zero_wt_loop(
            [&](int* ind) {
              int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];

#if 0
              DECL_VLA_PTR_PT    (T,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch);
#else
              DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);
#endif
              //libxsmm_meltw_unary_param zero_param;
              //zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, i_n, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              //zero_kernel( &zero_param );
              zero_tpp(scratch[i_n][i_k][i_c][i_r][i_s][0]);
            },
            [&]() {},
            [&]() {});

          conv_bwd_upd_loop(
            [&](int* ind) {
              int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];
              int tid = omp_get_thread_num();

              //DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
              DECL_VLA_PTR_PT_EXT(T,    gradout,   [Kb][ofhp][ofwp][bk],   t_GO, (pad_h_out * ofwp * bk + pad_w_out * bk));
              //DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
              DECL_VLA_PTR_PT    (T,    inp,       [Cb][ifhp][ifwp][bc],   t_I);
              //DType *scratch_libxsmm = (DType*)libxsmm_aligned_malloc( nThreads*C*K*R*S*sizeof(DType), 2097152);
#if 0
              DECL_VLA_PTR_PT    (T,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch);
#else
              DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);
#endif

              //libxsmm_gemm_param gemm_param;
              //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              //gemm_kernel.gemm( &gemm_param );
              gemm_as_brgemm_tpp(inp    [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                                 gradout[i_n][i_k][i_h]                 [i_w],
                                 scratch[tid][i_k][i_c][i_r][i_s][0],
                                 1, /* brcount */
                                 true);
            },
            [&]() {if (sizeof(T) == 2) gemm_as_brgemm_tpp.config();},
            [&]() {if (sizeof(T) == 2) gemm_as_brgemm_tpp.release();});

          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];

              DECL_VLA_PTR_PT    (T,    weight2d,  [chunk0],               t_grad_weight);
#if 0
              DECL_VLA_PTR_PT    (T,    scratch2d, [chunk0],               t_scratch);
#else
              DECL_VLA_PTR_PT_EXT_CAST(T,    unsigned char, scratch2d, [chunk0],               t_scratch_experimental, 0);
#endif
              //libxsmm_meltw_unary_param reduce_param;
              //reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm, i_n, 0, chunk0);
              //reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)filter_libxsmm,  i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                //wt_reduce_kernel0_f32( &reduce_param );
                wt_reduce0_T_tpp( scratch2d[i_n], weight2d[i_n] );
              } else {
                //wt_reduce_kernel1_f32( &reduce_param );
                wt_reduce1_T_tpp( scratch2d[i_n], weight2d[i_n] );
              }
            },
            [&]() {},
            [&]() {});
        }
//#endif
      } else { /* T = bfloat16 goes into else */
#if 1
        if (bf16_use_nchw_format > 0) {
          //printf("Case bf16_use_nchw_format > 0 is untested so far!\n");
          //exit(-1);
          if (bf16_fuse_upd_transposes == 0) {
            //printf("Case bf16_fuse_upd_transposes == 0 is untested so far!\n");
            //exit(-1);
            tr_input_nchw_loop(
              [&](int* ind) {
                int i_n = ind[0], i_c = ind[1];
                //DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
                DECL_VLA_PTR_PT         (T,                input,                      [Cb][ifhp][ifwp][bc],   t_I);
                DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char, input_mylinearized_pixels,  [Cb][bc][input_pixels], t_scratch_experimental, input_mylinearized_pixels_offset);
                //libxsmm_meltw_unary_param unary_param;
                for (int ij = 0; ij < ifhp; ij++) {
                  //unary_param.in.primary = (void*) LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm,           i_n, i_c, ij, 0, 0, Cb, ifhp, ifwp, bc);
                  //unary_param.out.primary= (void*) LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, ij*ifwp, Cb, bc, input_pixels);
                  //transpose_input_pixels_bf16( &unary_param );
                  transpose_input_pixels_bf16_xform_tpp(input[i_n][i_c][ij][0], &input_mylinearized_pixels[i_n][i_c][0][ij*ifwp]);
                }
              },
              [&]() {},
              [&]() {});

            tr_output_nchw_loop(
              [&](int* ind) {
                int i_n = ind[0], i_k = ind[1];
                //DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
                DECL_VLA_PTR_PT         (T,                gradout,                    [Kb][ofhp][ofwp][bk],   t_GO);
                DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char, output_mylinearized_pixels, [Kb][output_pixels][bk], t_scratch_experimental, output_mylinearized_pixels_offset);

                //libxsmm_meltw_unary_param unary_param;
                //unary_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm,           i_n, i_k, pad_h, pad_w, 0, Kb, ofhp, ofwp, bk);
                //unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, 0, 0, Kb, output_pixels, bk);
                //vnni_output_compute_pixels_bf16( &unary_param );
                //if (upd_remaining_pixels > 0) {
                //  unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, (compute_pixels+1)/2, 0, Kb, output_pixels, bk);
                //  vnni_output_zero_remaining_pixels_bf16( &unary_param );
                //}
                vnni_output_compute_pixels_bf16_xform_tpp(gradout[i_n][i_k][pad_h][pad_w], output_mylinearized_pixels[i_n][i_k][0]);
                if (upd_remaining_pixels > 0) {
                  //unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, (compute_pixels+1)/2, 0, Kb, output_pixels, bk);
                  //vnni_output_zero_remaining_pixels_bf16( &unary_param );
                  printf("Case upd_remaining_pixels > 0 is untested so far!\n");
                  exit(-1);
                  vnni_output_zero_remaining_pixels_bf16_tpp(output_mylinearized_pixels[i_n][i_k][(compute_pixels+1)/2]);
                }
              },
              [&]() {},
              [&]() {});
          } /* for if bf16_fuse_upd_transposes == 0 */

          if (use_hybrid_imgfm_parallelization == 0) {
            //printf("Case use_hybrid_imgfm_parallelization == 0 is untested so far!\n");
            //exit(-1);
//            #if 0

/*
            typedef T DType;
            T* input_libxsmm = t_I.data_ptr<T>();
            //T* input_linearized_pixels = t_input_linearized_pixels.data_ptr<T>();
            T* output_libxsmm = t_GO.data_ptr<T>();
            //T* output_linearized_pixels = t_output_linearized_pixels.data_ptr<T>();
            float *scratch_libxsmm = reinterpret_cast<float*>(t_scratch_experimental.data_ptr<unsigned char>() + scratch_float_offset);
            T *filter_libxsmm = t_grad_weight.data_ptr<T>();
            T *scratch_libxsmm_bf16_weights = reinterpret_cast<T*>(t_scratch_experimental.data_ptr<unsigned char>() + scratch_bf16_weight_offset);
*/
            //printf("dbg: pointers here: input_libxsmm = %p output_libxsmm %p scratch_libxsmm %p filter_libxsmm %p scratch_bf16... %p \n", input_libxsmm, output_libxsmm, scratch_libxsmm, filter_libxsmm, scratch_libxsmm_bf16_weights);

        conv_loop_bf16_nchw(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], pix = ind[3], i_r = ind[4], i_s = ind[5];
            libxsmm_gemm_param gemm_param;
            libxsmm_meltw_unary_param unary_param;
            int tid = omp_get_thread_num();

              //DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
              DECL_VLA_PTR_PT         (T,                    gradout,   [Kb][ofhp][ofwp][bk],   t_GO);
              //DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
              DECL_VLA_PTR_PT         (T,                    input,     [Cb][ifhp][ifwp][bc],   t_I);

              //DType *scratch_libxsmm = (DType*)libxsmm_aligned_malloc( nThreads*C*K*R*S*sizeof(DType), 2097152);
              DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, scratch,       [Kb][Cb][R][S][bc][bk], t_scratch_experimental, 0);
              DECL_VLA_PTR_PT_EXT_CAST(float, unsigned char, scratch_float, [Kb][Cb][R][S][bc][bk], t_scratch_experimental, scratch_float_offset);

              DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, output_mylinearized_pixels, [Kb][output_pixels][bk], t_scratch_experimental, output_mylinearized_pixels_offset);
              DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, input_mylinearized_pixels,  [Cb][bc][input_pixels],  t_scratch_experimental, input_mylinearized_pixels_offset);

            if (bf16_fuse_upd_transposes == 1 && pix == 0 && i_c == 0 && i_r == 0 && i_s == 0) {
              //printf("Case bf16_fuse_upd_transposes == 1 + bunch of conditions is untested so far!\n");
              //exit(-1);
              //unary_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm,           i_n, i_k, pad_h, pad_w, 0, Kb, ofhp, ofwp, bk);
              //unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, 0, 0, Kb, output_pixels, bk);
              //vnni_output_compute_pixels_bf16( &unary_param );
              vnni_output_compute_pixels_bf16_xform_tpp(gradout[i_n][i_k][pad_h][pad_w], output_mylinearized_pixels[i_n][i_k][0]);
              if (upd_remaining_pixels > 0) {
                //unary_param.out.primary= LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, (compute_pixels+1)/2, 0, Kb, output_pixels, bk);
                //vnni_output_zero_remaining_pixels_bf16( &unary_param );
                printf("Case upd_remaining_pixels > 0 is untested so far!\n");
                exit(-1);
                vnni_output_zero_remaining_pixels_bf16_tpp(output_mylinearized_pixels[i_n][i_k][(compute_pixels+1)/2]);
              }
            }

            if (bf16_fuse_upd_transposes == 1 && pix == 0 && i_k == 0 && i_r == 0 && i_s == 0) {
              //printf("Case bf16_fuse_upd_transposes == 1 + bunch of conditions is untested so far!\n");
              //exit(-1);

              for (int ij = 0; ij < ifhp; ij++) {
                //unary_param.in.primary = (void*) LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm,           i_n, i_c, ij, 0, 0, Cb, ifhp, ifwp, bc);
                //unary_param.out.primary= (void*) LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, ij*ifwp, Cb, bc, input_pixels);
                //transpose_input_pixels_bf16( &unary_param );
                transpose_input_pixels_bf16_xform_tpp(input[i_n][i_c][ij][0], &input_mylinearized_pixels[i_n][i_c][0][ij*ifwp]);
              }

            }
       
            if (use_f32_wt_reduction_and_external_wt_vnni > 0) {
              //printf("Case use_f32_wt_reduction_and_external_wt_vnni > 0 is untested so far!\n");
              //exit(-1);
              //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);     
              //gemm_param.a.primary = (void*)output_mylinearized_pixels[i_n][i_k][pix];
              //gemm_param.b.primary = (void*)&input_mylinearized_pixels[i_n][i_c][0][pix + i_r * ifwp + i_s];
              //gemm_param.c.primary = (void*)scratch_float[tid][i_k][i_c][i_r][i_s][0];
              if (pix == 0) {
                //libxsmm_meltw_unary_param zero_param;
                //zero_param.out.primary = (void*)gemm_param.c.primary;
                //zero_kernel( &zero_param );
                zero_float_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0]);
              }
              //gemm_kernel_non_hybrid.gemm( &gemm_param );
//#if 0
              gemm_kernel_non_hybrid_as_brgemm_tpp(&input_mylinearized_pixels[i_n][i_c][0][pix + i_r * ifwp + i_s],
                                                    output_mylinearized_pixels[i_n][i_k][pix],
                                                    scratch_float[tid][i_k][i_c][i_r][i_s][0],
                                                    1 /* brcount */,
                                                    true);
//#endif
            } else {
              //printf("Case else for use_f32_wt_reduction_and_external_wt_vnni > 0 is untested so far!\n");
              //exit(-1);
              /* Use beta = 0 kernel with c_vnni formating */
              //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);     
              //printf("Calling gemm_kernel_non_hybrid_zerobeta_cvnni \n");
              //exit(-1);
              //gemm_kernel_non_hybrid_zerobeta_cvnni.gemm( &gemm_param );
//#if 0
              gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp(&input_mylinearized_pixels[i_n][i_c][0][pix + i_r * ifwp + i_s],
                                                                  output_mylinearized_pixels[i_n][i_k][pix],
                                                                  scratch[tid][i_k][i_c][i_r][i_s][0],
                                                                  1 /* brcount */,
                                                                  true);
//#endif
            }
          },
          [&]() {if (sizeof(T) == 2) gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp.config();},
          [&]() {if (sizeof(T) == 2) gemm_kernel_non_hybrid_zerobeta_cvnni_as_brgemm_tpp.release();});

        if (use_f32_wt_reduction_and_external_wt_vnni > 0) {
            //printf("Case use_f32_wt_reduction_and_external_wt_vnni > 0 is untested so far!\n");
            //exit(-1);

          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float_2d,        [chunk0], t_scratch_experimental,              scratch_float_offset);
              DECL_VLA_PTR_PT_EXT_CAST    (T,     unsigned char, scratch_bf16_weight_2d,  [chunk0], t_scratch_experimental,              scratch_bf16_weight_offset);

              //libxsmm_meltw_unary_param reduce_param;
              //reduce_param.in.primary = (void*)scratch_float_2d[i_n];
              //reduce_param.out.primary = (void*)scratch_bf16_weight_2d[i_n];
              //reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(float), (float*)scratch_libxsmm, i_n, 0, chunk0);
              //reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm_bf16_weights, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                //wt_reduce_kernel0_f32bf16( &reduce_param );
                wt_reduce0_f32bf16_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
              } else {
                //wt_reduce_kernel1_f32bf16( &reduce_param );
                wt_reduce1_f32bf16_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
              }
            },
            [&]() {},
            [&]() {});
//!!!
          vnni_wt_loop(
            [&](int* ind) {
              int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];
              DECL_VLA_PTR_PT_EXT_CAST(T,     unsigned char, scratch_bf16_weight, [Cb][R][S][bc][bk], t_scratch_experimental, scratch_bf16_weight_offset);
              DECL_VLA_PTR_PT         (T,                    filter,              [Cb][R][S][bc][bk], t_grad_weight);
              //libxsmm_meltw_unary_param xform_param;
              //xform_param.in.primary = (void*)scratch_bf16_weight[i_k][i_c][i_r][i_s][0];
              //xform_param.out.primary = (void*)filter[i_k][i_c][i_r][i_s][0];
              //xform_param.in.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), scratch_libxsmm_bf16_weights, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              //xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              //wt_vnni_kernel( &xform_param );
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
              //libxsmm_meltw_unary_param reduce_param;
              //reduce_param.in.primary   = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm, i_n, 0, chunk0);
              //reduce_param.out.primary  = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)filter_libxsmm, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                //wt_reduce_kernel0_bf16bf16( &reduce_param );
                wt_reduce0_bf16bf16_tpp( scratch2d[i_n], weight2d[i_n] );
              } else {
                //wt_reduce_kernel1_bf16bf16( &reduce_param );
                wt_reduce1_bf16bf16_tpp( scratch2d[i_n], weight2d[i_n] );
              }
            },
            [&]() {},
            [&]() {});
        }

//            #endif
          } else { /* for if use_hybrid_imgfm_parallelization == 0 */
            printf("Case else for use_hybrid_imgfm_parallelization == 0 is untested so far!\n");
            exit(-1);
            #if 0
       conv_loop_bf16_nchw(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], pix = ind[3], i_r = ind[4], i_s = ind[5];
            int my_col_id;
            unsigned long long brcount = _n_step;
            libxsmm_gemm_param gemm_param;
            libxsmm_meltw_unary_param unary_param;
            
            if (compute_full_wt_output_block == 0) {
              my_col_id = ind[9];
              if (use_f32_wt_reduction_and_external_wt_vnni > 0) { 
                gemm_param.op.tertiary = (void*)&brcount;        
                gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
                gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, my_col_id, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);     
                if (pix == 0) {
                  libxsmm_meltw_unary_param zero_param;
                  zero_param.out.primary = (void*)gemm_param.c.primary;
                  zero_kernel( &zero_param );
                }
                brgemm_kernel_hybrid.gemm( &gemm_param );
              } else {
                gemm_param.op.tertiary = (void*)&brcount;        
                gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
                gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
                gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, my_col_id, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);     
                brgemm_kernel_hybrid_zerobeta_cvnni.gemm( &gemm_param );       
              }
            } else {
              gemm_param.op.tertiary = (void*)&brcount;        
              gemm_param.a.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), output_linearized_pixels, i_n, i_k, pix, 0, Kb, output_pixels, bk);
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(4, sizeof(DType), input_linearized_pixels, i_n, i_c, 0, pix + i_r * ifwp + i_s, Cb, bc, input_pixels);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), (DType*)filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);     
              brgemm_kernel_hybrid_zerobeta_cvnni.gemm( &gemm_param );  
            }
          },
          [&]() {if (sizeof(DType) == 2) tileconfig_kernel.gemm(NULL);},
          [&]() {if (sizeof(DType) == 2) tilerelease_kernel.gemm(NULL);});

        if (use_f32_wt_reduction_and_external_wt_vnni > 0) { 
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              libxsmm_meltw_unary_param reduce_param;
              reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(float), (float*)scratch_libxsmm, i_n, 0, chunk0);
              reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm_bf16_weights, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce_kernel0_f32bf16( &reduce_param );
              } else {
                wt_reduce_kernel1_f32bf16( &reduce_param );  
              } 
            },
            [&]() {},
            [&]() {});

          vnni_wt_loop(
            [&](int* ind) {
              int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];
              libxsmm_meltw_unary_param xform_param;
              xform_param.in.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), scratch_libxsmm_bf16_weights, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              wt_vnni_kernel( &xform_param );
            },
            [&]() {},
            [&]() {});
        } else if (compute_full_wt_output_block == 0) {
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];
              libxsmm_meltw_unary_param reduce_param;
              reduce_param.in.primary   = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm, i_n, 0, chunk0);
              reduce_param.out.primary  = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)filter_libxsmm, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                wt_reduce_kernel0_bf16bf16( &reduce_param );
              } else {
                wt_reduce_kernel1_bf16bf16( &reduce_param );  
              } 
            },
            [&]() {},
            [&]() {});
        }
//      }
            #endif
          } /* if-else for use_hybrid_imgfm_parallelization */
        } else if (bf16_use_chwn_format > 0) { /* for if bf16_use_nchw_format > 0 */
          //printf("Case bf16_use_chwn_format > 0 is untested so far!\n");
          //exit(-1);
          if (use_private_trans == 0) {
            //printf("Case use_private_trans == 0 is untested so far!\n");
            //exit(-1);
            tr_input_loop(
              [&](int* ind) {
                int i_c = ind[0], i_h = ind[1], i_w = ind[2];

                DECL_VLA_PTR_PT    (T,    input,          [Cb]  [ifhp][ifwp][bc],   t_I);
                DECL_VLA_PTR_PT_EXT_CAST(T,  unsigned char,  tr_input,       [ifhp][ifwp][bc]  [bn],   t_scratch_experimental, tr_input_offset);
                //libxsmm_meltw_unary_param trans_param;
                //trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType),  input_libxsmm,    0,   i_c, i_h, i_w, 0, Cb,   ifhp, ifwp, bc);
                //trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h, i_w, 0,   0, ifhp, ifwp, bc,   bn);
                //trans_xform_kernel( &trans_param );
                trans_xform_tpp(input[0][i_c][i_h][i_w], tr_input[i_c][i_h][i_w][0]);
              },
              [&]() {},
              [&]() {});

  //          printf("calling vnni_xform_tpp");

            tr_output_loop(
              [&](int* ind) {
                int i_k = ind[0], i_h = ind[1], i_w = ind[2];

                DECL_VLA_PTR_PT    (T,    output,         [Kb]  [ofhp][ofwp][bk],   t_GO);
                DECL_VLA_PTR_PT_EXT_CAST (T, unsigned char,    tr_output,      [ofhp][ofwp][bn]  [bk],   t_scratch_experimental, tr_output_offset);
                //libxsmm_meltw_unary_param trans_param;
                //trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType),  output_libxsmm,    0,   i_k, i_h, i_w, 0, Kb,   ofhp, ofwp, bk);
                //trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h, i_w, 0,   0, ofhp, ofwp, bn,   bk);
                //vnni_xform_kernel( &trans_param );
                vnni_xform_tpp(output[0][i_k][i_h][i_w], tr_output[i_k][i_h][i_w][0]);
              },
              [&]() {},
              [&]() {});
          } else { /* dummy else for if use_private_trans == 0 */
            //printf("Case else use_private_trans == 0 is untested so far!\n");
            //exit(-1);
          }
          if (par_over_h_pixels > 0) {
            zero_wt_loop(
              [&](int* ind) {
                int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];
                DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float, [Kb][Cb][R][S][bc][bk], t_scratch_experimental, scratch_float_offset);
                //libxsmm_meltw_unary_param zero_param;
                //zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, i_n, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //zero_kernel( &zero_param );
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
                //libxsmm_meltw_unary_param zero_param;
                //zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //zero_kernel( &zero_param );
                zero_float_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0]);
              }

              if (use_private_trans > 0) {

                DECL_VLA_PTR_PT    (T,     input,             [Cb][ifhp][ifwp][bc],   t_I);
                DECL_VLA_PTR_PT    (T,     output,            [Kb][ofhp][ofwp][bk],   t_GO);

                DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,     private_tr_input,  [ifhp][ifwp][bc][bn],   t_scratch_experimental, private_tr_input_offset  + tid*(N*ifhp*ifwp*C) * sizeof(T));
                DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,     private_tr_output, [ofhp][ofwp][bn][bk],   t_scratch_experimental, private_tr_output_offset + tid*(N*ofhp*ofwp*K) * sizeof(T));

                //libxsmm_gemm_param gemm_param;
                int *inp_loc = (int*) trans_tracker.get() + tid * trans_tracker_size + i_c;
                int *out_loc = (int*) trans_tracker.get() + tid * trans_tracker_size + Cb + i_k;

                int is_inp_trans = *inp_loc;
                int is_out_trans = *out_loc;

                if (is_inp_trans == 0) {
                  for (int _ih = 0; _ih < ifhp; _ih++) {
                    for (int _iw = 0; _iw < ifwp; _iw++) {
                      //libxsmm_meltw_unary_param trans_param;
                      //trans_param.in.primary  = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm,                           0, i_c, _ih, _iw, 0, Cb,   ifhp, ifwp, bc);
                      //trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_input_libxsmm[tid], i_c, _ih, _iw,   0, 0, ifhp, ifwp, bc,   bn);
                      //trans_xform_kernel( &trans_param );
                      trans_xform_tpp(input[0][i_c][_ih][_iw], private_tr_input[i_c][_ih][_iw][0]);
                    }
                  }
                  *inp_loc = 1;
                }

                if (is_out_trans == 0) {
                  for (int _ih = 0; _ih < ofhp; _ih++) {
                    for (int _iw = 0; _iw < ofwp; _iw++) {
                      //libxsmm_meltw_unary_param trans_param;
                      //trans_param.in.primary  = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm,                           0, i_k, _ih, _iw, 0, Kb,  ofhp,  ofwp, bk);
                      //trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_output_libxsmm[tid], i_k, _ih, _iw,   0, 0, ofhp, ofwp, bn,   bk);
                      //vnni_xform_kernel( &trans_param );
                      vnni_xform_tpp(output[0][i_k][_ih][_iw], private_tr_output[i_k][_ih][_iw][0]);
                    }
                  }
                  *out_loc = 1;
                }

                //unsigned long long brcount = w_step*h_step;
                //gemm_param.op.tertiary = (void*)&brcount;
                //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_output_libxsmm[tid], i_k, i_h + pad_h_out, i_w + pad_w_out, 0, 0, ofhp, ofwp, bn, bk);
                //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_input_libxsmm[tid] , i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, 0, ifhp, ifwp, bc, bn);
                //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //brgemm_kernel_acc_pixel.gemm( &gemm_param );
                brgemm_acc_pixel_tpp(private_tr_input  [i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][0],
                                     private_tr_output [i_k][i_h + pad_h_out]     [i_w + pad_w_out]     [0],
                                     scratch_float     [tid][i_k][i_c][i_r][i_s][0],
                                     w_step * h_step, /* brcount */
                                     true);


              } else { /* for if use_private_trans > 0 */
                //libxsmm_gemm_param gemm_param;

                DECL_VLA_PTR_PT_EXT_CAST    (T, unsigned char,     tr_input,      [ifhp][ifwp][bc]  [bn],   t_scratch_experimental, tr_input_offset);
                DECL_VLA_PTR_PT_EXT_CAST    (T, unsigned char,     tr_output,     [ofhp][ofwp][bn]  [bk],   t_scratch_experimental, tr_output_offset);

                //unsigned long long brcount = w_step*h_step;
                //gemm_param.op.tertiary = (void*)&brcount;
                //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h + pad_h_out, i_w + pad_w_out, 0, 0, ofhp, ofwp, bn, bk);
                //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, 0, ifhp, ifwp, bc, bn);
                //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //brgemm_kernel_acc_pixel.gemm( &gemm_param );
                brgemm_acc_pixel_tpp(tr_input     [i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][0],
                                     tr_output    [i_k][i_h + pad_h_out]     [i_w + pad_w_out]     [0],
                                     scratch_float[tid][i_k][i_c][i_r][i_s][0],
                                     w_step * h_step, /* brcount */
                                     true);
              } /* for if-else use_private_trans > 0 */

              if ((i_h == ofh - h_step) && (i_w == ofw - w_step) && (par_over_h_pixels == 0)) {
                //libxsmm_meltw_unary_param xform_param;
                //xform_param.in.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //xform_param.out.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //fp32bf16_cvt_kernel( &xform_param );
                fp32bf16_cvt_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0], (T*)(scratch_float[tid][i_k][i_c][i_r][i_s][0]));
                //xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
                //wt_vnni_kernel( &xform_param );
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
                //libxsmm_meltw_unary_param reduce_param;
                //reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(float), (float*)scratch_libxsmm, i_n, 0, chunk0);
                //reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm_bf16_weights, i_n, 0, chunk0);
                if (i_n < reduce_work_tripcount - 1) {
                  //wt_reduce_kernel0_f32bf16( &reduce_param );
                  wt_reduce0_float_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
                } else {
                  //wt_reduce_kernel1_f32bf16( &reduce_param );
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

                //libxsmm_meltw_unary_param xform_param;
                //xform_param.in.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), scratch_libxsmm_bf16_weights, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
                //xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
                //wt_vnni_kernel( &xform_param );
                wt_vnni_xform_tpp(scratch_bf16_weight[i_k][i_c][i_r][i_s][0], filter[i_k][i_c][i_r][i_s][0] );
              },
              [&]() {},
              [&]() {});

          } /* for if par_over_h_pixels > 0 */
        } /* if-else for bf16_use_nchw_format/bf16_use_chwn_format */
#else /* for if 1 around new/old upd code*/
        if (bf16_upfront_trans > 0  && use_private_trans == 0) {
#ifdef DEBUGGING
        tr_input_loop(
          [&](int* ind) {
            int i_c = ind[0], i_h = ind[1], i_w = ind[2];
            libxsmm_meltw_unary_param trans_param;
            //DECL_VLA_PTR_PT    (T,    input,          [Cb]  [ifhp][ifwp][bc],   t_I);
            //DECL_VLA_PTR_PT    (T,    tr_input,       [ifhp][ifwp][bc]  [bn],   t_tr_input);
            //trans_param.in.primary  = (void*)input[0][i_c][i_h][i_w];
            //trans_param.out.primary = (void*)tr_input[i_c][i_h][i_w][0];
            typedef T DType;
            T* input_libxsmm = t_I.data_ptr<T>();
            T* tr_input_libxsmm = t_tr_input.data_ptr<T>();
            trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, 0, i_c, i_h, i_w, 0, Cb, ifhp, ifwp, bc);
            trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h, i_w, 0, 0, ifhp, ifwp, bc, bn);
            trans_xform_kernel( &trans_param );
          },
          [&]() {},
          [&]() {});

        tr_output_loop(
          [&](int* ind) {
            int i_k = ind[0], i_h = ind[1], i_w = ind[2];
            libxsmm_meltw_unary_param trans_param;
            //DECL_VLA_PTR_PT    (T,    output,         [Kb]  [ofhp][ofwp][bk],   t_GO);
            //DECL_VLA_PTR_PT    (T,    tr_output,      [ofhp][ofwp][bn]  [bk],   t_tr_output);
            //trans_param.in.primary = (void*)output[0][i_k][i_h][i_w];
            //trans_param.out.primary = (void*)tr_output[i_k][i_h][i_w][0];
            typedef T DType;
            T* output_libxsmm = t_GO.data_ptr<T>();
            T* tr_output_libxsmm = t_tr_output.data_ptr<T>();
            trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, 0, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
            trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h, i_w, 0, 0, ofhp, ofwp, bn, bk);
            vnni_xform_kernel( &trans_param );
          },
          [&]() {},
          [&]() {});
#else
//#if 0
//          printf("calling trans_xform_tpp");
          tr_input_loop(
            [&](int* ind) {
              int i_c = ind[0], i_h = ind[1], i_w = ind[2];

              DECL_VLA_PTR_PT    (T,    input,          [Cb]  [ifhp][ifwp][bc],   t_I);
#if 0
              DECL_VLA_PTR_PT    (T,    tr_input,       [ifhp][ifwp][bc]  [bn],   t_tr_input);
#else
              DECL_VLA_PTR_PT_EXT_CAST(T,  unsigned char,  tr_input,       [ifhp][ifwp][bc]  [bn],   t_scratch_experimental, tr_input_offset);
#endif
              //libxsmm_meltw_unary_param trans_param;
              //trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType),  input_libxsmm,    0,   i_c, i_h, i_w, 0, Cb,   ifhp, ifwp, bc);
              //trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h, i_w, 0,   0, ifhp, ifwp, bc,   bn);
              //trans_xform_kernel( &trans_param );
//              printf("i_c = %d i_h = %d i_w = %d base input = %p input[...] = %p base tr_input = %p tr_input[...] = %p \n", i_c, i_h, i_w,
//                                                                              input[0][0][0][0], input[0][i_c][i_h][i_w], tr_input[0][0][0][0], tr_input[i_c][i_h][i_w][0]);
              trans_xform_tpp(input[0][i_c][i_h][i_w], tr_input[i_c][i_h][i_w][0]);
            },
            [&]() {},
            [&]() {});
//          printf("calling trans_xform_tpp...finished");
//#endif
//#if 0
//          printf("calling vnni_xform_tpp");

          tr_output_loop(
            [&](int* ind) {
              int i_k = ind[0], i_h = ind[1], i_w = ind[2];

              DECL_VLA_PTR_PT    (T,    output,         [Kb]  [ofhp][ofwp][bk],   t_GO);
#if 0
              DECL_VLA_PTR_PT    (T,    tr_output,      [ofhp][ofwp][bn]  [bk],   t_tr_output);
#else
              DECL_VLA_PTR_PT_EXT_CAST (T, unsigned char,    tr_output,      [ofhp][ofwp][bn]  [bk],   t_scratch_experimental, tr_output_offset);
#endif
              //libxsmm_meltw_unary_param trans_param;
              //trans_param.in.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType),  output_libxsmm,    0,   i_k, i_h, i_w, 0, Kb,   ofhp, ofwp, bk);
              //trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h, i_w, 0,   0, ofhp, ofwp, bn,   bk);
              //vnni_xform_kernel( &trans_param );
//              printf("i_k = %d i_h = %d i_w = %d base output = %p output[...] = %p base tr_output = %p tr_output[...] = %p \n", i_k, i_h, i_w,
//                                                                              output[0][0][0][0], output[0][i_k][i_h][i_w], tr_output[0][0][0][0], tr_output[i_k][i_h][i_w][0]);
              //for (
              vnni_xform_tpp(output[0][i_k][i_h][i_w], tr_output[i_k][i_h][i_w][0]);
//              printf("i_k = %d i_h = %d i_w = %d finished \n", i_k, i_h, i_w);
            },
            [&]() {},
            [&]() {});
//#endif
#endif
        }

        if (par_over_h_pixels > 0) {
          zero_wt_loop(
            [&](int* ind) {
              int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];

#if 0
              DECL_VLA_PTR_PT    (float, scratch_float, [Kb][Cb][R][S][bc][bk], t_scratch_float);
#else
              DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float, [Kb][Cb][R][S][bc][bk], t_scratch_experimental, scratch_float_offset);
#endif
              //libxsmm_meltw_unary_param zero_param;
              //zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, i_n, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              //zero_kernel( &zero_param );
              zero_float_tpp(scratch_float[i_n][i_k][i_c][i_r][i_s][0]);
            },
            [&]() {},
            [&]() {});
        }

        if (use_private_trans > 0) {
          memset(trans_tracker.get(), 0, trans_tracker_size*nThreads*sizeof(int));
        }

//#if 0
#ifdef DEBUGGING
      conv_loop_bf16(
        [&](int* ind) {
          if (use_private_trans > 0) {
          } else {
            int i_c = ind[0], i_k = ind[1], i_h = ind[2], i_w = ind[3], i_r = ind[4], i_s = ind[5];
            int tid = omp_get_thread_num();
            libxsmm_gemm_param gemm_param;
            
            typedef T DType;
              //DECL_VLA_PTR_PT    (T,     tr_input,      [ifhp][ifwp][bc]  [bn],   t_tr_input);
              //DECL_VLA_PTR_PT    (T,     tr_output,     [ofhp][ofwp][bn]  [bk],   t_tr_output);
              //DECL_VLA_PTR_PT    (float, scratch_float, [Kb][Cb][R][S][bc][bk],   t_scratch_float);
              //DECL_VLA_PTR_PT    (T,     filter,        [Cb][R][S][bc][bk],       t_grad_weight);
            float *scratch_libxsmm = t_scratch_float.data_ptr<float>();

            if (i_h == 0 && i_w == 0 && par_over_h_pixels == 0) {
              libxsmm_meltw_unary_param zero_param;
              //zero_param.out.primary = (void*)scratch_float[tid][i_k][i_c][i_r][i_s][0];
              zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              zero_kernel( &zero_param );
            }

            unsigned long long brcount = w_step*h_step;
            gemm_param.op.tertiary = (void*)&brcount;
            T *tr_input_libxsmm = t_tr_input.data_ptr<T>();
            T *tr_output_libxsmm = t_tr_output.data_ptr<T>();
            //float *scratch_libxsmm = t_scratch_float.data_ptr<float>();
            //gemm_param.a.primary = (void*)tr_output    [i_k][i_h + pad_h_out]     [i_w + pad_w_out]     [0];
            //gemm_param.b.primary = (void*)tr_input     [i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][0];
            //gemm_param.c.primary = (void*)scratch_float[tid][i_k][i_c][i_r][i_s][0];
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h + pad_h_out, i_w + pad_w_out, 0, 0, ofhp, ofwp, bn, bk);
            gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, 0, ifhp, ifwp, bc, bn);
            gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
            brgemm_kernel_acc_pixel.gemm( &gemm_param );

            T *filter_libxsmm = t_grad_weight.data_ptr<T>();
            if ((i_h == ofh - h_step) && (i_w == ofw - w_step) && (par_over_h_pixels == 0)) {
              libxsmm_meltw_unary_param xform_param;
              xform_param.in.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              xform_param.out.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              //xform_param.in.primary = (void*)scratch_float[tid][i_k][i_c][i_r][i_s][0];
              //xform_param.out.primary = (void*)scratch_float[tid][i_k][i_c][i_r][i_s][0];
              fp32bf16_cvt_kernel( &xform_param );
              xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              //xform_param.out.primary = (void*)filter[i_k][i_c][i_r][i_s][0];
              wt_vnni_kernel( &xform_param );
            }
          }
        },
        [&]() {if (sizeof(T) == 2) tileconfig_kernel.gemm(NULL);},
        [&]() {if (sizeof(T) == 2) tilerelease_kernel.gemm(NULL);});

#else /* for #ifdef DEBUGGING */
        conv_loop_bf16(
          [&](int* ind) {
            if (use_private_trans > 0) {
              int i_c = ind[0], i_k = ind[1], i_h = ind[2], i_w = ind[3], i_r = ind[4], i_s = ind[5];
              int tid = omp_get_thread_num();

              DECL_VLA_PTR_PT    (T,     input,             [Cb][ifhp][ifwp][bc],   t_I);
              DECL_VLA_PTR_PT    (T,     output,            [Kb][ofhp][ofwp][bk],   t_GO);
#if 0
              DECL_VLA_PTR_PT    (float, scratch_float,     [Kb][Cb][R][S][bc][bk], t_scratch_float);
#else
              DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float,     [Kb][Cb][R][S][bc][bk], t_scratch_experimental, scratch_float_offset);
#endif
              DECL_VLA_PTR_PT    (T,     filter,            [Cb][R][S][bc][bk],     t_grad_weight);

#if 0
              DECL_VLA_PTR_PT_EXT(T,     private_tr_input,  [ifhp][ifwp][bc][bn],   t_private_tr_input,  tid*(N*ifhp*ifwp*C));
              DECL_VLA_PTR_PT_EXT(T,     private_tr_output, [ofhp][ofwp][bn][bk],   t_private_tr_output, tid*(N*ofhp*ofwp*K));
#else
              DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,     private_tr_input,  [ifhp][ifwp][bc][bn],   t_scratch_experimental, private_tr_input_offset  + tid*(N*ifhp*ifwp*C) * sizeof(T));
              DECL_VLA_PTR_PT_EXT_CAST(T, unsigned char,     private_tr_output, [ofhp][ofwp][bn][bk],   t_scratch_experimental, private_tr_output_offset + tid*(N*ofhp*ofwp*K) * sizeof(T));
#endif

              //libxsmm_gemm_param gemm_param;
              int *inp_loc = (int*) trans_tracker.get() + tid * trans_tracker_size + i_c;
              int *out_loc = (int*) trans_tracker.get() + tid * trans_tracker_size + Cb + i_k;

              int is_inp_trans = *inp_loc;
              int is_out_trans = *out_loc;

              if (is_inp_trans == 0) {
                for (int _ih = 0; _ih < ifhp; _ih++) {
                  for (int _iw = 0; _iw < ifwp; _iw++) {
                    //libxsmm_meltw_unary_param trans_param;
                    //trans_param.in.primary  = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm,                           0, i_c, _ih, _iw, 0, Cb,   ifhp, ifwp, bc);
                    //trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_input_libxsmm[tid], i_c, _ih, _iw,   0, 0, ifhp, ifwp, bc,   bn);
                    //trans_xform_kernel( &trans_param );
                    trans_xform_tpp(input[0][i_c][_ih][_iw], private_tr_input[i_c][_ih][_iw][0]);
                  }
                }
                *inp_loc = 1;
              }

              if (is_out_trans == 0) {
                for (int _ih = 0; _ih < ofhp; _ih++) {
                  for (int _iw = 0; _iw < ofwp; _iw++) {
                    //libxsmm_meltw_unary_param trans_param;
                    //trans_param.in.primary  = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm,                           0, i_k, _ih, _iw, 0, Kb,  ofhp,  ofwp, bk);
                    //trans_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_output_libxsmm[tid], i_k, _ih, _iw,   0, 0, ofhp, ofwp, bn,   bk);
                    //vnni_xform_kernel( &trans_param );
                    vnni_xform_tpp(output[0][i_k][_ih][_iw], private_tr_output[i_k][_ih][_iw][0]);
                  }
                }
                *out_loc = 1;
              }

              if (i_h == 0 && i_w == 0 && par_over_h_pixels == 0) {
                //libxsmm_meltw_unary_param zero_param;
                //zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //zero_kernel( &zero_param );
                zero_float_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0]);
              }

              //unsigned long long brcount = w_step*h_step;
              //gemm_param.op.tertiary = (void*)&brcount;
              //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_output_libxsmm[tid], i_k, i_h + pad_h_out, i_w + pad_w_out, 0, 0, ofhp, ofwp, bn, bk);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), (DType*)private_tr_input_libxsmm[tid] , i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, 0, ifhp, ifwp, bc, bn);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              //brgemm_kernel_acc_pixel.gemm( &gemm_param );
              brgemm_acc_pixel_tpp(private_tr_input  [i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][0],
                                   private_tr_output [i_k][i_h + pad_h_out]     [i_w + pad_w_out]     [0],
                                   scratch_float     [tid][i_k][i_c][i_r][i_s][0],
                                   w_step * h_step, /* brcount */
                                   true);

              if ((i_h == ofh - h_step) && (i_w == ofw - w_step) && (par_over_h_pixels == 0)) {
                //libxsmm_meltw_unary_param xform_param;
                //xform_param.in.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //xform_param.out.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //fp32bf16_cvt_kernel( &xform_param );
                fp32bf16_cvt_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0], (T*)(scratch_float[tid][i_k][i_c][i_r][i_s][0]));
                //xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
                //wt_vnni_kernel( &xform_param );
                wt_vnni_xform_tpp((T*)(scratch_float[tid][i_k][i_c][i_r][i_s][0]), filter[i_k][i_c][i_r][i_s][0] );
              }

            } else { /* for if use_private_trans > 0 */
              int i_c = ind[0], i_k = ind[1], i_h = ind[2], i_w = ind[3], i_r = ind[4], i_s = ind[5];
              int tid = omp_get_thread_num();
              //libxsmm_gemm_param gemm_param;

#if 0
              DECL_VLA_PTR_PT    (T,     tr_input,      [ifhp][ifwp][bc]  [bn],   t_tr_input);
              DECL_VLA_PTR_PT    (T,     tr_output,     [ofhp][ofwp][bn]  [bk],   t_tr_output);
              DECL_VLA_PTR_PT    (float, scratch_float, [Kb][Cb][R][S][bc][bk],   t_scratch_float);
#else
              DECL_VLA_PTR_PT_EXT_CAST    (T, unsigned char,     tr_input,      [ifhp][ifwp][bc]  [bn],   t_scratch_experimental, tr_input_offset);
              DECL_VLA_PTR_PT_EXT_CAST    (T, unsigned char,     tr_output,     [ofhp][ofwp][bn]  [bk],   t_scratch_experimental, tr_output_offset);
              DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float, [Kb][Cb][R][S][bc][bk],   t_scratch_experimental, scratch_float_offset);
#endif
              DECL_VLA_PTR_PT    (T,     filter,        [Cb][R][S][bc][bk],       t_grad_weight);


              if (i_h == 0 && i_w == 0 && par_over_h_pixels == 0) {
                //libxsmm_meltw_unary_param zero_param;
                //zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //zero_kernel( &zero_param );
                zero_float_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0]);
              }
/*
              if (i_c == 0 && i_k == 0 && i_r == 0 && i_s == 0 && i_h == 0 && i_w == 0) {
                for (int i = 0; i < 10; i++) {
                  float ftmp = scratch_float[tid][i_k][i_c][i_r][i_s][0][0];
                  printf("i = %d scratch_float after zero = %f \n", i, ftmp);
                }
              }
*/
              //unsigned long long brcount = w_step*h_step;
              //gemm_param.op.tertiary = (void*)&brcount;
              //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_output_libxsmm, i_k, i_h + pad_h_out, i_w + pad_w_out, 0, 0, ofhp, ofwp, bn, bk);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), tr_input_libxsmm, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, 0, ifhp, ifwp, bc, bn);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
              //brgemm_kernel_acc_pixel.gemm( &gemm_param );
/*
                if (i_c == 0 && i_k == 0 && i_r == 0 && i_s == 0 && ((i_h == 0 && i_w == 0) || (i_h == ofh/4 && i_w == ofw/4) || (i_h == ofh - h_step && i_w == ofw - w_step))) {
                  for (int i = 0; i < 10; i++) {
                    float ftmp = 0.0f;
                    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&tr_input     [i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][0][i]), &ftmp, 1);
                    printf("i = %d tr_input = %f \n", i, ftmp);
                  }
                }
                if (i_c == 0 && i_k == 0 && i_r == 0 && i_s == 0 && ((i_h == 0 && i_w == 0) || (i_h == ofh/4 && i_w == ofw/4) || (i_h == ofh - h_step && i_w == ofw - w_step))) {
                  for (int i = 0; i < 10; i++) {
                    float ftmp = 0.0f;
                    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&tr_output    [i_k][i_h + pad_h_out]     [i_w + pad_w_out]     [0][i]), &ftmp, 1);
                    printf("i = %d tr_output = %f \n", i, ftmp);
                  }
                }
*/
              brgemm_acc_pixel_tpp(tr_input     [i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][0],
                                   tr_output    [i_k][i_h + pad_h_out]     [i_w + pad_w_out]     [0],
                                   scratch_float[tid][i_k][i_c][i_r][i_s][0],
                                   w_step * h_step, /* brcount */
                                   true);
/*
              if (i_c == 0 && i_k == 0 && i_r == 0 && i_s == 0) {
                for (int i = 0; i < 10; i++) {
                  float ftmp = scratch_float[tid][i_k][i_c][i_r][i_s][0][0];
                  printf("i = %d scratch_float after brgemm = %f \n", i, ftmp);
                }
              }
*/
              if ((i_h == ofh - h_step) && (i_w == ofw - w_step) && (par_over_h_pixels == 0)) {
                //libxsmm_meltw_unary_param xform_param;
                //xform_param.in.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm,  tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //xform_param.out.primary = LIBXSMM_ACCESS_RAW(7, sizeof(float), (float*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
                //fp32bf16_cvt_kernel( &xform_param );
/*
              if (i_c == 0 && i_k == 0 && i_r == 0 && i_s == 0) {
                for (int i = 0; i < 10; i++) {
                  float ftmp = scratch_float[tid][i_k][i_c][i_r][i_s][0][0];
                  printf("i = %d scratch_float before cvt = %f \n", i, ftmp);
                }
              }
*/
                fp32bf16_cvt_tpp(scratch_float[tid][i_k][i_c][i_r][i_s][0], (T*)(scratch_float[tid][i_k][i_c][i_r][i_s][0]));
                //xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
                //wt_vnni_kernel( &xform_param );
                wt_vnni_xform_tpp((T*)(scratch_float[tid][i_k][i_c][i_r][i_s][0]), filter[i_k][i_c][i_r][i_s][0] );
/*
                if (i_c == 0 && i_k == 0 && i_r == 0 && i_s == 0) {
                  for (int i = 0; i < bk*gemm_n; i++) {
                    float ftmp = 0.0f;
                    libxsmm_convert_bf16_f32( (libxsmm_bfloat16*)(&filter[i_k][i_c][i_r][i_s][0][i]), &ftmp, 1);
                    printf("i = %d upconverted filter = %10.10f \n", i, ftmp);
                  }
                }
*/
              }
            } /* for if-else use_private_trans > 0 */
          },
          [&]() {if (sizeof(T) == 2) brgemm_acc_pixel_tpp.config();},
          [&]() {if (sizeof(T) == 2) brgemm_acc_pixel_tpp.release();});
        /* end of gemm_loop_bf16 definition */
//#endif
#endif
//#if 0
        if (par_over_h_pixels > 0) {
          reduce_wt_loop(
            [&](int* ind) {
              int i_n = ind[0];

#if 0
              DECL_VLA_PTR_PT    (float, scratch_float_2d,        [chunk0],               t_scratch_float);
              DECL_VLA_PTR_PT    (T,     scratch_bf16_weight_2d,  [chunk0],               t_scratch_bf16_weight);
#else
              DECL_VLA_PTR_PT_EXT_CAST    (float, unsigned char, scratch_float_2d,        [chunk0], t_scratch_experimental,              scratch_float_offset);
              DECL_VLA_PTR_PT_EXT_CAST    (T,     unsigned char, scratch_bf16_weight_2d,  [chunk0], t_scratch_experimental,              scratch_bf16_weight_offset);
#endif
              //libxsmm_meltw_unary_param reduce_param;
              //reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(float), (float*)scratch_libxsmm, i_n, 0, chunk0);
              //reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm_bf16_weights, i_n, 0, chunk0);
              if (i_n < reduce_work_tripcount - 1) {
                //wt_reduce_kernel0_f32bf16( &reduce_param );
                wt_reduce0_float_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
              } else {
                //wt_reduce_kernel1_f32bf16( &reduce_param );
                wt_reduce1_float_tpp(scratch_float_2d[i_n], scratch_bf16_weight_2d[i_n]);
              }
            },
            [&]() {},
            [&]() {});

          vnni_wt_loop(
            [&](int* ind) {
              int i_k = ind[0], i_c = ind[1], i_r = ind[2], i_s = ind[3];

#if 0
              DECL_VLA_PTR_PT    (T,     scratch_bf16_weight, [Cb][R][S][bc][bk],     t_scratch_bf16_weight);
#else
              DECL_VLA_PTR_PT_EXT_CAST    (T,     unsigned char, scratch_bf16_weight,  [Cb][R][S][bc][bk], t_scratch_experimental,              scratch_bf16_weight_offset);
#endif
              DECL_VLA_PTR_PT    (T,     filter,              [Cb][R][S][bc][bk],     t_grad_weight);

              //libxsmm_meltw_unary_param xform_param;
              //xform_param.in.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), scratch_libxsmm_bf16_weights, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              //xform_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
              //wt_vnni_kernel( &xform_param );
              wt_vnni_xform_tpp(scratch_bf16_weight[i_k][i_c][i_r][i_s][0], filter[i_k][i_c][i_r][i_s][0] );
            },
            [&]() {},
            [&]() {});
        } /* par_over_h_pixels if-condition */
//#endif
#endif /* for #if 1 - else  around new/old bf16 upd code */
      } /* if-else over T */

    } /* end of the scope with recorded parallel for */
  } /* end of the conv_bwd_upd scope */

//return std::vector<at::Tensor>({t_grad_input, t_grad_weight});
//#if 0

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

  //void *zero_tpp, *brgemm_tpp
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

    //printf("Didn't allocate A_offsets and B_offsets");
    //            printf("A_offsets get = %p B_offsets get = %p \n", A_offsets.get(), B_offsets.get());

  } else {
    //printf("Not implemented (missing support for LIBXSMM_GEMM_BATCH_REDUCE_OFFSET for now \n");
    //exit(-1);

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
          /* printf("A_offsets[%d] = %llu B_offsets[%d] = %llu \n", i, A_offsets[i], i, B_offsets[i]); */
          i++;
        }
      }
    } /* outer loop for filling the offsets */
    //printf("Allocated and initialized A_offsets and B_offsets");
    //            printf("A_offsets get = %p B_offsets get = %p \n", A_offsets.get(), B_offsets.get());
  }

  auto wt_trans_loop = ThreadedLoop<4>({
      LoopSpecs{0, Kb, 1, false}, // true},
      LoopSpecs{0, Cb, 1, false},//, true},
      LoopSpecs{0, R, 1, false},//, true},
      LoopSpecs{0, S, 1, false}},//, true}},
      "ABCD");

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
          //libxsmm_meltw_unary_param trans_param;
          //trans_param.in.primary  = LIBXSMM_ACCESS_RAW(6, sizeof(DType),    filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
          //trans_param.out.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), tr_filter_libxsmm, i_c, i_k, R-1-i_r, S-1-i_s, 0, 0, Kb, R, S, bk, bc);
          //wt_trans_kernel(&trans_param);
          wt_trans_tpp(weight[i_k][i_c][i_r][i_s][0], weight_tr[i_c][i_k][R-1-i_r][S-1-i_s][0]);
        },
        [&]() {},
        [&]() {});

      conv_bwd_d_loop(
        [&](int* ind) {
          int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

          //DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
          DECL_VLA_PTR_PT    (T,    gradout,     [Kb][ofhp][ofwp][bk],   t_GO);
          //DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
          DECL_VLA_PTR_PT_EXT(T,    gradout_off, [Kb][ofhp][ofwp][bk],   t_GO, (pad_h_out * ofwp * bk + pad_w_out * bk));
          //DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
          DECL_VLA_PTR_PT    (T,    dinp,        [Cb][ifhp][ifwp][bc],   t_grad_input);
          //DType *input_libxsmm_off= (DType*)input_libxsmm + (size_t) (pad_h_in * ifwp * bc + pad_w_in * bc);
          DECL_VLA_PTR_PT_EXT(T,    dinp_off,    [Cb][ifhp][ifwp][bc],   t_grad_input, (pad_h_in * ifwp * bc + pad_w_in * bc));
          //DType *tr_filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
          DECL_VLA_PTR_PT    (T,    weight_tr,   [Kb][R][S][bk][bc],     t_WT);

          if (avoid_rim_fmas == 0) {
            if (non_1x1_with_strides == 0) {
              //unsigned long long brcount = Kb_step * r_step * s_step;
              //libxsmm_gemm_param gemm_param;
              //gemm_param.op.tertiary = (void*)&brcount;
              //gemm_param.a.secondary = (void*)A_offsets;
              //gemm_param.b.secondary = (void*)B_offsets;      
              //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), tr_filter_libxsmm, i_c, i_k, i_r, i_s, 0, 0, Kb, R, S, bk, bc);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);  
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc); 
              if (i_k == 0 && i_r == 0 && i_s == 0) {
                if (stride_h != 1) {
                  if (i_w == 0 && i_h == 0) {
                    //libxsmm_meltw_unary_param zero_param;
                    //zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, 0, 0, 0, Cb, ifhp, ifwp, bc);
                    //zero_kernel_all_pixels( &zero_param );
                    zero_all_pixels_tpp(dinp[i_n][i_c][0][0]);
                  }
                } else {
                  //libxsmm_meltw_unary_param zero_param;
                  //zero_param.out.primary = (void*)gemm_param.c.primary;
                  //zero_kernel( &zero_param );
                  zero_rim_tpp(dinp_off[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
                }
              }
            /*
            if (i_n == 0 && i_c ==  0 && i_k == 0 && i_h == 0 && i_w == 0 && i_r == 0 && i_s == 0)
            {
                for (int i = 0; i < bc; i++)
                  printf("dinp_off before [off + %d] = %f \n", i, *((float*)(&(dinp_off[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][i]))) );
            }
            if (i_n == 0 && i_c ==  0 && i_k == 0 && i_h == 0 && i_w == 0 && i_r == 0 && i_s == 0)
            {
                for (int i = 0; i < bk; i++)
                  printf("gradout[%d] = %f \n", i, *((float*)(&(gradout[i_n][i_k][i_h][i_w][i]))) );
                for (int i = 0; i < bk; i++)
                  printf("weight_tr[%d] = %f \n", i, *((float*)(&( weight_tr[i_c][i_k][i_r][i_s][0][i]))) );
            }
            */

/*
              if (i_n == 0 && i_c == 0 && i_k == 0 && i_r == 0 && i_s == 0 && i_h == 0 && i_w == 0) {
                printf("count = %d \n", Kb_step * r_step * s_step);
                printf("A_offsets get = %p B_offsets get = %p \n", A_offsets.get(), B_offsets.get());
                for (int i = 0; i < 10; i++) {
                  float ftmp = *(((float*)&(gradout  [i_n][i_k][i_h][i_w])));
                  printf("i = %d gradout = %f \n", i, ftmp);
                }
                for (int i = 0; i < 10; i++) {
                  float ftmp = *(((float*)&(weight_tr[i_c][i_k][i_r][i_s][0])));
                  printf("i = %d weight_tr = %f \n", i, ftmp);
                }
                for (int i = 0; i < 10; i++) {
                  float ftmp = *(((float*)&(dinp_off [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s])));
                  printf("i = %d dinp_off before = %f \n", i, ftmp);
                }
              }
*/
              //brgemm_kernel.gemm( &gemm_param );
              brgemm_tpp(gradout  [i_n][i_k][i_h][i_w],
                         weight_tr[i_c][i_k][i_r][i_s][0],
                         dinp_off [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         B_offsets.get(), A_offsets.get(),
                         Kb_step * r_step * s_step,
                         true);
/*
              if (i_n == 0 && i_c == 0 && i_k == 0 && i_r == 0 && i_s == 0 && i_h == 0 && i_w == 0) {
                for (int i = 0; i < 10; i++) {
                  float ftmp = *(((float*)&(dinp_off [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s])));
                  printf("i = %d dinp_off after = %f \n", i, ftmp);
                }
              }
*/
            /*
            if (i_n == 0 && i_c ==  0 && i_k == 0 && i_h == 0 && i_w == 0 && i_r == 0 && i_s == 0)
            {
                for (int i = 0; i < bc; i++)
                  printf("dinp_off[off + %d] = %f \n", i, *((float*)(&(dinp_off[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s][i]))) );
            }
            */

            } else { /* for non_1x1_with_strides == 0 */
              //unsigned long long brcount = Kb_step * r_step * s_step;
              //libxsmm_gemm_param gemm_param;
              //gemm_param.op.tertiary = (void*)&brcount;
              //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), tr_filter_libxsmm, i_c, i_k, R-1-i_r, S-1-i_s, 0, 0, Kb, R, S, bk, bc);
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);  
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc); 
              if (i_k == 0 && i_r == 0 && i_s == 0) {
                if (stride_h != 1) {
                  if (i_w == 0 && i_h == 0) {
                    //libxsmm_meltw_unary_param zero_param;
                    //zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, 0, 0, 0, Cb, ifhp, ifwp, bc);
                    //zero_kernel_all_pixels( &zero_param );
                    zero_all_pixels_tpp(dinp[i_n][i_c][0][0]);
                  }
                } else {
                  //libxsmm_meltw_unary_param zero_param;
                  //zero_param.out.primary = (void*)gemm_param.c.primary;
                  //zero_kernel( &zero_param );
                  zero_rim_tpp(dinp[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
                }
              }
              //brgemm_kernel.gemm( &gemm_param );
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
                      //libxsmm_meltw_unary_param zero_param;
                      //zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, ij, ii, 0, Cb, ifhp, ifwp, bc);
                      //zero_kernel_bc( &zero_param );
                      zero_bc_tpp(dinp[i_n][i_c][ij][ii]);
                    }
                  }
                }
              }
            } /* else-if for non_1x1_with_strides == 0 */
          } else { /* avoid_rim_fmas == 0 */
            //unsigned long long brcount = Kb_step;
            //libxsmm_gemm_param gemm_param;
            //gemm_param.op.tertiary = (void*)&brcount;
            //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), tr_filter_libxsmm, i_c, i_k, i_r, i_s, 0, 0, Kb, R, S, bk, bc);      
            if (i_k == 0 && i_r == 0 && i_s == 0) {
              //libxsmm_meltw_unary_param zero_param;
              //zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h * stride_h, i_w * stride_w , 0, Cb, ifhp, ifwp, bc);
              //zero_kernel( &zero_param );
              zero_rim_tpp(dinp_off[i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s]);
            }
            if (i_r == 0 && i_h == 0) {
              /* Do no FLOPS  */
            } else if (i_r == R-1 && i_h == ofh-1 ) {
              /* Do no FLOPS  */
            } else if ( i_w == 0 && i_s == 0 ) {
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, i_h + i_r, i_w + i_s + 1, 0, Kb, ofhp, ofwp, bk);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h, i_w + 1, 0, Cb, ifhp, ifwp, bc);       
              //brgemm_kernel2.gemm( &gemm_param );
              brgemm2_tpp(gradout  [i_n][i_k][i_h + i_r][i_w + i_s + 1],
                          weight_tr[i_c][i_k][i_r][i_s][0],
                          dinp_off [i_n][i_c][i_h][i_w + 1],
                          Kb_step,
                          true);
            } else if ( i_w + w_step == ofw  && i_s == S-1) {
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, i_h + i_r, i_w + i_s, 0, Kb, ofhp, ofwp, bk);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h, i_w, 0, Cb, ifhp, ifwp, bk);
              //brgemm_kernel2.gemm( &gemm_param );
              brgemm2_tpp(gradout  [i_n][i_k][i_h + i_r][i_w + i_s],
                          weight_tr[i_c][i_k][i_r][i_s][0],
                          dinp_off [i_n][i_c][i_h][i_w],
                          Kb_step,
                          true);
            } else {
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm, i_n, i_k, i_h + i_r, i_w + i_s, 0, Kb, ofhp, ofwp, bk);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm_off, i_n, i_c, i_h, i_w, 0, Cb, ifhp, ifwp, bc);    
              //brgemm_kernel.gemm( &gemm_param );
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

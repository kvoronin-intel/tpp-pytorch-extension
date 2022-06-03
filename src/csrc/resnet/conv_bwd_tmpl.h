RECORD_FUNCTION("conv_bwd", std::vector<c10::IValue>());

// ( grad_output, input, weight) = inputs

auto t_GO = inputs[0]; // [N][Kb][H][W][bk]
auto t_I  = inputs[1]; // [N][Cb][H][W][bc]
auto t_W  = inputs[2];

auto sizes = t_I.sizes();
std::cout << "t_I sizes = " << t_I.sizes() << std::endl;

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
const long CP = sizes[1];
//const long H  = sizes[2] - 2 * pad_h_in;
//const long W  = sizes[3] - 2 * pad_w_in;
//const long bc = sizes[4];

std::vector<long> output_size{N, Kb, ofhp, ofwp, bk};
std::cout << "output_size = " << output_size << std::endl;

std::cout << "CP Cb bc Kb bk = " << CP << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;

std::vector<long> weight_tr_size{Cb, Kb, R, S, bk, bc};
std::cout << "weight_tr_size = " << weight_tr_size << std::endl;

auto t_grad_input  = at::empty(t_I.sizes(), torch::TensorOptions().dtype(t_I.dtype()));
auto t_grad_weight = at::empty(t_W.sizes(), torch::TensorOptions().dtype(t_W.dtype()));
auto t_WT          = at::empty(weight_tr_size, torch::TensorOptions().dtype(t_W.dtype()));

//  DType *scratch_libxsmm = (DType*)libxsmm_aligned_malloc( nThreads*C*K*R*S*sizeof(DType), 2097152);
std::vector<long> scratch_size{nThreads, C, K, R, S};

at::Tensor t_scratch;
if (sizeof(T) == 4)
  t_scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kFloat));
else /* bfloat16 */
  t_scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kBFloat16)); /* Hopefully, not a problem */


at::Tensor t_private_tr_input, t_private_tr_output;
if (sizeof(T) == 2) {
  t_private_tr_input  = at::empty({nThreads, N, ifhp, ifwp, C}, torch::TensorOptions().dtype(at::kBFloat16));
  t_private_tr_output = at::empty({nThreads, N, ofhp, ofwp, K}, torch::TensorOptions().dtype(at::kBFloat16));
  //DType **private_tr_input_libxsmm  = (DType**)libxsmm_aligned_malloc( nThreads*sizeof(DType*), 2097152);
  //DType **private_tr_output_libxsmm = (DType**)libxsmm_aligned_malloc( nThreads*sizeof(DType*), 2097152);
  //for (int thr = 0; thr < nThreads; thr++) {
  //  private_tr_input_libxsmm[thr] = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  //  private_tr_output_libxsmm[thr] = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  //}
}

at::Tensor t_tr_input, t_tr_output;
if (sizeof(T) == 2) {
  //DType *tr_input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  //DType *tr_output_libxsmm = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  t_tr_input  = at::empty({N, ifhp, ifwp, C}, torch::TensorOptions().dtype(at::kBFloat16));
  t_tr_output = at::empty({N, ofhp, ofwp, K}, torch::TensorOptions().dtype(at::kBFloat16));
}

at::Tensor t_scratch_float, t_scratch_bf16_weight;
if (sizeof(T) == 2) {
  //float *scratch_libxsmm = (float*)libxsmm_aligned_malloc( nThreads*C*K*R*S*sizeof(float), 2097152);
  //libxsmm_bfloat16 *scratch_libxsmm_bf16_weights = (libxsmm_bfloat16*)libxsmm_aligned_malloc(C*K*R*S*sizeof(libxsmm_bfloat16), 2097152);
  t_scratch_float       = at::empty({nThreads, C, K, R, S}, torch::TensorOptions().dtype(at::kFloat));
  t_scratch_bf16_weight = at::empty({C, K, R, S},           torch::TensorOptions().dtype(at::kBFloat16));
}

//return std::vector<at::Tensor>({t_grad_input, t_grad_weight});

{ /* main dummy scope */

  long use_mb_par = 1;

  long fm_blocking = (bk % 16 == 0) ? 16 : bk;
  long reduce_work = Kb * C * R * S * (bk/fm_blocking);
  long reduce_chunk_size = (reduce_work + nThreads - 1)/nThreads;
  long reduce_work_tripcount = (reduce_work + reduce_chunk_size - 1) / reduce_chunk_size;

  long chunk0 = reduce_chunk_size * fm_blocking;
  long chunk1 = K * C * R * S  - (reduce_work_tripcount-1) * chunk0;
  chunk1 = (chunk1 <= 0) ? chunk0 : chunk1;

  int gemm_m = 0, gemm_n = 0, gemm_k = 0;

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
    printf("trans_xform_kernel:\n");
    printf("tr_unary_shape m n ldi ldo = %d %d %d %d \n", tr_unary_shape.m, tr_unary_shape.n, tr_unary_shape.ldi, tr_unary_shape.ldo);
    trans_xform_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, K*ofhp*ofwp, bk, dtype, dtype, dtype);
    printf("vnni_xform_kernel:\n");
    printf("tr_unary_shape m n ldi ldo = %d %d %d %d \n", tr_unary_shape.m, tr_unary_shape.n, tr_unary_shape.ldi, tr_unary_shape.ldo);
    vnni_xform_kernel =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    /* Tr input is : Cb ifhp ifwp bc bn  */
    /* Tr output is: Kb ofhp ofwp bn/2 bk 2  */
    gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );

    printf("brgemm_kernel_acc_pixel:\n");
    printf("l_shape       m n k lda ldb ldc = %d %d %d %d %d %d\n", l_shape.m, l_shape.n, l_shape.k, l_shape.lda, l_shape.ldb, l_shape.ldc);
    printf("l_flags = %d \n", l_flags);

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
    printf("wt_vnni_kernel:\n");
    printf("l_unary_shape m n ldi ldo = %d %d %d %d \n", l_unary_shape.m, l_unary_shape.n, l_unary_shape.ldi, l_unary_shape.ldo);

    l_unary_shape = libxsmm_create_meltw_unary_shape(chunk0, nThreads, K * C *R * S, chunk0, LIBXSMM_DATATYPE_F32, dtype, LIBXSMM_DATATYPE_F32);
    wt_reduce_kernel0_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    l_unary_shape.m         = chunk1;
    l_unary_shape.ldo       = chunk1;
    wt_reduce_kernel1_f32bf16 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ; 
    printf("l_unary_shape m n ldi ldo = %d %d %d %d \n", l_unary_shape.m, l_unary_shape.n, l_unary_shape.ldi, l_unary_shape.ldo);
  }
#endif

  if (sizeof(T) == 4) {
//#if 0
    gemm_n = bc;
    gemm_m = bk;
    gemm_k = ofw;

    std::cout << "gemm_n gemm_m gemm_k for bwd_upd = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;

    //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, dtype, dtype, dtype);
    //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);

    //std::cout << "Got here  0 " << std::endl;

    //l_unary_shape.m         = chunk0;
    //l_unary_shape.n         = nThreads;
    //l_unary_shape.ldi       = K * C * R * S ;
    //l_unary_shape.ldo       = chunk0;
    //wt_reduce_kernel0_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce0_T_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(nThreads, chunk0, K*C*R*S, chunk0)), EW_RED);

    //std::cout << "Got here  1 " << std::endl;
    //l_unary_shape.m         = chunk1;
    //l_unary_shape.ldo       = chunk1;
    //wt_reduce_kernel1_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce1_T_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(nThreads, chunk1, K*C*R*S, chunk1)), EW_RED);

    //std::cout << "Got here  2 " << std::endl;

    //auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    gemm_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* irrelevant strides */ 1, 1, bc*stride_w, bk, bk, 1.0, 1 /*a_trans*/, 0)));//, BRGEMM);

    //std::cout << "Got here  3 " << std::endl;
//#endif
  } else {

    //printf("Untested so far!\n");
    //exit(-1);

    gemm_n = bc;
    gemm_m = bk;
    gemm_k = bn;

    //auto tr_unary_shape = libxsmm_create_meltw_unary_shape(bc, bn, C*ifhp*ifwp, bn, dtype, dtype, dtype);
    //trans_xform_kernel = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    std::cout << "trans_xform_tpp " << std::endl;
    trans_xform_tpp = SCOPEIT(XformExtTPP<T>(bn, bc, bc, bn, C*ifhp*ifwp, bn, XformTPP::XFORM_XPOSE_TPP, false), XPOSE); /* assuming row-major-ness */

    //tr_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, K*ofhp*ofwp, bk, dtype, dtype, dtype);
    //vnni_xform_kernel =  libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, tr_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE );
    std::cout << "vnni_xform_tpp " << std::endl;
    vnni_xform_tpp = SCOPEIT(XformExtTPP<T>(bn, bk, bn, bk, K*ofhp*ofwp, bk, XformTPP::XFORM_N2V_TPP, false), XPOSE); /* assuming row-major-ness */
    /* Tr input is : Cb ifhp ifwp bc bn  */
    /* Tr output is: Kb ofhp ofwp bn/2 bk 2  */

    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bn, bk, dtype, dtype, LIBXSMM_DATATYPE_F32, dtype);
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    //tileconfig_kernel.gemm  = libxsmm_dispatch_gemm_v2( l_shape, l_tc_flags, l_prefetch_flags );
    //tilerelease_kernel.gemm = libxsmm_dispatch_gemm_v2( l_shape, l_tr_flags, l_prefetch_flags );
    //gemm_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* irrelevant strides */ 1, 1, bn, bk, bk, 1.0, 0 /*a_trans*/, 0)));//, BRGEMM);

    std::cout << "brgemm_acc_pixel_tpp " << std::endl;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bn*bk*sizeof(DType), stride_w*bc*bn*sizeof(DType), 0 );
    //brgemm_kernel_acc_pixel.gemm  = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm_acc_pixel_tpp = SCOPEITGEMM((BrgemmTPP<T,float>(gemm_n, gemm_m, gemm_k, stride_w*bc*bn, bn*bk, bn, bk, bk, 1.0, 0 /*a_trans*/, 0)));//, BRGEMM);

    std::cout << "zero_bf16_tpp " << std::endl;
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
    std::cout << "wt_vnni_xform_tpp " << std::endl;
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
  } /* if-else over T */

  // JIT requested nested loop specs
  long n_step = 1;
  long c_step = 1;
  long k_step = 1;
  long h_step = 1;
  long w_step = ofw;
  long r_step = 1;
  long s_step = 1;

  std::cout << "debug: fm_blocking reduce_work reduce_work_tripcount chunk0 chunk1 = " << fm_blocking << " " <<  reduce_work << " " << reduce_work_tripcount << " " << chunk0 << " " << chunk1 << std::endl;

  std::cout << "debug: N = nThreads? n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " = " << nThreads << " " << n_step << " " << Cb << " " << c_step << " "
                                                                                                << Kb << " " << k_step << " " << ofh << " " << h_step << " "
                                                                                                << ofw << " " << w_step << " " << R << " " << r_step << " "
                                                                                                << S << " " << s_step << " " << std::endl;

  // FIXME: Is not necessary?
  DECL_VLA_PTR_PT    (T,    tmp_scratch,   [Kb][Cb][R][S][bc][bk], t_scratch);
  memset((T*)(tmp_scratch[0][0][0][0][0][0]), 0, nThreads*C*K*R*S*sizeof(T));

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
      loop_specs_str);
//#endif

  long tr_step  = 1;
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

  std::cout << "gemm_n gemm_m gemm_k for bwd_upd = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;
  std::cout << "bn bk bc = " << bn << " " << bk << " " << bc << std::endl;
  std::cout << "bf16_upfront_trans = " << bf16_upfront_trans << std::endl;
  std::cout << "use_private_trans = " << use_private_trans << std::endl;
  std::cout << "use_mb_par = " << use_mb_par << std::endl;
  std::cout << "par_over_h_pixels = " << par_over_h_pixels << std::endl;

  {
    RECORD_SCOPE(conv_bwd_upd, {});
    {
      if (sizeof(T) == 4) {
//#if 0
        if (use_mb_par == 0) {
          printf("Untested so far!\n");
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

          zero_wt_loop(
            [&](int* ind) {
              int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];

              DECL_VLA_PTR_PT    (T,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch);
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
              DECL_VLA_PTR_PT    (T,    scratch,   [Kb][Cb][R][S][bc][bk], t_scratch);

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
              DECL_VLA_PTR_PT    (T,    scratch2d, [chunk0],               t_scratch);
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
              DECL_VLA_PTR_PT    (T,    tr_input,       [ifhp][ifwp][bc]  [bn],   t_tr_input);
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
              DECL_VLA_PTR_PT    (T,    tr_output,      [ofhp][ofwp][bn]  [bk],   t_tr_output);
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

              DECL_VLA_PTR_PT    (float, scratch_float, [Kb][Cb][R][S][bc][bk], t_scratch_float);
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

#else
        conv_loop_bf16(
          [&](int* ind) {
            if (use_private_trans > 0) {
              int i_c = ind[0], i_k = ind[1], i_h = ind[2], i_w = ind[3], i_r = ind[4], i_s = ind[5];
              int tid = omp_get_thread_num();

              DECL_VLA_PTR_PT    (T,     input,             [Cb][ifhp][ifwp][bc],   t_I);
              DECL_VLA_PTR_PT    (T,     output,            [Kb][ofhp][ofwp][bk],   t_GO);
              DECL_VLA_PTR_PT    (float, scratch_float,     [Kb][Cb][R][S][bc][bk], t_scratch_float);
              DECL_VLA_PTR_PT    (T,     filter,            [Cb][R][S][bc][bk],     t_grad_weight);

              DECL_VLA_PTR_PT_EXT(T,     private_tr_input,  [ifhp][ifwp][bc][bn],   t_private_tr_input,  tid*(N*ifhp*ifwp*C));
              DECL_VLA_PTR_PT_EXT(T,     private_tr_output, [ofhp][ofwp][bn][bk],   t_private_tr_output, tid*(N*ofhp*ofwp*K));

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

              DECL_VLA_PTR_PT    (T,     tr_input,      [ifhp][ifwp][bc]  [bn],   t_tr_input);
              DECL_VLA_PTR_PT    (T,     tr_output,     [ofhp][ofwp][bn]  [bk],   t_tr_output);
              DECL_VLA_PTR_PT    (float, scratch_float, [Kb][Cb][R][S][bc][bk],   t_scratch_float);
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

              DECL_VLA_PTR_PT    (float, scratch_float_2d,        [chunk0],               t_scratch_float);
              DECL_VLA_PTR_PT    (T,     scratch_bf16_weight_2d,  [chunk0],               t_scratch_bf16_weight);
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

              DECL_VLA_PTR_PT    (T,     scratch_bf16_weight, [Cb][R][S][bc][bk],     t_scratch_bf16_weight);
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

  std::cout << "gemm_n gemm_m gemm_k for bwd_d = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;
  std::cout << "avoid_rim_fmas, non_1x1_with_strides = " << avoid_rim_fmas << " " << non_1x1_with_strides << std::endl;

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

              //brgemm_kernel.gemm( &gemm_param );
              brgemm_tpp(gradout  [i_n][i_k][i_h][i_w],
                         weight_tr[i_c][i_k][i_r][i_s][0],
                         dinp_off [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         B_offsets.get(), A_offsets.get(),
                         Kb_step * r_step * s_step,
                         true);

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

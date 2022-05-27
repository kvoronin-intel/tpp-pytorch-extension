RECORD_FUNCTION("conv_bwd", std::vector<c10::IValue>());

// ( grad_output, input, weight) = inputs

auto t_GO = inputs[0]; // [N][CP][H][W][bc]
auto t_I  = inputs[1]; // [N][CP][H][W][bc]
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
  t_scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kHalf)); /* Hopefully, not a problem */

{

  long use_mb_par = 1;

  long fm_blocking = (bk % 16 == 0) ? 16 : bk;
  long reduce_work = Kb * C * R * S * (bk/fm_blocking);
  long reduce_chunk_size = (reduce_work + nThreads - 1)/nThreads;
  long reduce_work_tripcount = (reduce_work + reduce_chunk_size - 1) / reduce_chunk_size;

  long chunk0 = reduce_chunk_size * fm_blocking;
  long chunk1 = K * C * R * S  - (reduce_work_tripcount-1) * chunk0;
  chunk1 = (chunk1 <= 0) ? chunk0 : chunk1;

  int gemm_m = 0, gemm_n = 0, gemm_k = 0;

  SCOPEITGEMM_DECL(BrgemmTPP<T, T>) gemm_as_brgemm_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>) zero_tpp;
  SCOPEIT_DECL(ReduceAddColExtTPP<T,T>) wt_reduce0_tpp, wt_reduce1_tpp;

  if (sizeof(T) == 4) {
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
    wt_reduce0_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(nThreads, chunk0, K*C*R*S, chunk0)), EW_RED);

    //std::cout << "Got here  1 " << std::endl;
    //l_unary_shape.m         = chunk1;
    //l_unary_shape.ldo       = chunk1;
    //wt_reduce_kernel1_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
    wt_reduce1_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(nThreads, chunk1, K*C*R*S, chunk1)), EW_RED);

    //std::cout << "Got here  2 " << std::endl;

    //auto l_flags    = (sizeof(DType) == 2) ? ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG ) : LIBXSMM_GEMM_FLAGS('N', 'T');
    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
    gemm_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* irrelevant strides */ 1, 1, bc*stride_w, bk, bk, 1.0, 1 /*a_trans*/, 0)));//, BRGEMM);

    //std::cout << "Got here  3 " << std::endl;

  } else {
    printf("Untested so far!\n");
    exit(-1);
  }

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

  DECL_VLA_PTR_PT    (T,    tmp_scratch,   [Kb][Cb][R][S][bc][bk], t_scratch);
  memset((T*)(tmp_scratch[0][0][0][0][0][0]), 0, nThreads*C*K*R*S*sizeof(T));

  auto zero_wt_loop = ThreadedLoop<5>({
      LoopSpecs{0, nThreads, 1, false},// true}, 
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      "Abcde");

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

  auto reduce_wt_loop = ThreadedLoop<1>({
      LoopSpecs{0, reduce_work_tripcount, 1, false}},// true}},
      "A");

  {
    RECORD_SCOPE(conv_bwd_upd, {});
    {

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
            DECL_VLA_PTR_PT    (T,    weight,    [CP][R][S][bc][bk],     t_grad_weight);

            //libxsmm_gemm_param gemm_param;
            //gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
            //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
            //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
            if (i_n == 0 && i_w == 0 && i_h == 0) {
              //libxsmm_meltw_unary_param zero_param;
              //zero_param.out.primary = (void*)gemm_param.c.primary;
              //zero_kernel( &zero_param );
              zero_tpp(weight[i_k][i_c][i_r][i_s][0]);
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
              wt_reduce0_tpp( scratch2d[i_n], weight2d[i_n] );
            } else {
              //wt_reduce_kernel1_f32( &reduce_param );
              wt_reduce1_tpp( scratch2d[i_n], weight2d[i_n] );
            }
          },
          [&]() {},
          [&]() {});
      }

    } /* end of the scope with recorded parallel for */
  } /* end of the conv_bwd_upd scope */


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
  /*
 XformExtTPP(
      int in_rows,
      int in_cols,
      int out_rows,
      int out_cols,
      int ldi,
      int ldo,
      XformTPP::XFORM_TYPE xtype,
      bool ignore_vnni_for_fp32 = false)
    */
  auto wt_trans_tpp = SCOPEIT(XformExtTPP<T>(bc, bk, bk, bc, bk, bc, XformTPP::XFORM_XPOSE_TPP, false), XPOSE); /* assuming row-major-ness */

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
          }
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});

    } /* end of the scope with recorded parallel for */
  } /* end of the conv_bwd_d scope */

} /* end of the dummy scope */

//auto t_dummy     = at::empty({0},  torch::TensorOptions().dtype(at::kFloat));
//return std::vector<at::Tensor>({t_dummy, t_grad_weight});
return std::vector<at::Tensor>({t_grad_input, t_grad_weight});


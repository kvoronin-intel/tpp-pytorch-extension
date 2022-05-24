RECORD_FUNCTION("conv_bwd", std::vector<c10::IValue>());

// ( grad_output, input, weight) = inputs

auto t_GO = inputs[0]; // [N][CP][H][W][bc]
auto t_I  = inputs[1]; // [N][CP][H][W][bc]
auto t_W  = inputs[2];

auto sizes = t_I.sizes();
std::cout << "t_I sizes = " << t_I.sizes() << std::endl;

int R = cfg.R;
int S = cfg.S;
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
const long H  = sizes[2] - 2 * pad_h_in;
const long W  = sizes[3] - 2 * pad_w_in;
//const long bc = sizes[4];

std::vector<long> output_size{N, Kb, ofhp, ofwp, bk};
std::cout << "output_size = " << output_size << std::endl;

std::cout << "CP Cb bc Kb bk = " << CP << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;

//auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));

auto t_grad_weight = at::empty(t_W.sizes(), torch::TensorOptions().dtype(t_W.dtype()));

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

  if (sizeof(T) == 4) {
    gemm_n = ofw;
    gemm_m = bk;
    gemm_k = bc;
  } else {
    printf("Untested so far!\n");
    exit(-1);
  }
  //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, dtype, dtype, dtype);
  //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  auto zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);

  //l_unary_shape.m         = chunk0;
  //l_unary_shape.n         = nThreads;
  //l_unary_shape.ldi       = K * C * R * S ;
  //l_unary_shape.ldo       = chunk0;
  //wt_reduce_kernel0_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
  auto wt_reduce0_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(nThreads, chunk0, K*C*R*S, chunk0)), EW_RED);
  //l_unary_shape.m         = chunk1;
  //l_unary_shape.ldo       = chunk1;
  //wt_reduce_kernel1_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
  auto wt_reduce1_tpp = SCOPEIT((ReduceAddColExtTPP<T,T>(nThreads, chunk1, K*C*R*S, chunk1)), EW_RED);

  //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
  //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  //gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );
  auto gemm_as_brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n, gemm_m, gemm_k, /* irrelevant strides */ bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 0.0, 0, 0)));//, BRGEMM);

  // JIT requested nested loop specs
  long n_step = 1;
  long c_step = 1;
  long k_step = 1;
  long h_step = 1;
  long w_step = ofw;
  long r_step = 1;
  long s_step = 1;

  auto zero_wt_loop = ThreadedLoop<5>({
      LoopSpecs{0, nThreads, 1, true},
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      "Abcde");

  /* FIXME: Fix this! */
  char loop_specs_str[256] = "Abcdefg";

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, true},
      LoopSpecs{0, Cb, c_step, true},
      LoopSpecs{0, Kb, k_step, true},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step, true},
      LoopSpecs{0, S, s_step, true}},
      loop_specs_str);

  auto reduce_wt_loop = ThreadedLoop<1>({
      LoopSpecs{0, reduce_work_tripcount, 1, true}},
      "A");

  {
    RECORD_SCOPE(conv_bwd_upd, {});
    {

      if (use_mb_par == 0) {
        printf("Untested so far!\n");
        exit(-1);

        conv_loop(
          [&](int* ind) {
            int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

          //DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
          DECL_VLA_PTR_PT_EXT(T,    gradout,   [Kb][ofhp][ofwp][bk],   t_GO, (pad_h_out * ofwp * bk + pad_w_out * bk));
          //DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
          DECL_VLA_PTR_PT    (T,    inp,       [Cb][ifhp][ifwp][bc],   t_I);
          //DType *filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
          DECL_VLA_PTR_PT    (T,    weight,    [CP][R][S][bc][bk],     t_W);

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

      } else {

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

        conv_loop(
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

            DECL_VLA_PTR_PT    (T,    weight2d,  [chunk0],               t_W);
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

} /* end of the dummy scope */

auto t_dummy     = at::empty({0},  torch::TensorOptions().dtype(at::kFloat));
return std::vector<at::Tensor>({t_dummy, t_grad_weight});

#if 0
  float *naive_input  = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_input_nchwc  = (float*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(float), 2097152);
  float *naive_output = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_output_nchwc = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_output_opt = (float*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(float), 2097152);
  float *naive_filter = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  float *naive_filter_opt = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  float *naive_filter_kcrsck = (float*)libxsmm_aligned_malloc( C*K*R*S*sizeof(float), 2097152);
  DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
  DType *output_libxsmm = (DType*)libxsmm_aligned_malloc( N*ofhp*ofwp*K*sizeof(DType), 2097152);
  DType *filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
  DType *scratch_libxsmm = (DType*)libxsmm_aligned_malloc( nThreads*C*K*R*S*sizeof(DType), 2097152);
  DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);

  if (sizeof(DType) == 4) {
    auto gemm_n = bc;
    auto gemm_m = bk;
    auto gemm_k = ofw;
    auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, dtype, dtype, dtype);
    zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    gemm_kernel.gemm      = libxsmm_dispatch_gemm_v2( l_shape, l_flags, l_prefetch_flags );

    l_unary_shape.m         = chunk0;
    l_unary_shape.n         = nThreads;
    l_unary_shape.ldi       = K * C * R * S ;
    l_unary_shape.ldo       = chunk0;
    wt_reduce_kernel0_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ;
   
    l_unary_shape.m         = chunk1;
    l_unary_shape.ldo       = chunk1;
    wt_reduce_kernel1_f32 = libxsmm_dispatch_meltw_unary_v2( LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS ) ; 
  } 

  // JIT requested nested loop specs
  long n_step = 1;
  long c_step = 1;
  long k_step = 1;
  long h_step = 1;
  long w_step = ofw;
  long r_step = 1;
  long s_step = 1;

  auto t0 = getTime();
  auto zero_wt_loop = ThreadedLoop<5>({
      LoopSpecs{0, nThreads, 1, true},
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      "Abcde");

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, true},
      LoopSpecs{0, Cb, c_step, true},
      LoopSpecs{0, Kb, k_step, true},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step, true},
      LoopSpecs{0, S, s_step, true}},
      loop_specs_str);

  auto reduce_wt_loop = ThreadedLoop<1>({
      LoopSpecs{0, reduce_work_tripcount, 1, true}},
      "A");
  auto t1 = getTime();

  //printf("N=%d Cb=%d Kb=%d ofh=%d ofw=%d R=%d S=%d\n", N, Cb, Kb, ofh, ofw, R, S); 

  // Warmup iteration for i-caches
  if (use_mb_par == 0) {
    conv_loop(
      [&](int* ind) {
        int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];
        libxsmm_gemm_param gemm_param;
        gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
        gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
        gemm_param.c.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);        
        if (i_n == 0 && i_w == 0 && i_h == 0) {
          libxsmm_meltw_unary_param zero_param;
          zero_param.out.primary = (void*)gemm_param.c.primary;
          zero_kernel( &zero_param );
        }
        gemm_kernel.gemm( &gemm_param );
      },
      [&]() {if (sizeof(DType) == 2) tileconfig_kernel.gemm(NULL);},
      [&]() {if (sizeof(DType) == 2) tilerelease_kernel.gemm(NULL);});
  } else {
    zero_wt_loop(
      [&](int* ind) {
        int i_n = ind[0], i_k = ind[1], i_c = ind[2], i_r = ind[3], i_s = ind[4];
        libxsmm_meltw_unary_param zero_param;
        zero_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, i_n, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);
        zero_kernel( &zero_param );
      },
      [&]() {},
      [&]() {});

    conv_loop(
      [&](int* ind) {
        int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];
        int tid = omp_get_thread_num();
        libxsmm_gemm_param gemm_param;
        gemm_param.a.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
        gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
        gemm_param.c.primary = LIBXSMM_ACCESS_RAW(7, sizeof(DType), (DType*)scratch_libxsmm, tid, i_k, i_c, i_r, i_s, 0, 0, Kb, Cb, R, S, bc, bk);        
        gemm_kernel.gemm( &gemm_param );
      },
      [&]() {if (sizeof(DType) == 2) tileconfig_kernel.gemm(NULL);},
      [&]() {if (sizeof(DType) == 2) tilerelease_kernel.gemm(NULL);});

    reduce_wt_loop(
      [&](int* ind) {
        int i_n = ind[0];
        libxsmm_meltw_unary_param reduce_param;
        reduce_param.in.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)scratch_libxsmm, i_n, 0, chunk0);
        reduce_param.out.primary = (void*)LIBXSMM_ACCESS_RAW(2, sizeof(DType), (DType*)filter_libxsmm,  i_n, 0, chunk0);
        if (i_n < reduce_work_tripcount - 1) {
          wt_reduce_kernel0_f32( &reduce_param );
        } else {
          wt_reduce_kernel1_f32( &reduce_param );  
        } 
      },
      [&]() {},
      [&]() {});
  }

#endif

#if 0
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

std::cout << "t_I sizes = " << t_I.sizes() << std::endl;

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

const long hi_start      = pad_h_in;
const long hi_end        = hi_start + H;
const long wi_start      = pad_w_in;
const long wi_end        = W + pad_w_in;
const long ifhp = H + 2 * pad_h_in;
const long ifwp = W + 2 * pad_w_in;

const long ho_start      = pad_h_out;
const long wo_start      = pad_w_out;
const long ofhp = H + 2 * pad_h_out;
const long ofwp = W + 2 * pad_w_out;

const float scale = 1.0f /((float)N * H * W);

const long sum_N_offset          = LIBXSMM_UP2(CP * 2 * bc, 64);
const long sumsq_N_offset        = LIBXSMM_UP2(sum_N_offset + CP * N * bc, 64);
const long full_fwd_scratch_size = sumsq_N_offset + LIBXSMM_UP2((size_t)CP * (size_t)N * (size_t)bc, 64);

const long dbeta_N_offset        = LIBXSMM_UP2(CP * N * bc, 64);
const long full_bwd_scratch_size = dbeta_N_offset + LIBXSMM_UP2(CP * N * bc, 64);

const long full_scratch_size     = std::max(full_fwd_scratch_size, full_bwd_scratch_size);

// FIXME: Save scratch somewhere to not allocate each time
std::vector<long> scratch_size{full_scratch_size};

auto scratch = at::empty(scratch_size, torch::TensorOptions().dtype(at::kFloat));

bool use_hw_blocking = true;

const long num_HW_blocks = (H > W ? H : W);
const long num_W_blocks  = (W % 64 == 0 ? W / 64 : 1);

long spatial_block_size = 0;

if (pad_h_in != 0 || pad_w_in != 0 || pad_h_out != 0 || pad_w_out != 0 ) {
  use_hw_blocking    = false; /* alternative is w blocking ([w, bc] blocks) */
  spatial_block_size = W / num_W_blocks;
} else {
  use_hw_blocking    = true; /* using [hw, bc] blocks */
  spatial_block_size = H * W / num_HW_blocks;
}


{
#ifndef THREADED_LOOPS
  DECL_VLA_PTR_PT    (T,             inp,      [CP][ifhp][ifwp][bc], t_I);
  DECL_VLA_PTR_PT    (float,         gamma,    [bc],                 t_W);
  DECL_VLA_PTR_PT    (float,         mean,     [bc],                 t_M);
  DECL_VLA_PTR_PT    (float,         var,      [bc],                 t_V);
  DECL_VLA_PTR_PT    (float,         dgamma_N, [N][bc],              scratch);
  DECL_VLA_PTR_PT_EXT(float,         dbeta_N,  [N][bc],              scratch, dbeta_N_offset);
  DECL_VLA_PTR_PT    (T,             din,      [CP][ifhp][ifwp][bc], t_grad_input);
  DECL_VLA_PTR_PT    (T,             din_add,  [CP][ifhp][ifwp][bc], t_grad_input_add);
  DECL_VLA_PTR_PT    (float,         dgamma,   [bc],                 t_grad_weight);
  DECL_VLA_PTR_PT    (float,         dbeta,    [bc],                 t_grad_bias);

  DECL_VLA_PTR_PT_EXT(T,             dout,     [CP][ofhp][ofwp][bc],               t_GO, (ho_start * ofwp + wo_start) * bc);
  DECL_VLA_PTR_PT_EXT(unsigned char, relumask, [CP][ofhp][ofwp][bc/BITS_PER_CHAR], t_R,  (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR);

#endif

  auto zero_tpp = SCOPEIT(SetZeroTPP<float>(bc), EW_ZERO);

  auto zero_hp_tpp = SCOPEIT(SetZeroTPP<T>((pad_h_in * ifwp), bc, bc), EW_ZERO); /* (pad_h_in * ifwp), bc because of row-major for unary */

  auto zero_wp_tpp = SCOPEIT(SetZeroTPP<T>(pad_w_in, bc, bc), EW_ZERO);          /* pad_w_in, bc because of row-major for unary */

  auto coeffs_tpp = SCOPEIT(BatchNormStatCoeffsTPP<float>(bc, eps), NORMALIZE);

  auto helper_copy_tpp = SCOPEIT((CpyTPP<float>(1, bc, bc, bc)), EW_COPY); /* 1, bc because of row-major for unary */

  auto helper_add_tpp = SCOPEIT(AddTPP<float>(1, bc, bc, bc), EW_ADD); /* 1, bc because of row-major for unary */

  auto grad_w_inpadd_tpp = SCOPEIT((BatchNormBwdWTPP<T,T>(bc, spatial_block_size, relu, eltwise)), NORMALIZE);

  auto abc_coeffs_tpp = SCOPEIT(BatchNormABCCoeffsTPP<float>(bc, scale, eps), NORMALIZE);

  auto grad_d_tpp = SCOPEIT((BatchNormBwdDTPP<T,T>(bc, spatial_block_size)), NORMALIZE);

#ifdef THREADED_LOOPS
  char ncp_loop_specs_str[256] = "AB";
  const long n_step = 1, cp_step = 1;
  auto ncp_loop = ThreadedLoop<2>({
      LoopSpecs{0, N,  n_step,  {/*l1_k_step, l0_k_step*/}},   // Logical N  loop specs
      LoopSpecs{0, CP, cp_step, {/*l1_n_step, l0_n_step*/}}},  // Logical CP loop specs
      ncp_loop_specs_str);

  char cp_loop_specs_str[256] = "A";
  auto cp_loop = ThreadedLoop<1>({
      LoopSpecs{0, CP, cp_step, {/*l1_k_step, l0_k_step*/}}},  // Logical CP loop specs
      cp_loop_specs_str);
#endif

  {
    RECORD_SCOPE(bn_bwd_w_inpadd, {});
    {
#ifdef THREADED_LOOPS
      ncp_loop(
        [&](int *ind) {
          const int n = ind[0], cp = ind[1];

          DECL_VLA_PTR_PT    (T,             inp,      [CP][ifhp][ifwp][bc], t_I);
          DECL_VLA_PTR_PT    (float,         gamma,    [bc],                 t_W);
          DECL_VLA_PTR_PT    (float,         mean,     [bc],                 t_M);
          DECL_VLA_PTR_PT    (float,         var,      [bc],                 t_V);
          DECL_VLA_PTR_PT_EXT(T,             dout,     [CP][ofhp][ofwp][bc],               t_GO, (ho_start * ofwp + wo_start) * bc);
          DECL_VLA_PTR_PT_EXT(unsigned char, relumask, [CP][ofhp][ofwp][bc/BITS_PER_CHAR], t_R,  (ho_start * ofwp + wo_start) * bc/BITS_PER_CHAR);
          DECL_VLA_PTR_PT    (float,         dgamma_N, [N][bc],              scratch);
          DECL_VLA_PTR_PT_EXT(float,         dbeta_N,  [N][bc],              scratch, dbeta_N_offset);
          DECL_VLA_PTR_PT    (T,             din_add,  [CP][ifhp][ifwp][bc], t_grad_input_add);

          LIBXSMM_ALIGNED(float lcl_dgamma_ptr[bc], 64);
          LIBXSMM_ALIGNED(float lcl_dbeta_ptr[bc], 64);

          LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
          LIBXSMM_ALIGNED(float b[bc], 64);

          zero_tpp(&lcl_dgamma_ptr[0]);
          zero_tpp(&lcl_dbeta_ptr [0]);

          coeffs_tpp(mean[cp], var[cp], &a[0], &b[0]);

          if (!use_hw_blocking) {

            if (pad_h_in != 0 && eltwise ) {
              //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
              //cfg.all_zero_hp_kernel(&all_zero_param);
              zero_hp_tpp(din_add[n][cp][0][0]);
            }
            for (int ho = 0, hi = hi_start; ho < H; ho++, hi++) {
              /* zeroing out starting [0, wi_start) x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
                //cfg.all_zero_wp_kernel(&all_zero_param);
                zero_wp_tpp(din_add[n][cp][hi][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                //void operator()(Tin* inp, float *a, float *b, Tout *dout, float *dgamma_local, float *dbeta_local, float* gamma, Tin* din_add, unsigned char* relumask)
                grad_w_inpadd_tpp(inp[n][cp][hi][wi_start + wb*(W/num_W_blocks)], &a[0], &b[0], dout[n][cp][ho][wb*(W/num_W_blocks)], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][wi_start + wb*(W/num_W_blocks)] : NULL,
                                relu ? relumask[n][cp][ho][wb*(W/num_W_blocks)] : NULL);
              }
              /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
                //cfg.all_zero_wp_kernel(&all_zero_param);
                zero_wp_tpp(din_add[n][cp][hi][wi_end]);
              }

            }
            /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
            if (pad_h_in != 0 && eltwise ) {
              //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
              //cfg.all_zero_hp_kernel(&all_zero_param);
              zero_hp_tpp(din_add[n][cp][hi_end][0]);
            }

            //printf("First parallel for is not implemented for w blocking in bwd\n");
            //exit(-1);
          } else {
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int ho = (hwb*(H*W/num_HW_blocks))/W;
              int hi = ho;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              //void operator()(Tin* inp, float *a, float *b, Tout *dout, float *dgamma_local, float *dbeta_local, float* gamma, Tin* din_add, unsigned char* relumask)
              grad_w_inpadd_tpp(inp[n][cp][hi][w], &a[0], &b[0], dout[n][cp][ho][w], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][w] : NULL,
                                relu ? relumask[n][cp][ho][w] : NULL);
            }
          } /* if-else for the presence of input padding */

          helper_copy_tpp(&lcl_dgamma_ptr[0], dgamma_N[cp][n]);
          helper_copy_tpp(&lcl_dbeta_ptr[0],  dbeta_N [cp][n]);
        },
        [&]() {},
        [&]() {});
#else /* THREADED_LOOPS */
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

            if (pad_h_in != 0 && eltwise ) {
              //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
              //cfg.all_zero_hp_kernel(&all_zero_param);
              zero_hp_tpp(din_add[n][cp][0][0]);
            }
            for (int ho = 0, hi = hi_start; ho < H; ho++, hi++) {
              /* zeroing out starting [0, wi_start) x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
                //cfg.all_zero_wp_kernel(&all_zero_param);
                zero_wp_tpp(din_add[n][cp][hi][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                //void operator()(Tin* inp, float *a, float *b, Tout *dout, float *dgamma_local, float *dbeta_local, float* gamma, Tin* din_add, unsigned char* relumask)
                grad_w_inpadd_tpp(inp[n][cp][hi][wi_start + wb*(W/num_W_blocks)], &a[0], &b[0], dout[n][cp][ho][wb*(W/num_W_blocks)], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][wi_start + wb*(W/num_W_blocks)] : NULL,
                                relu ? relumask[n][cp][ho][wb*(W/num_W_blocks)] : NULL);
              }
              /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
                //cfg.all_zero_wp_kernel(&all_zero_param);
                zero_wp_tpp(din_add[n][cp][hi][wi_end]);
              }

            }
            /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
            if (pad_h_in != 0 && eltwise ) {
              //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din_add, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
              //cfg.all_zero_hp_kernel(&all_zero_param);
              zero_hp_tpp(din_add[n][cp][hi_end][0]);
            }

            //printf("First parallel for is not implemented for w blocking in bwd\n");
            //exit(-1);
          } else {
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int ho = (hwb*(H*W/num_HW_blocks))/W;
              int hi = ho;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              //void operator()(Tin* inp, float *a, float *b, Tout *dout, float *dgamma_local, float *dbeta_local, float* gamma, Tin* din_add, unsigned char* relumask)
              grad_w_inpadd_tpp(inp[n][cp][hi][w], &a[0], &b[0], dout[n][cp][ho][w], &lcl_dgamma_ptr[0], &lcl_dbeta_ptr[0], gamma[cp],
                                eltwise ? din_add[n][cp][hi][w] : NULL,
                                relu ? relumask[n][cp][ho][w] : NULL);
            }
          } /* if-else for the presence of input padding */

          helper_copy_tpp(&lcl_dgamma_ptr[0], dgamma_N[cp][n]);
          helper_copy_tpp(&lcl_dbeta_ptr[0],  dbeta_N [cp][n]);

        } /* end of cp loop */
      } /* end of n loop */
#endif /* THREADED_LOOPS */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_w_inpadd scope */

  {
    RECORD_SCOPE(bn_bwd_w_add, {});
    {
#ifdef THREADED_LOOPS
      cp_loop(
        [&](int *ind) {
          const int cp = ind[0];

          DECL_VLA_PTR_PT    (float, dgamma,   [bc],    t_grad_weight);
          DECL_VLA_PTR_PT    (float, dbeta,    [bc],    t_grad_bias);
          DECL_VLA_PTR_PT    (float, dgamma_N, [N][bc], scratch);
          DECL_VLA_PTR_PT_EXT(float, dbeta_N,  [N][bc], scratch, dbeta_N_offset);

          zero_tpp(dgamma[cp]);
          zero_tpp(dbeta [cp]);

          for(int ni = 0; ni < N; ni++) {
            helper_add_tpp(dgamma[cp], dgamma_N[cp][ni], dgamma[cp]);
            helper_add_tpp(dbeta [cp], dbeta_N [cp][ni], dbeta [cp]);
          } /* end of ni loop */
        },
        [&]() {},
        [&]() {});
#else /* THREADED_LOOPS */
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int cp = 0; cp < CP; cp++) {
        zero_tpp(dgamma[cp]);
        zero_tpp(dbeta [cp]);

        for(int ni = 0; ni < N; ni++) {
          helper_add_tpp(dgamma[cp], dgamma_N[cp][ni], dgamma[cp]);
          helper_add_tpp(dbeta [cp], dbeta_N [cp][ni], dbeta [cp]);
        } /* end of ni loop */
      } /* end of cp loop */
#endif /* THREADED_LOOPS */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_w_add scope */

  {
    RECORD_SCOPE(bn_bwd_d, {});
    {

#ifdef THREADED_LOOPS
      ncp_loop(
        [&](int *ind) {
          const int n = ind[0], cp = ind[1];

          DECL_VLA_PTR_PT    (T,             inp,      [CP][ifhp][ifwp][bc], t_I);
          DECL_VLA_PTR_PT    (T,             din,      [CP][ifhp][ifwp][bc], t_grad_input);
          DECL_VLA_PTR_PT    (float,         gamma,    [bc],                 t_W);
          DECL_VLA_PTR_PT    (float,         mean,     [bc],                 t_M);
          DECL_VLA_PTR_PT    (float,         var,      [bc],                 t_V);
          DECL_VLA_PTR_PT_EXT(T,             dout,     [CP][ofhp][ofwp][bc], t_GO, (ho_start * ofwp + wo_start) * bc);
          DECL_VLA_PTR_PT    (float,         dgamma,   [bc],                 t_grad_weight);
          DECL_VLA_PTR_PT    (float,         dbeta,    [bc],                 t_grad_bias);

          LIBXSMM_ALIGNED(float a[bc], 64); /* could also get moved into the scratch but left on the private stack as these are small, same below */
          LIBXSMM_ALIGNED(float b[bc], 64);
          LIBXSMM_ALIGNED(float c[bc], 64);

          //void operator()(Tin* gamma, Tin* dgamma, Tin* var, Tin* mean, Tin* dbeta, Tout* a, Tout* b, Tout* c)
          abc_coeffs_tpp(gamma[cp], dgamma[cp], var[cp], mean[cp], dbeta[cp], &a[0], &b[0], &c[0]);

          if (!use_hw_blocking) {
            if (pad_h_in != 0 && eltwise ) {
              //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
              //cfg.all_zero_hp_kernel(&all_zero_param);
              zero_hp_tpp(din[n][cp][0][0]);
            }
            for (int ho = 0, hi = hi_start; ho < H; ho++, hi++) {
              /* zeroing out starting [0, wi_start) x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
                //cfg.all_zero_wp_kernel(&all_zero_param);
                zero_wp_tpp(din[n][cp][hi][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                //void operator()(Tin* inp, float* a, float* b, float *c, float *gamma, Tout* dout, Tout* din)
                grad_d_tpp(inp[n][cp][hi][wi_start + wb*(W/num_W_blocks)], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][wb*(W/num_W_blocks)], din[n][cp][hi][wi_start + wb*(W/num_W_blocks)]);
              }
              /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
                //cfg.all_zero_wp_kernel(&all_zero_param);
                zero_wp_tpp(din[n][cp][hi][wi_end]);
              }

            }
            /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
            if (pad_h_in != 0 && eltwise ) {
              //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
              //cfg.all_zero_hp_kernel(&all_zero_param);
              zero_hp_tpp(din[n][cp][hi_end][0]);
            }

            //printf("Third parallel for is not implemented for w blocking in bwd\n");
            //exit(-1);
          } else {
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int ho = (hwb*(H*W/num_HW_blocks))/W;
              int hi = ho;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              //void operator()(Tin* inp, float* a, float* b, float *c, float *gamma, Tout* dout, Tout* din)
              grad_d_tpp(inp[n][cp][hi][w], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][w], din[n][cp][hi][w]);
            }
          } /* if-else for the presence of input padding */
        },
        [&]() {},
        [&]() {});
#else /* THREADED_LOOPS */
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
            if (pad_h_in != 0 && eltwise ) {
              //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, 0, 0, 0, CP, ifhp, ifwp, bc);
              //cfg.all_zero_hp_kernel(&all_zero_param);
              zero_hp_tpp(din[n][cp][0][0]);
            }
            for (int ho = 0, hi = hi_start; ho < H; ho++, hi++) {
              /* zeroing out starting [0, wi_start) x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, 0, 0, CP, ifhp, ifwp, bc);
                //cfg.all_zero_wp_kernel(&all_zero_param);
                zero_wp_tpp(din[n][cp][hi][0]);
              }
              for (int wb = 0; wb < num_W_blocks; wb++) {
                //void operator()(Tin* inp, float* a, float* b, float *c, float *gamma, Tout* dout, Tout* din)
                grad_d_tpp(inp[n][cp][hi][wi_start + wb*(W/num_W_blocks)], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][wb*(W/num_W_blocks)], din[n][cp][hi][wi_start + wb*(W/num_W_blocks)]);
              }
              /* zeroing out ending [wi_end, ifwp] x bc block for fixed hi */
              if (pad_w_in != 0 && eltwise ) {
                //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi, wi_end, 0, CP, ifhp, ifwp, bc);
                //cfg.all_zero_wp_kernel(&all_zero_param);
                zero_wp_tpp(din[n][cp][hi][wi_end]);
              }

            }
            /* zeroing out strip [hi_end, ifhp) x ifwp x bc */
            if (pad_h_in != 0 && eltwise ) {
              //all_zero_param.out.primary = &LIBXSMM_VLA_ACCESS(5, din, n, cp, hi_end, 0, 0, CP, ifhp, ifwp, bc);
              //cfg.all_zero_hp_kernel(&all_zero_param);
              zero_hp_tpp(din[n][cp][hi_end][0]);
            }

            //printf("Third parallel for is not implemented for w blocking in bwd\n");
            //exit(-1);
          } else {
            for(int hwb = 0; hwb < num_HW_blocks; hwb++){
              int ho = (hwb*(H*W/num_HW_blocks))/W;
              int hi = ho;
              int w  = (hwb*(H*W/num_HW_blocks))%W;

              //void operator()(Tin* inp, float* a, float* b, float *c, float *gamma, Tout* dout, Tout* din)
              grad_d_tpp(inp[n][cp][hi][w], &a[0], &b[0], &c[0], gamma[cp], dout[n][cp][ho][w], din[n][cp][hi][w]);
            }
          } /* if-else for the presence of input padding */
        } /* end of cp loop */
      } /* end of n loop */
#endif /* THREADED_LOOPS */
    } /* end of the scope with recorded parallel for */
  } /* end of the bn_bwd_d scope */

}

return std::vector<at::Tensor>({t_grad_input, t_grad_input_add, t_grad_weight, t_grad_bias});

#endif

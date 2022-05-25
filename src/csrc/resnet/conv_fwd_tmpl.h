
RECORD_FUNCTION("conv_fwd", std::vector<c10::IValue>());

// ( input, weight) = inputs

auto t_I  = inputs[0]; // [N][CP][H][W][bc]
auto t_W  = inputs[1];

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

const long N  = sizes[0];
const long CP = sizes[1];
const long H  = sizes[2] - 2 * pad_h_in;
const long W  = sizes[3] - 2 * pad_w_in;
//const long bc = sizes[4];

std::vector<long> output_size{N, Kb, ofhp, ofwp, bk};
std::cout << "output_size = " << output_size << std::endl;

std::cout << "CP Cb bc Kb bk = " << CP << " " << Cb << " " << bc << " " << Kb << " " << bk << std::endl;

auto t_O = at::empty(output_size, torch::TensorOptions().dtype(t_I.dtype()));

//return std::vector<at::Tensor>({t_O});

/*
class BrgemmTPP {
 public:
  BrgemmTPP() {}
  BrgemmTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      float beta = 1.0,
      int a_trans = 0,
      int unroll_hint = 0)
      : BrgemmTPP(
            M,
            N,
            K,
            str_a,
            str_b,
            (a_trans == 0 ? K : M),
            N,
            N,
            beta,
            a_trans,
            unroll_hint) {}


  void operator()(
      Tin* A,
      Tin* B,
      Tout* C,
      unsigned long long count,
      bool no_tile_cfg = false) {

*/

{

  auto gemm_n = ofw;
  auto gemm_m = bk;
  auto gemm_k = bc;

  std::cout << "gemm_n gemm_m gemm_k = " << gemm_n << " " << gemm_m << " " << gemm_k << std::endl;

  //void *brgemm_tpp, *zero_tpp;

  //ScopedTPP<BrgemmTPP<T, T>, 0> brgemm_tpp, brgemm2_tpp;
  //ScopedTPP<SetZeroTPP<T>, 0> zero_tpp;
  SCOPEITGEMM_DECL(BrgemmTPP<T, T>) brgemm_tpp, brgemm2_tpp;
  SCOPEIT_DECL(SetZeroTPP<T>) zero_tpp;

  /* n,m,k, stride_b, stride_a, ldb, lda, ldc, beta, a_trans, unroll_hint because of the row-major */
  if ((R == 1 && S == 1) || (cfg.avoid_fmas_in_rim == 1)) {
    //auto l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    //auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
    //auto l_brconfig = libxsmm_create_gemm_batch_reduce_config( LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, R*S*bc*bk*sizeof(DType), bc*ifhp*ifwp*sizeof(DType), Cb_step );
    //brgemm_kernel.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    //l_shape = libxsmm_create_gemm_shape( gemm_m, gemm_n-1, gemm_k, bk, bc*stride_w, bk, dtype, dtype, dtype, dtype );
    //brgemm_kernel2.gemm      = libxsmm_dispatch_brgemm_v2( l_shape, l_flags, l_prefetch_flags, l_brconfig );
    brgemm_tpp  = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n  , gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);
    brgemm2_tpp = SCOPEITGEMM((BrgemmTPP<T,T>(gemm_n-1, gemm_m, gemm_k, bc*ifhp*ifwp, R*S*bc*bk, bc*stride_w, bk, bk, 1.0, 0, 0)));//, BRGEMM);

    //auto l_unary_shape = libxsmm_create_meltw_unary_shape(bk*gemm_n, 1, bk*gemm_n, bk*gemm_n, dtype, dtype, dtype);
    //zero_kernel = libxsmm_dispatch_meltw_unary_v2(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    zero_tpp = SCOPEIT(SetZeroTPP<T>(bk*gemm_n), EW_ZERO);
  } else {
    std::cout << "This case has not been implemented as it requires support for brgemm with LIBXSMM_GEMM_BATCH_REDUCE_OFFSET which is absent in the functors" << std::endl;
    exit(-1);
  }

  long Cb_step = Cb;
  long n_step = 1;
  long c_step = Cb_step;
  long k_step = 1;
  long h_step = 1;
  long w_step = ofw;
  long r_step = R;
  long s_step = S;

  if (cfg.avoid_fmas_in_rim == 1) {
    r_step = 1;
    s_step = 1;
  }

  std::cout << "debug: N n_step Cb c_step Kb k_step ofh h_step ofw w_step R r_step S s_step = " << N << " " << n_step << " " << Cb << " " << c_step << " "
                                                                                                << Kb << " " << k_step << " " << ofh << " " << h_step << " "
                                                                                                << ofw << " " << w_step << " " << R << " " << r_step << " "
                                                                                                << S << " " << s_step << " " << std::endl;

  /* FIXME: Fix this! */
  char loop_specs_str[256] = "Abcdefg";
  //char loop_specs_str[256] = "ABc";

  auto conv_loop = ThreadedLoop<7>({
      LoopSpecs{0, N, n_step, true},
      LoopSpecs{0, Cb, c_step},
      LoopSpecs{0, Kb, k_step},
      LoopSpecs{0, ofh, h_step},
      LoopSpecs{0, ofw, w_step},
      LoopSpecs{0, R, r_step},
      LoopSpecs{0, S, s_step}},
      loop_specs_str);

  {
    RECORD_SCOPE(conv_fwd, {});
    {
//#if 0
      conv_loop(
        [&](int* ind) {
          int i_n = ind[0], i_c = ind[1], i_k = ind[2], i_h = ind[3], i_w = ind[4], i_r = ind[5], i_s = ind[6];

          //DType *output_libxsmm_off= (DType*)output_libxsmm + (size_t) (pad_h_out * ofwp * bk + pad_w_out * bk);
          DECL_VLA_PTR_PT_EXT(T,     output,   [Kb][ofhp][ofwp][bk],   t_O, (pad_h_out * ofwp * bk + pad_w_out * bk));
          //DType *input_libxsmm  = (DType*)libxsmm_aligned_malloc( N*ifhp*ifwp*C*sizeof(DType), 2097152);
          DECL_VLA_PTR_PT    (T,     inp,      [Cb][ifhp][ifwp][bc],   t_I);
          //DType *filter_libxsmm = (DType*)libxsmm_aligned_malloc( C*K*R*S*sizeof(DType), 2097152);
          DECL_VLA_PTR_PT    (T,     weight,   [CP][R][S][bc][bk],     t_W);

          if (cfg.avoid_fmas_in_rim == 0) {
            if (i_c == 0 && i_r == 0 && i_s == 0) {
              zero_tpp(output[i_n][i_k][i_h][i_w]);
            }
            brgemm_tpp(inp   [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                       weight[i_k][i_c][i_r][i_s][0],
                       output[i_n][i_k][i_h]                 [i_w],
                       Cb_step * r_step * s_step,
                       true);
          } else {
            if (i_c == 0 && i_r == 0 && i_s == 0) {
              //libxsmm_meltw_unary_param zero_param;
              //zero_param.out.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              //zero_kernel( &zero_param );
              zero_tpp(output[i_n][i_k][i_h][i_w]);
            }
#if 0
            unsigned long long brcount = Cb_step;
            libxsmm_gemm_param gemm_param;
            gemm_param.op.tertiary = (void*)&brcount;
            gemm_param.a.primary = LIBXSMM_ACCESS_RAW(6, sizeof(DType), filter_libxsmm, i_k, i_c, i_r, i_s, 0, 0, Cb, R, S, bc, bk);
            if (i_r == 0 && i_h == 0) {
              /* Do no FLOPS  */
            } else if (i_r == R-1 && i_h == ofh-1 ) {
              /* Do no FLOPS  */
            } else if ( i_w == 0 && i_s == 0 ) {
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s + 1, 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w + 1, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel2.gemm( &gemm_param );
            } else if ( i_w + w_step == ofw  && i_s == S-1) {
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel2.gemm( &gemm_param );
            } else {
              gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
              gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              brgemm_kernel.gemm( &gemm_param );
            }
#endif
            if (i_r == 0 && i_h == 0) {
              /* Do no FLOPS  */
            } else if (i_r == R-1 && i_h == ofh-1 ) {
              /* Do no FLOPS  */
            } else if ( i_w == 0 && i_s == 0 ) {
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s + 1, 0, Cb, ifhp, ifwp, bc);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w + 1, 0, Kb, ofhp, ofwp, bk);
              //brgemm_kernel2.gemm( &gemm_param );
              brgemm2_tpp(inp   [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s + 1],
                          weight[i_k][i_c][i_r][i_s][0],
                          output[i_n][i_k][i_h]                 [i_w + 1],
                          Cb_step,
                          true);
            } else if ( i_w + w_step == ofw  && i_s == S-1) {
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              //brgemm_kernel2.gemm( &gemm_param );
              brgemm2_tpp(inp   [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                          weight[i_k][i_c][i_r][i_s][0],
                          output[i_n][i_k][i_h]                 [i_w],
                          Cb_step,
                          true);
            } else {
              //gemm_param.b.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), input_libxsmm, i_n, i_c, i_h * stride_h + i_r, i_w * stride_w + i_s, 0, Cb, ifhp, ifwp, bc);
              //gemm_param.c.primary = LIBXSMM_ACCESS_RAW(5, sizeof(DType), output_libxsmm_off, i_n, i_k, i_h, i_w, 0, Kb, ofhp, ofwp, bk);
              //brgemm_kernel.gemm( &gemm_param );
              brgemm_tpp(inp   [i_n][i_c][i_h * stride_h + i_r][i_w * stride_w + i_s],
                         weight[i_k][i_c][i_r][i_s][0],
                         output[i_n][i_k][i_h]                 [i_w],
                         Cb_step,
                         true);
            }


          }
        },
        [&]() {if (sizeof(T) == 2) brgemm_tpp.config();},
        [&]() {if (sizeof(T) == 2) brgemm_tpp.release();});
//#endif
    } /* end of the scope with recorded parallel for */
  } /* end of the conv_fwd_scale scope */

} /* end of the dummy scope */

//auto t_dummy     = at::empty({0},  torch::TensorOptions().dtype(at::kFloat));
return std::vector<at::Tensor>({t_O});

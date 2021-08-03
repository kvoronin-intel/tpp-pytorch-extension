
RECORD_FUNCTION("gsage_mlp_bwd", std::vector<c10::IValue>());

at::Tensor t_in, t_in_res, t_wt, t_wt_res;
int i = 0;

auto t_grad_out = inputs[i++].contiguous();

if (res) {
  t_in = inputs[i++];
  t_in_res = inputs[i++];
  t_wt = inputs[i++];
  t_wt_res = inputs[i++];
} else {
  t_in = inputs[i++];
  t_wt = inputs[i++];
}
auto t_relu_mask = inputs[i++];
auto t_dp_mask = inputs[i++];

auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto N = in_sizes[0];
auto C = in_sizes[1];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto nk = wt_sizes[0];
auto nc = wt_sizes[1];
auto bc = wt_sizes[2];
if (t_wt.dtype() == at::kBFloat16)
  bc = bc * wt_sizes[4];
auto bk = wt_sizes[3];

auto bnp = bn;
auto bkp = bk;
auto bcp = bc;
auto remp = rem;

auto K = nk * bk;

if (t_in.dtype() == at::kBFloat16) {
  bnp = bn + bn % 2;
  remp = rem + rem % 2;
}

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
  bkp = bk + bk % 2;
}

int rd = (bk + 15) / 16;
DECL_VLA_PTR_PT(short, relu_mask, [bn][nk][rd], t_relu_mask);
DECL_VLA_PTR_PT(short, dp_mask, [bn][nk][rd], t_dp_mask);

const auto input_trans_flag =
    (t_in.dtype() == at::kFloat ? XformTPP::XFORM_XPOSE_TPP
                                : XformTPP::XFORM_NONE_TPP);

auto t_wt_TV = wt_tensor_for_bwd(nk, bk, nc, bc, t_wt);

at::Tensor t_wt_res_TV;
if (res)
  t_wt_res_TV = wt_tensor_for_bwd(nk, bk, nc, bc, t_wt_res);

auto t_grad_in = t_in.new_empty({N, C});
at::Tensor t_grad_in_res;
if (res) {
  t_grad_in_res = t_in_res.new_empty({N, C});
}

auto t_grad_wt = at::empty_like(t_wt);
at::Tensor t_grad_wt_tmp;
if (t_wt.dtype() == at::kBFloat16)
  t_grad_wt_tmp = at::empty({nk, nc, bc, bk});
else
  t_grad_wt_tmp = t_grad_wt;

at::Tensor t_grad_wt_res, t_grad_wt_res_tmp;
if (res) {
  t_grad_wt_res = at::empty_like(t_wt);
  if (t_wt.dtype() == at::kBFloat16)
    t_grad_wt_res_tmp = at::empty({nk, nc, bc, bk});
  else
    t_grad_wt_res_tmp = t_grad_wt_res;
}

auto t_grad_bias = t_wt.new_empty({nk * bk});

at::Tensor t_grad_out_f32 = t_grad_out;
if (t_grad_out.dtype() == at::kBFloat16)
  t_grad_out_f32 = at::empty({N, K});

DECL_VLA_PTR_PT(T, grad_out, [bn][nk][bk], t_grad_out);
DECL_VLA_PTR_PT(float, grad_out_f32, [bn][nk][bk], t_grad_out_f32);

DECL_VLA_PTR_PT(T, grad_in, [bn][nc][bc], t_grad_in);
DECL_VLA_PTR_PT(T, grad_in_res, [bn][nc][bc], t_grad_in_res);

// del-weights and weights in blocked layout
DECL_VLA_PTR_PT(T, grad_wt, [nc][bc * bk], t_grad_wt);
DECL_VLA_PTR_PT(T, grad_wt_res, [nc][bc * bk], t_grad_wt_res);
DECL_VLA_PTR_PT(float, grad_wt_tmp, [nc][bc * bk], t_grad_wt_tmp);
DECL_VLA_PTR_PT(float, grad_wt_res_tmp, [nc][bc * bk], t_grad_wt_res_tmp);

DECL_VLA_PTR_PT(T, wt_TV, [nc][bkp * bc], t_wt_TV);
DECL_VLA_PTR_PT(T, wt_res_TV, [nc][bkp * bc], t_wt_res_TV);
DECL_VLA_PTR_PT(T, grad_bias, [bk], t_grad_bias);

DECL_VLA_PTR_PT(T, in, [bn][nc][bc], t_in); // flat layout for fp32
DECL_VLA_PTR_PT(T, in_res, [bn][nc][bc], t_in_res); // flat layout for fp32

auto set_zero_tpp = SCOPEIT(SetZeroTPP<float>(nk * bk), EW_ZERO);
auto set_zero_col_tpp = SCOPEIT(SetZeroTPP<T>(bn, 1, bkp), EW_ZERO);
auto grad_bias_tpp = SCOPEIT(GradBiasTPP<float>(bn, bk, K), BIAS);
auto dropout_bwd_tpp =
    SCOPEIT((DropOutBwdTPP<T, float>(bn, bk, K, K, p)), DROPOUT);
auto relu_bwd_tpp = SCOPEIT(ReLUBwdTPP<float>(bn, bk, K, K), ACT);
auto n2v_tpp = SCOPEIT(
    XformExtTPP<T>(bn, bk, bnp, bk, nk* bk, bk, XformTPP::XFORM_N2V_TPP, true),
    VNNI);
auto n2v_wt_tpp = SCOPEIT(
    XformExtTPP<T>(bc, bk, bcp, bk, XformTPP::XFORM_N2V_TPP, true),
    VNNI);
auto cpy_tpp = SCOPEIT(CpyTPP<T>(bn, bk, bk, bkp), EW_COPY);
auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bn, bk, K, K)), EW_COPY);
auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(bn, bk, K, K)), EW_COPY);
auto add_gwt_tpp = SCOPEIT((AddTPP<float, float>(bc, bk)), EW_ADD);

auto brgemm_di_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(bn, bc, bkp, bkp, nc* bc* bkp, nk* bkp, bc, nc* bc, 0.0, 0, nk)));

auto brgemm_dw_f32_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
    bc,
    bk,
    bnp,
    nc* bc* bnp,
    nk* bk* bnp,
    nc* bc,
    nk* bk,
    bk,
    0.0,
    input_trans_flag,
    16)));
auto brgemm_dw_f32_tpp_b1 = SCOPEITGEMM((BrgemmTPP<T, float>(
    bc,
    bk,
    bnp,
    nc* bc* bnp,
    nk* bk* bnp,
    nc* bc,
    nk* bk,
    bk,
    1.0,
    input_trans_flag,
    16)));

// BF16 del-wt brgemms
auto brgemm_dw_bf16_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(bc, bk, bnp, bc* bnp, bk* bnp, bnp, bk, bk, 0.0, 0, 16)));
auto brgemm_dw_bf16_tpp_b1 = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(bc, bk, bnp, bc* bnp, bk* bnp, bnp, bk, bk, 1.0, 0, 16)));

{
  RECORD_SCOPE(gdbias, {t_grad_out});
  {
    tensor_set_zero(nk, bk, t_grad_bias);
    int threads = omp_get_max_threads();
    float* bias_ptrs[threads];
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        float prv_grad_bias[nk][bk];
        bias_ptrs[tid] = prv_grad_bias[0];
        set_zero_tpp(prv_grad_bias[0]);
#pragma omp for collapse(2)
        for (int n = 0; n < nn; n++) {
          for (int k = 0; k < nk; k++) {
            if (p > 0) {
              dropout_bwd_tpp(
                  grad_out[n][0][k], grad_out_f32[n][0][k], dp_mask[n][0][k]);
            } else
              cvt_f32_tpp(grad_out[n][0][k], grad_out_f32[n][0][k]);

            if (act == "relu") {
              relu_bwd_tpp(
                  grad_out_f32[n][0][k],
                  grad_out_f32[n][0][k],
                  relu_mask[n][0][k]);
            }
            grad_bias_tpp(grad_out_f32[n][0][k], prv_grad_bias[k]);
            if (p > 0 || act == "relu")
              cvt_tpp(grad_out_f32[n][0][k], grad_out[n][0][k]);
          }
        }
#pragma omp barrier
        omp_reduce_buf(threads, nk * bk, bias_ptrs, grad_bias[0]);
      }
      if (rem > 0) {
        DECL_VLA_PTR_PT(T, grad_out, [nk][bk], t_grad_out);
        DECL_VLA_PTR_PT(float, grad_out_f32, [nk][bk], t_grad_out_f32);
        DECL_VLA_PTR_PT(short, relu_mask, [nk][rd], t_relu_mask);
        DECL_VLA_PTR_PT(short, dp_mask, [nk][rd], t_dp_mask);

        auto dropout_bwd_tpp =
            SCOPEIT((DropOutBwdTPP<T, float>(rem, bk, K, K, p)), DROPOUT);
        auto relu_bwd_tpp = SCOPEIT(ReLUBwdTPP<float>(rem, bk, K, K), ACT);
        auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(rem, bk, K, K)), EW_COPY);
        auto cvt_f32_tpp =
            SCOPEIT((ConvertTPP<T, float>(rem, bk, K, K)), EW_COPY);
        auto grad_bias_tpp = SCOPEIT(GradBiasTPP<float>(rem, bk, K), BIAS);

        float prv_grad_bias[nk][bk];
        bias_ptrs[0] = prv_grad_bias[0];
        set_zero_tpp(prv_grad_bias[0]);

        for (int k = 0; k < nk; k++) {
          if (p > 0) {
            dropout_bwd_tpp(
                grad_out[nn * bn][k],
                grad_out_f32[nn * bn][k],
                dp_mask[nn * bn][k]);
          } else
            cvt_f32_tpp(grad_out[nn * bn][k], grad_out_f32[nn * bn][k]);
          if (act == "relu") {
            relu_bwd_tpp(
                grad_out_f32[nn * bn][k],
                grad_out_f32[nn * bn][k],
                relu_mask[nn * bn][k]);
          }
          grad_bias_tpp(grad_out_f32[nn * bn][k], prv_grad_bias[k]);
          if (p > 0 || act == "relu")
            cvt_tpp(grad_out_f32[nn * bn][k], grad_out[nn * bn][k]);
        }
        omp_reduce_buf(1, nk * bk, bias_ptrs, grad_bias[0], true);
      }
    }
  }
}

{
  RECORD_SCOPE(gdi_gemm, {t_grad_out, t_wt});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      T tmp[bn][nk * bkp];

#pragma omp for collapse(2)
      for (int n = 0; n < nn; n++) {
        for (int c = 0; c < nc; c++) {
          if (bk != bkp) {
            for (int k = 0; k < nk; k++) {
              set_zero_col_tpp(&tmp[0][k * bkp] + bk);
              cpy_tpp(grad_out[n][0][k], &tmp[0][k * bk]);
            }

            brgemm_di_tpp(tmp[0], wt_TV[0][c], grad_in[n][0][c], nk);
          } else
            brgemm_di_tpp(grad_out[n][0][0], wt_TV[0][c], grad_in[n][0][c], nk);
          if (res) {
            if (bk != bkp) {
              set_zero_col_tpp(tmp[0] + bk);
              cpy_tpp(grad_out[n][0][0], tmp[0]);

              brgemm_di_tpp(tmp[0], wt_res_TV[0][c], grad_in_res[n][0][c], nk);
            } else
              brgemm_di_tpp(
                  grad_out[n][0][0], wt_res_TV[0][c], grad_in_res[n][0][c], nk);
          }
        }
      }
    }
    if (rem > 0) {
      DECL_VLA_PTR_PT(T, grad_out, [nk][bk], t_grad_out);
      DECL_VLA_PTR_PT(T, grad_in, [nc][bc], t_grad_in);
      DECL_VLA_PTR_PT(T, grad_in_res, [nc][bc], t_grad_in_res);

      auto set_zero_col_tpp = SCOPEIT(SetZeroTPP<T>(rem, 1, bkp), EW_ZERO);
      auto cpy_tpp = SCOPEIT(CpyTPP<T>(rem, bk, bk, bkp), EW_COPY);
      auto brgemm_di_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
          rem,
          bc,
          bkp,
          bkp,
          nc * bc * bkp,
          nk * bkp,
          bc,
          nc * bc,
          0.0,
          0,
          nk)));

      T tmp[rem][nk * bkp];
      if (bk != bkp) {
        for (int k = 0; k < nk; k++) {
          set_zero_col_tpp(&tmp[0][k * bk] + bk);
          cpy_tpp(grad_out[nn * bn][0], &tmp[0][k * bk]);
        }
      }

      for (int c = 0; c < nc; c++) {
        if (bk != bkp)
          brgemm_di_tpp(tmp[0], wt_TV[0][c], grad_in[nn * bn][c], nk);
        else
          brgemm_di_tpp(
              grad_out[nn * bn][0], wt_TV[0][c], grad_in[nn * bn][c], nk);

        if (res) {
          if (bk != bkp)
            brgemm_di_tpp(tmp[0], wt_res_TV[0][c], grad_in_res[nn * bn][c], nk);
          else
            brgemm_di_tpp(
                grad_out[nn * bn][0],
                wt_res_TV[0][c],
                grad_in_res[nn * bn][c],
                nk);
        }
      }
    }
  }
}

int threads = omp_get_max_threads();
std::mutex lock[nk * nc];
auto setzero_delwt_tpp = SCOPEIT(SetZeroTPP<float>(nk * nc * bc * bk), EW_ZERO);
setzero_delwt_tpp(t_grad_wt_tmp.data_ptr<float>());
if (res)
  setzero_delwt_tpp(t_grad_wt_res_tmp.data_ptr<float>());

auto trans_tpp = SCOPEIT(
    XformExtTPP<
        T>(bn, bc, bc, bnp, nc* bc, bnp, XformTPP::XFORM_XPOSE_TPP, true),
    XPOSE);
{
  RECORD_SCOPE(gdw_gemm, {t_in, t_grad_out});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
    {
      constexpr int BS = 16;
      float tmp[bc * bk];
      int tid = omp_get_thread_num();

      int g_start = (nn * nk * nc) * tid / threads;
      int g_end = (nn * nk * nc) * (tid + 1) / threads;
      int s_ck = g_start / nn;
      int e_ck = (g_end - 1) / nn;
      int s_nn = g_start % nn;
      int e_nn = (g_end - 1) % nn;
      int start_nn, end_nn;
      for (int ck = s_ck; ck <= e_ck; ck++) {
        if (ck == s_ck)
          start_nn = s_nn;
        else
          start_nn = 0;
        if (ck == e_ck)
          end_nn = e_nn;
        else
          end_nn = nn - 1;

        int k = ck / nc;
        int c = ck % nc;

        if (t_in.dtype() == at::kBFloat16) {
          T tmp_go[BS * bnp * bk];
          T tmp_inT[BS * bnp * bc];
          for (int start_nn1 = start_nn; start_nn1 <= end_nn; start_nn1 += BS) {
            int count = start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1;
            n2v_tpp(
                count,
                nk * bk * bnp,
                bk * bnp,
                grad_out[start_nn1][0][k],
                tmp_go);
            trans_tpp(
                count, nc * bc * bnp, bc * bnp, in[start_nn1][0][c], tmp_inT);

            if (start_nn1 == start_nn)
              brgemm_dw_bf16_tpp(
                  tmp_inT,
                  tmp_go,
                  tmp,
                  (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
            else
              brgemm_dw_bf16_tpp_b1(
                  tmp_inT,
                  tmp_go,
                  tmp,
                  (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
          }
        } else if (t_in.dtype() == at::kFloat) {
          for (int start_nn1 = start_nn; start_nn1 <= end_nn; start_nn1 += BS) {
            if (start_nn1 == start_nn)
              brgemm_dw_f32_tpp(
                  in[start_nn1][0][c],
                  grad_out[start_nn1][0][k],
                  tmp,
                  (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
            else
              brgemm_dw_f32_tpp_b1(
                  in[start_nn1][0][c],
                  grad_out[start_nn1][0][k],
                  tmp,
                  (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
          }
        }
        lock[k * nc + c].lock();
        add_gwt_tpp(tmp, grad_wt_tmp[k][c], grad_wt_tmp[k][c]);
        lock[k * nc + c].unlock();
      }
    }
    if (rem > 0) {
      if (t_in.dtype() == at::kBFloat16) {
        DECL_VLA_PTR_PT(T, grad_out, [nk][bk], t_grad_out);
        DECL_VLA_PTR_PT(T, in, [nc][bc], t_in); // blocked layout for bf16
        auto n2v_tpp = SCOPEIT(
            XformExtTPP<T>(
                rem, bk, remp, bk, nk * bk, bk, XformTPP::XFORM_N2V_TPP, true),
            VNNI);
        auto trans_tpp = SCOPEIT(
            XformExtTPP<T>(
                rem,
                bc,
                bc,
                remp,
                nc * bc,
                remp,
                XformTPP::XFORM_XPOSE_TPP,
                true),
            XPOSE);
        auto brgemm_dw_bf16_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
            bc, bk, remp, bc * remp, bk * remp, remp, bk, bk, 0.0, 0, 1)));

#pragma omp parallel
        {
          T tmp_go[remp * bk];
          T tmp_inT[remp * bc];
          float tmp[bc * bk];

          int tid = omp_get_thread_num();

          int g_start = (nk * nc) * tid / threads;
          int g_end = (nk * nc) * (tid + 1) / threads;
          int s_ck = g_start;
          int e_ck = g_end - 1;

          for (int ck = s_ck; ck <= e_ck; ck++) {
            int k = ck / nc;
            int c = ck % nc;

            n2v_tpp(grad_out[nn * bn][k], tmp_go);
            trans_tpp(in[nn * bn][c], tmp_inT);
            brgemm_dw_bf16_tpp(tmp_inT, tmp_go, tmp, 1);

            lock[k * nc + c].lock();
            add_gwt_tpp(tmp, grad_wt_tmp[k][c], grad_wt_tmp[k][c]);
            lock[k * nc + c].unlock();
          }
        }
      } else if (t_in.dtype() == at::kFloat) {
        DECL_VLA_PTR_PT(T, grad_out, [nk][bk], t_grad_out);
        DECL_VLA_PTR_PT(T, in_rem, [nc][bc], t_in);
        auto brgemm_dw_f32_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
            bc,
            bk,
            remp,
            nc * bc * remp,
            nk * bk * remp,
            nc * bc,
            nk * bk,
            bk,
            0.0,
            input_trans_flag,
            1)));

#pragma omp parallel
        {
          float tmp[bc * bk];

          int tid = omp_get_thread_num();

          int g_start = (nk * nc) * tid / threads;
          int g_end = (nk * nc) * (tid + 1) / threads;
          int s_ck = g_start;
          int e_ck = g_end - 1;

          for (int ck = s_ck; ck <= e_ck; ck++) {
            int k = ck / nc;
            int c = ck % nc;

            brgemm_dw_f32_tpp(in_rem[nn * bn][c], grad_out[nn * bn][k], tmp, 1);

            lock[k * nc + c].lock();
            add_gwt_tpp(tmp, grad_wt_tmp[k][c], grad_wt_tmp[k][c]);
            lock[k * nc + c].unlock();
          }
        }
      }
    }
#pragma omp parallel for collapse(2)
    for (int k = 0; k < nk; k++) {
      for (int c = 0; c < nc; c++) {
        n2v_wt_tpp(grad_wt_tmp[k][c], grad_wt[k][c]);
      }
    }
    if (res) {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel
      {
        constexpr int BS = 16;
        float tmp[bc * bk];

        int tid = omp_get_thread_num();

        int g_start = (nn * nk * nc) * tid / threads;
        int g_end = (nn * nk * nc) * (tid + 1) / threads;
        int s_ck = g_start / nn;
        int e_ck = (g_end - 1) / nn;
        int s_nn = g_start % nn;
        int e_nn = (g_end - 1) % nn;
        int start_nn, end_nn;
        for (int ck = s_ck; ck <= e_ck; ck++) {
          if (ck == s_ck)
            start_nn = s_nn;
          else
            start_nn = 0;
          if (ck == e_ck)
            end_nn = e_nn;
          else
            end_nn = nn - 1;

          int k = ck / nc;
          int c = ck % nc;

          if (t_in_res.dtype() == at::kBFloat16) {
            T tmp_go[BS * bnp * bk];
            T tmp_inT[BS * bnp * bc];
            for (int start_nn1 = start_nn; start_nn1 <= end_nn;
                 start_nn1 += BS) {
              int count =
                  start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1;
              n2v_tpp(
                  count,
                  nk * bk * bnp,
                  bk * bnp,
                  grad_out[start_nn1][0][k],
                  tmp_go);
              trans_tpp(
                  count,
                  nc * bc * bnp,
                  bc * bnp,
                  in_res[start_nn1][0][c],
                  tmp_inT);

              if (start_nn1 == start_nn)
                brgemm_dw_bf16_tpp(
                    tmp_inT,
                    tmp_go,
                    tmp,
                    (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
              else
                brgemm_dw_bf16_tpp_b1(
                    tmp_inT,
                    tmp_go,
                    tmp,
                    (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
            }
          } else if (t_in_res.dtype() == at::kFloat) {
            for (int start_nn1 = start_nn; start_nn1 <= end_nn;
                 start_nn1 += BS) {
              if (start_nn1 == start_nn)
                brgemm_dw_f32_tpp(
                    in_res[start_nn1][0][c],
                    grad_out[start_nn1][0][k],
                    tmp,
                    (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
              else
                brgemm_dw_f32_tpp_b1(
                    in_res[start_nn1][0][c],
                    grad_out[start_nn1][0][k],
                    tmp,
                    (start_nn1 + BS <= end_nn ? BS : end_nn - start_nn1 + 1));
            }
          }
          lock[k * nc + c].lock();
          add_gwt_tpp(tmp, grad_wt_res_tmp[k][c], grad_wt_res_tmp[k][c]);
          lock[k * nc + c].unlock();
        }
      }
      if (rem > 0) {
        if (t_in.dtype() == at::kBFloat16) {
          DECL_VLA_PTR_PT(T, grad_out, [nk][bk], t_grad_out);
          DECL_VLA_PTR_PT(
              T, in_res, [nc][bc], t_in_res); // blocked layout for bf16
          auto n2v_tpp = SCOPEIT(
              XformExtTPP<T>(
                  rem,
                  bk,
                  remp,
                  bk,
                  nk * bk,
                  bk,
                  XformTPP::XFORM_N2V_TPP,
                  true),
              VNNI);
          auto trans_tpp = SCOPEIT(
              XformExtTPP<T>(
                  rem,
                  bc,
                  bc,
                  remp,
                  nc * bc,
                  remp,
                  XformTPP::XFORM_XPOSE_TPP,
                  true),
              XPOSE);
          auto brgemm_dw_bf16_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
              bc, bk, remp, bc * remp, bk * remp, remp, bk, bk, 0.0, 0, 1)));

#pragma omp parallel
          {
            T tmp_go[remp * bk];
            T tmp_inT[remp * bc];
            float tmp[bc * bk];

            int tid = omp_get_thread_num();

            int g_start = (nk * nc) * tid / threads;
            int g_end = (nk * nc) * (tid + 1) / threads;
            int s_ck = g_start;
            int e_ck = g_end - 1;

            for (int ck = s_ck; ck <= e_ck; ck++) {
              int k = ck / nc;
              int c = ck % nc;

              n2v_tpp(grad_out[nn * bn][k], tmp_go);
              trans_tpp(in_res[nn * bn][c], tmp_inT);
              brgemm_dw_bf16_tpp(tmp_inT, tmp_go, tmp, 1);

              lock[k * nc + c].lock();
              add_gwt_tpp(tmp, grad_wt_res_tmp[k][c], grad_wt_res_tmp[k][c]);
              lock[k * nc + c].unlock();
            }
          }
        } else if (t_in.dtype() == at::kFloat) {
          DECL_VLA_PTR_PT(T, grad_out, [nk][bk], t_grad_out);
          DECL_VLA_PTR_PT(T, in_rem, [nc][bc], t_in_res);
          auto brgemm_dw_f32_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
              bc,
              bk,
              remp,
              nc * bc * remp,
              nk * bk * remp,
              nc * bc,
              nk * bk,
              bk,
              0.0,
              input_trans_flag,
              1)));

#pragma omp parallel
          {
            float tmp[bc * bk];

            int tid = omp_get_thread_num();

            int g_start = (nk * nc) * tid / threads;
            int g_end = (nk * nc) * (tid + 1) / threads;
            int s_ck = g_start;
            int e_ck = g_end - 1;

            for (int ck = s_ck; ck <= e_ck; ck++) {
              int k = ck / nc;
              int c = ck % nc;
              brgemm_dw_f32_tpp(
                  in_rem[nn * bn][c], grad_out[nn * bn][k], tmp, 1);

              lock[k * nc + c].lock();
              add_gwt_tpp(tmp, grad_wt_res_tmp[k][c], grad_wt_res_tmp[k][c]);
              lock[k * nc + c].unlock();
            }
          }
        }
      }
#pragma omp parallel for collapse(2)
      for (int k = 0; k < nk; k++) {
        for (int c = 0; c < nc; c++) {
          n2v_wt_tpp(grad_wt_res_tmp[k][c], grad_wt_res[k][c]);
        }
      }
    }
  }
}

if (res) {
  return {t_grad_in, t_grad_in_res, t_grad_wt, t_grad_wt_res, t_grad_bias};
} else {
  return {t_grad_in, t_grad_wt, t_grad_bias};
}


RECORD_FUNCTION("rgcn_mlp_fwd", std::vector<c10::IValue>());

int i = 0;

auto t_degs = inputs[i++];
auto t_in = inputs[i++];
auto t_wt = inputs[i++];

auto in_sizes = t_in.sizes();
auto wt_sizes = t_wt.sizes();
auto N = in_sizes[0];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto nk = wt_sizes[0];
auto nc = wt_sizes[1];
auto bc = wt_sizes[2];
if (t_wt.dtype() == at::kBFloat16)
  bc = bc * wt_sizes[4];
auto bk = wt_sizes[3];
auto bcp = bc;
auto K = nk * bk;

auto t_out = t_in.new_empty({N, K});
at::Tensor t_out_f32;
if (t_out.dtype() == at::kFloat)
  t_out_f32 = t_out;
else
  t_out_f32 = at::empty({N, K});

auto t_norm = t_degs.new_empty({N});

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
}

auto t_wt_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt);

DECL_VLA_PTR_PT(T, in, [bn][nc][bcp], t_in);
DECL_VLA_PTR_PT(T, wt_V, [nc][bcp * bk], t_wt_V);
DECL_VLA_PTR_PT(float, degs, [bn], t_degs);
DECL_VLA_PTR_PT(float, norm, [bn], t_norm);
DECL_VLA_PTR_PT(float, out_f32, [bn][nk][bk], t_out_f32);
DECL_VLA_PTR_PT(T, out, [bn][nk][bk], t_out);

auto brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(bn, bk, bcp, bcp, bk* bcp, nc* bcp, bk, nk* bk, 0.0, 0, nc)));
auto recp_tpp = SCOPEIT(RecpTPP<float>(bn), EW_RECP);
auto recp_sqrt_tpp = SCOPEIT(RecpSqrtTPP<float>(bn), EW_RECP_SQRT);
auto mul_norm_tpp = SCOPEIT((MulNormTPP<float, T>(bn, bk, K, K)), EW_MUL);

{
  RECORD_SCOPE(rgo_mlp, {t_in, t_wt_V});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#if 0
#pragma omp parallel for
    for (int n = 0; n < nn; n++) {
      if (norm_type == "right")
        recp_tpp(degs[n], norm[n]);
      else if (norm_type == "both")
        recp_sqrt_tpp(degs[n], norm[n]);
    }
#endif
#pragma omp parallel 
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_num_threads();
      int work = nn * nk;
      int chunk =
        (work % threads == 0) ? (work / threads) : (work / threads) + 1;
      int chunk_start = (tid * chunk < work) ? (tid * chunk) : work;
      int chunk_end = ((tid + 1) * chunk < work) ? ((tid + 1) * chunk) : work;

      brgemm_tpp.config();
      for (int n3k = chunk_start; n3k < chunk_end; n3k++) {
        int n = n3k / nk;
        int k = n3k % nk;
#if 1
        if (norm_type == "right")
          recp_tpp(degs[n], norm[n]);
        else if (norm_type == "both")
          recp_sqrt_tpp(degs[n], norm[n]);
#endif
        brgemm_tpp(in[n][0][0], wt_V[k][0], out_f32[n][0][k], nc);
        mul_norm_tpp(norm[n], out_f32[n][0][k], out[n][0][k]);
      }
      brgemm_tpp.release();
    }
    if (rem > 0) {
      DECL_VLA_PTR_PT(T, in, [nc][bcp], t_in);
      DECL_VLA_PTR_PT(T, out, [nk][bk], t_out);
      DECL_VLA_PTR_PT(float, out_f32, [nk][bk], t_out_f32);
      DECL_VLA_PTR_PT(float, degs, [1], t_degs);
      DECL_VLA_PTR_PT(float, norm, [1], t_norm);

      auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
          rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));
      auto recp_tpp = SCOPEIT(RecpTPP<float>(rem), EW_RECP);
      auto recp_sqrt_tpp = SCOPEIT(RecpSqrtTPP<float>(rem), EW_RECP_SQRT);
      auto mul_norm_tpp =
          SCOPEIT((MulNormTPP<float, T>(rem, bk, K, K)), EW_MUL);

      if (norm_type == "right")
        recp_tpp(degs[nn * bn], norm[nn * bn]);
      else if (norm_type == "both")
        recp_sqrt_tpp(degs[nn * bn], norm[nn * bn]);

      brgemm_tpp.config();
      
      for (int k = 0; k < nk; k++) {
        brgemm_tpp(in[nn * bn][0], wt_V[k][0], out_f32[nn * bn][k], nc);
        mul_norm_tpp(norm[nn * bn], out_f32[nn * bn][k], out[nn * bn][k]);
      }
      
      brgemm_tpp.release();
    }
  }
}

return {t_out, t_norm};

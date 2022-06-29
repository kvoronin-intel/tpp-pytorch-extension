
RECORD_FUNCTION("rgcn_gemm_fwd", std::vector<c10::IValue>());

int i = 0;

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

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
}

auto t_wt_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt);
auto t_out = t_in.new_empty({N, K});

DECL_VLA_PTR_PT(T, in, [bn][nc][bcp], t_in);
DECL_VLA_PTR_PT(T, out, [bn][nk][bk], t_out);
DECL_VLA_PTR_PT(T, wt_V, [nc][bcp * bk], t_wt_V);

auto brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        T>(bn, bk, bcp, bcp, bk* bcp, nc* bcp, bk, nk* bk, 0.0, 0, nc)));
{
  RECORD_SCOPE(rgo_gemm, {t_in, t_wt_V});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
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
        brgemm_tpp(in[n][0][0], wt_V[k][0], out[n][0][k], nc);
      }

      brgemm_tpp.release();
    }
    if (rem > 0) {
      DECL_VLA_PTR_PT(T, in, [nc][bcp], t_in);
      DECL_VLA_PTR_PT(T, out, [nk][bk], t_out);

      auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, T>(
          rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));

      brgemm_tpp.config();

      for (int k = 0; k < nk; k++) {
        brgemm_tpp(in[nn * bn][0], wt_V[k][0], out[nn * bn][k], nc);
      }

      brgemm_tpp.release();
    }
  }
}

return t_out;

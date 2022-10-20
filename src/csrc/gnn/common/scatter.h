{
  at::Tensor t_in = inputs[0];
  at::Tensor t_idx = inputs[1];
  at::Tensor t_out = inputs[2];

  auto E = t_in.size(1);
  auto N = t_idx.size(0);
  auto nn = N / alignN;
  auto bn = alignN;
  auto rem = N % alignN;

  DECL_VLA_PTR_PT(T, in, [bn][E], t_in);
  DECL_VLA_PTR_PT(T, out, [E], t_out);
  DECL_VLA_PTR_PT(int64_t, idx, [bn], t_idx);

  auto scatter_tpp = SCOPEIT((ScatterTPP<T, int64_t, T>(bn, E, E, E)), ROW_ST);

  {
    RECORD_SCOPE(scatter, {t_out, t_idx});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++) {
        scatter_tpp(in[n][0], idx[n], out[0]);
      }
    }
    if (rem > 0) {
      DECL_VLA_PTR_PT(int64_t, idx, [1], t_idx);
      DECL_VLA_PTR_PT(T, in, [E], t_in);
      auto scatter_tpp =
          SCOPEIT((ScatterTPP<T, int64_t, T>(rem, E, E, E)), ROW_ST);
      scatter_tpp(in[nn * bn], idx[nn * bn], out[0]);
    }
  }
}

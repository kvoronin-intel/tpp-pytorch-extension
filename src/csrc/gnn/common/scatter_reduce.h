{
  at::Tensor t_in = inputs[0];
  at::Tensor t_idx = inputs[1];
  at::Tensor t_out = inputs[2];

  auto E = t_in.size(1);
  auto N = t_idx.size(0);
  auto nn = N / alignN;
  auto bn = alignN;
  auto rem = N % alignN;

  auto t_temp = t_out.new_empty({alignN, E});
  DECL_VLA_PTR_PT(T, in, [bn][E], t_in);
  DECL_VLA_PTR_PT(T, out, [E], t_out);
  DECL_VLA_PTR_PT(T, temp, [E], t_temp);
  DECL_VLA_PTR_PT(long, idx, [bn], t_idx);

  auto scatter_tpp = SCOPEIT((ScatterTPP<T, long, T>(bn, E, E, E)), ROW_ST);
  auto gather_tpp = SCOPEIT((EmbeddingFwdTPP<T, long, T>(bn, E, E, E)), ROW_GT);
  auto add_tpp = SCOPEIT((AddTPP<T, T>(bn, E)), EW_ADD);
  {
    RECORD_SCOPE(scatter, {t_out, t_idx});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++) {
        gather_tpp(out[0], idx[n], temp[0]);
        add_tpp(in[n][0], temp[0], temp[0]);
        scatter_tpp(temp[0], idx[n], out[0]);
        long i = idx[n][0];
      }
    }
    if (rem > 0) {
      auto t_temp_rem = t_out.new_empty({rem, E});
      DECL_VLA_PTR_PT(T, tempr, [E], t_temp_rem);
      DECL_VLA_PTR_PT(long, idx, [1], t_idx);
      DECL_VLA_PTR_PT(T, in, [E], t_in);
      auto scatter_tpp =
          SCOPEIT((ScatterTPP<T, long, T>(rem, E, E, E)), ROW_ST);
      auto gather_tpp =
          SCOPEIT((EmbeddingFwdTPP<T, long, T>(rem, E, E, E)), ROW_GT);
      auto add_tpp = SCOPEIT((AddTPP<T, T>(rem, E)), EW_ADD);
      gather_tpp(out[0], idx[nn * bn], tempr[0]);
      add_tpp(in[nn * bn], tempr[0], tempr[0]);
      scatter_tpp(tempr[0], idx[nn * bn], out[0]);
    }
  }
}

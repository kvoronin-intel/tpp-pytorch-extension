{
  at::Tensor t_in = inputs[0];
  at::Tensor t_idx = inputs[1];

  auto E = t_in.size(1);
  auto N = t_idx.size(0);
  auto nn = N / alignN;
  auto bn = alignN;
  auto rem = N % alignN;

  auto t_out = t_in.new_empty({N, E});

  DECL_VLA_PTR_PT(T, in, [E], t_in);
  DECL_VLA_PTR_PT(T, out, [bn][E], t_out);
  DECL_VLA_PTR_PT(int64_t, idx, [bn], t_idx);

  auto gather_tpp =
      SCOPEIT((EmbeddingFwdTPP<T, int64_t, T>(bn, E, E, E)), ROW_GT);

  {
    RECORD_SCOPE(gather, {t_in, t_idx});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++) {
        gather_tpp(in[0], idx[n], out[n][0]);
      }
    }
    if (rem > 0) {
      DECL_VLA_PTR_PT(int64_t, idx, [1], t_idx);
      DECL_VLA_PTR_PT(T, out, [E], t_out);
      auto gather_tpp =
          SCOPEIT((EmbeddingFwdTPP<T, int64_t, T>(rem, E, E, E)), ROW_GT);
      gather_tpp(in[0], idx[nn * bn], out[nn * bn]);
    }
  }

  return t_out;
}

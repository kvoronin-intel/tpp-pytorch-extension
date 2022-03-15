
RECORD_FUNCTION("dropout_fwd", std::vector<c10::IValue>());

int i = 0;
auto t_in = inputs[i++];
auto t_out = t_in;

at::Tensor t_dp_mask;

auto in_sizes = t_in.sizes();
auto N = in_sizes[0];
auto C = in_sizes[1];

auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

if (training && p > 0.0) {
  t_out = t_in.new_empty({N, C});
  int rd = (C + 15) / 16;
  t_dp_mask = at::empty({N, rd}, at::kShort);

  DECL_VLA_PTR_PT(T, in, [bn][C], t_in);
  DECL_VLA_PTR_PT(T, out, [bn][C], t_out);
  DECL_VLA_PTR_PT(short, dp_mask, [bn][rd], t_dp_mask);
  auto dropout_fwd_tpp =
      SCOPEIT((DropOutFwdTPP<T, T>(bn, C, C, C, p)), DROPOUT);
  {
    RECORD_SCOPE(rgo_dropout, {t_in});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++)
        dropout_fwd_tpp(
            in[n][0], (void*)get_rng_state(), out[n][0], dp_mask[n][0]);
      if (rem > 0) {
        DECL_VLA_PTR_PT(T, in, [C], t_in);
        DECL_VLA_PTR_PT(T, out, [C], t_out);
        DECL_VLA_PTR_PT(short, dp_mask, [rd], t_dp_mask);
        auto dropout_fwd_tpp =
            SCOPEIT((DropOutFwdTPP<T, T>(1, C, C, C, p)), DROPOUT);
        for (int r = 0; r < rem; r++)
          dropout_fwd_tpp(
              in[nn * bn + r],
              (void*)get_rng_state(),
              out[nn * bn + r],
              dp_mask[nn * bn + r]);
      }
    }
  }
}
return {t_out, t_dp_mask};

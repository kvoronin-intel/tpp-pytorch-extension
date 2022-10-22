
RECORD_FUNCTION("dropout_bwd", std::vector<c10::IValue>());

int i = 0;
auto t_grad_out = inputs[i++];
auto t_dp_mask = inputs[i++];

at::Tensor t_grad_in = t_grad_out;

auto in_sizes = t_grad_out.sizes();
auto N = in_sizes[0];
auto K = in_sizes[1];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

if (p > 0.0) {
  t_grad_in = t_grad_out.new_empty({N, K});
  int rd = (K + 15) / 16;
  auto grad_out = GetVLAPtr<T>(t_grad_out, {bn, K});
  auto grad_in = GetVLAPtr<T>(t_grad_in, {bn, K});
  auto dp_mask = GetVLAPtr<short>(t_dp_mask, {bn, rd});

  auto dropout_bwd_tpp =
      SCOPEIT((DropOutBwdTPP<T, T>(bn, K, K, K, p)), DROPOUT);
  {
    RECORD_SCOPE(rgdo_dropout, {t_grad_out});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++)
        dropout_bwd_tpp(grad_out[n][0], grad_in[n][0], dp_mask[n][0]);
      if (rem > 0) {
        auto grad_out = GetVLAPtr<T>(t_grad_out, {K});
        auto grad_in = GetVLAPtr<T>(t_grad_in, {K});
        auto dp_mask = GetVLAPtr<short>(t_dp_mask, {rd});

        auto dropout_bwd_tpp =
            SCOPEIT((DropOutBwdTPP<T, T>(1, K, K, K, p)), DROPOUT);

        for (int r = 0; r < rem; r++) {
          dropout_bwd_tpp(
              grad_out[nn * bn + r],
              grad_in[nn * bn + r],
              dp_mask[nn * bn + r]);
        }
      }
    }
  }
}
return t_grad_in;

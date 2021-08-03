
RECORD_FUNCTION("gsage_mlp_fwd", std::vector<c10::IValue>());

at::Tensor t_in, t_in_res, t_wt, t_wt_res, t_bias;
int i = 0;

if (res) {
  t_in = inputs[i++];
  t_in_res = inputs[i++];
  t_wt = inputs[i++];
  t_wt_res = inputs[i++];
} else {
  t_in = inputs[i++];
  t_wt = inputs[i++];
}
t_bias = inputs[i++];

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

at::Tensor t_wt_res_V;

if (res) {
  t_wt_res_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt_res);
}

auto t_out = t_in.new_empty({N, K});
auto t_out_res = at::empty({N, K});
at::Tensor t_out_f32;
if (t_out.dtype() == at::kFloat)
  t_out_f32 = t_out;
else
  t_out_f32 = at::empty({N, K});

int rd = (bk + 15) / 16;

at::Tensor t_relu_mask = at::empty({N, nk* rd}, at::kShort);
at::Tensor t_dp_mask = at::empty({N, nk* rd}, at::kShort);

DECL_VLA_PTR_PT(T, in, [bn][nc][bcp], t_in);
DECL_VLA_PTR_PT(T, in_res, [bn][nc][bcp], t_in_res);
DECL_VLA_PTR_PT(T, wt_V, [nc][bcp * bk], t_wt_V);
DECL_VLA_PTR_PT(T, wt_res_V, [nc][bcp * bk], t_wt_res_V);
DECL_VLA_PTR_PT(T, bias, [bk], t_bias);
DECL_VLA_PTR_PT(T, out, [bn][nk][bk], t_out);
DECL_VLA_PTR_PT(float, out_f32, [bn][nk][bk], t_out_f32);
DECL_VLA_PTR_PT(float, out_res, [bn][nk][bk], t_out_res);
DECL_VLA_PTR_PT(short, relu_mask, [bn][nk][rd], t_relu_mask);
DECL_VLA_PTR_PT(short, dp_mask, [bn][nk][rd], t_dp_mask);

auto brgemm_tpp = SCOPEITGEMM(
    (BrgemmTPP<
        T,
        float>(bn, bk, bcp, bcp, bk* bcp, nc* bcp, bk, nk* bk, 0.0, 0, nc)));
auto add_bias_tpp = SCOPEIT(AddBiasTPP<T>(bn, bk, K), BIAS);
auto add_tpp = SCOPEIT((AddTPP<float, float>(bn, bk, nk* bk, nk* bk)), EW_ADD);
auto relu_fwd_tpp = SCOPEIT(ReLUFwdTPP<float>(bn, bk, nk* bk, nk* bk), ACT);
auto dropout_fwd_tpp =
    SCOPEIT((DropOutFwdTPP<float, T>(bn, bk, nk* bk, nk* bk, p)), DROPOUT);
auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bn, bk, K, K)), EW_COPY);

{
  RECORD_SCOPE(go_gemm, {t_in, t_wt_V});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int n = 0; n < nn; n++) {
      for (int k = 0; k < nk; k++) {
        brgemm_tpp(in[n][0][0], wt_V[k][0], out_f32[n][0][k], nc);
        if (res) {
          brgemm_tpp(in_res[n][0][0], wt_res_V[k][0], out_res[n][0][k], nc);
        }
        add_bias_tpp(bias[k], out_f32[n][0][k]);
        add_tpp(out_f32[n][0][k], out_res[n][0][k], out_f32[n][0][k]);
        if (act == "relu") {
          relu_fwd_tpp(out_f32[n][0][k], out_f32[n][0][k], relu_mask[n][0][k]);
        }
        if (p > 0 && training) {
          dropout_fwd_tpp(
              out_f32[n][0][k],
              (void*)rng_state,
              out[n][0][k],
              dp_mask[n][0][k]);
        } else
          cvt_tpp(out_f32[n][0][k], out[n][0][k]);
      }
    }
    if (rem > 0) {
      DECL_VLA_PTR_PT(T, in, [nc][bcp], t_in);
      DECL_VLA_PTR_PT(T, in_res, [nc][bcp], t_in_res);
      DECL_VLA_PTR_PT(T, out, [nk][bk], t_out);
      DECL_VLA_PTR_PT(float, out_f32, [nk][bk], t_out_f32);
      DECL_VLA_PTR_PT(float, out_res, [nk][bk], t_out_res);
      DECL_VLA_PTR_PT(short, relu_mask, [nk][rd], t_relu_mask);
      DECL_VLA_PTR_PT(short, dp_mask, [nk][rd], t_dp_mask);

      auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
          rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));
      auto add_bias_tpp = SCOPEIT(AddBiasTPP<T>(rem, bk, K), BIAS);
      auto add_tpp =
          SCOPEIT((AddTPP<float, float>(rem, bk, nk * bk, nk * bk)), EW_ADD);
      auto relu_fwd_tpp =
          SCOPEIT(ReLUFwdTPP<float>(rem, bk, nk * bk, nk * bk), ACT);
      auto dropout_fwd_tpp = SCOPEIT(
          (DropOutFwdTPP<float, T>(rem, bk, nk * bk, nk * bk, p)), DROPOUT);
      auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(rem, bk, K, K)), EW_COPY);

#pragma omp parallel for
      for (int k = 0; k < nk; k++) {
        brgemm_tpp(in[nn * bn][0], wt_V[k][0], out_f32[nn * bn][k], nc);
        if (res) {
          brgemm_tpp(
              in_res[nn * bn][0], wt_res_V[k][0], out_res[nn * bn][k], nc);
        }
        add_bias_tpp(bias[k], out_f32[nn * bn][k]);
        add_tpp(out_f32[nn * bn][k], out_res[nn * bn][k], out_f32[nn * bn][k]);
        if (act == "relu") {
          relu_fwd_tpp(
              out_f32[nn * bn][k], out_f32[nn * bn][k], relu_mask[nn * bn][k]);
        }
        if (p > 0 && training) {
          dropout_fwd_tpp(
              out_f32[nn * bn][k],
              (void*)rng_state,
              out[nn * bn][k],
              dp_mask[nn * bn][k]);
        } else
          cvt_tpp(out_f32[nn * bn][k], out[nn * bn][k]);
      }
    }
  }
}

return {t_out, t_relu_mask, t_dp_mask};


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
auto nn = in_sizes[0];
auto nc = in_sizes[1];
auto bn = in_sizes[2];
auto bc = in_sizes[3];
auto bcp = bc;

auto nk = wt_sizes[0];
auto bk = wt_sizes[3];

if (t_wt.dtype() == at::kBFloat16) {
  bcp = bc + bc % 2;
}

auto t_wt_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt);

at::Tensor t_wt_res_V;

if (res) {
  t_wt_res_V = wt_tensor_for_fwd(nk, bk, nc, bc, t_wt_res);
}

auto t_out = t_in.new_empty({nn, nk, bn, bk});
auto t_out_res = at::empty({nn, nk, bn, bk});
at::Tensor t_out_f32;
if (t_out.dtype() == at::kFloat)
  t_out_f32 = t_out;
else
  t_out_f32 = at::empty({nn, nk, bn, bk});

int rd = (bn * bk + 15) / 16;

at::Tensor t_relu_mask = at::empty({nn, nk, rd}, at::kShort);
at::Tensor t_dp_mask = at::empty({nn, nk, rd}, at::kShort);

DECL_VLA_PTR_PT(T, in, [nc][bn * bcp], t_in);
DECL_VLA_PTR_PT(T, wt_V, [nc][bk * bcp], t_wt_V);
DECL_VLA_PTR_PT(T, wt_res_V, [nc][bk * bcp], t_wt_res_V);
DECL_VLA_PTR_PT(T, bias, [bk], t_bias);
DECL_VLA_PTR_PT(T, in_res, [nc][bn * bcp], t_in_res);
DECL_VLA_PTR_PT(T, out, [nk][bn * bk], t_out);
DECL_VLA_PTR_PT(float, out_f32, [nk][bn * bk], t_out);
DECL_VLA_PTR_PT(float, out_res, [nk][bn * bk], t_out_res);

DECL_VLA_PTR_PT(short, relu_mask, [nk][rd], t_relu_mask);
DECL_VLA_PTR_PT(short, dp_mask, [nk][rd], t_dp_mask);

auto brgemm_tpp = SCOPEITGEMM2((BrgemmTPP<T, float>(
    bn,
    bk,
    bcp,
    bn* bcp,
    bk* bcp,
    bn,
    C,
    bn,
    0.0,
    XformTPP::XFORM_NONE_TPP,
    0,
    nc)));
auto add_bias_tpp = SCOPEIT(AddBiasTPP<T>(bn, bk), BIAS);
auto add_tpp = SCOPEIT((AddTPP<float, float>(bn, bk)), EW_ADD);
auto relu_fwd_tpp = SCOPEIT(ReLUFwdTPP<float>(bn * bk), ACT);
auto dropout_fwd_tpp = SCOPEIT((DropOutFwdTPP<float, T>(bn * bk, p)), DROPOUT);

{
  RECORD_SCOPE(go_gemm, {t_in, t_wt_V});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
    for (int n = 0; n < nn; n++) {
      for (int k = 0; k < nk; k++) {
        brgemm_tpp(in[n][0], wt_V[k][0], out_f32[n][k], nc);
        if (res) {
          brgemm_tpp(in_res[n][0], wt_res_V[k][0], out_res[n][k], nc);
        }
        add_bias_tpp(bias[k], out_f32[n][k]);
        add_tpp(out_f32[n][k], out_res[n][k], out_f32[n][k]);
        if (act == "relu") {
          relu_fwd_tpp(out_f32[n][k], out_f32[n][k], relu_mask[n][k]);
        }
        if (p > 0 && training) {
          dropout_fwd_tpp(
              out_f32[n][k], (void*)rng_state, out[n][k], dp_mask[n][k]);
        }
      }
    }
  }
}
return {t_out, t_relu_mask, t_dp_mask};

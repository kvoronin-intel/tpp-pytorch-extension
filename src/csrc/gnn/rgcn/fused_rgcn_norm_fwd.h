
RECORD_FUNCTION("rgcn_norm_fwd", std::vector<c10::IValue>());

int i = 0;

auto t_degs = inputs[i++];
auto t_in = inputs[i++];

auto in_sizes = t_in.sizes();
auto N = in_sizes[0];
auto C = in_sizes[1];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

auto t_out = t_in.new_empty({N, C});

at::Tensor t_in_f32;
if (t_in.dtype() == at::kFloat)
  t_in_f32 = t_in;
else
  t_in_f32 = at::empty({N, C});

auto t_norm = t_degs.new_empty({N});

DECL_VLA_PTR_PT(T, in, [bn][C], t_in);
DECL_VLA_PTR_PT(T, out, [bn][C], t_out);
DECL_VLA_PTR_PT(float, degs, [bn], t_degs);
DECL_VLA_PTR_PT(float, norm, [bn], t_norm);
DECL_VLA_PTR_PT(float, in_f32, [bn][C], t_in_f32);

auto recp_tpp = SCOPEIT((RecpTPP<float, float>(bn)), EW_RCP);
auto recp_sqrt_tpp = SCOPEIT((RecpSqrtTPP<float, float>(bn)), EW_RSQRT);
auto mul_norm_tpp = SCOPEIT((MulNormTPP<float, T>(bn, C, C, C)), EW_MUL);
auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(bn, C)), EW_COPY);

{
  RECORD_SCOPE(rgo_norm, {t_out});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < nn; n++) {
      if (norm_type == "left" || norm_type == "right")
        recp_tpp(degs[n], norm[n]);
      else if (norm_type == "both")
        recp_sqrt_tpp(degs[n], norm[n]);
      cvt_f32_tpp(in[n][0], in_f32[n][0]);
      mul_norm_tpp(norm[n], in_f32[n][0], out[n][0]);
    }
    /**/
    if (rem > 0) {
      DECL_VLA_PTR_PT(float, degs, [1], t_degs);
      DECL_VLA_PTR_PT(float, norm, [1], t_norm);
      DECL_VLA_PTR_PT(T, in, [C], t_in);
      DECL_VLA_PTR_PT(T, out, [C], t_out);
      DECL_VLA_PTR_PT(float, in_f32, [C], t_in_f32);

      auto recp_tpp = SCOPEIT((RecpTPP<float, float>(rem)), EW_RCP);
      auto recp_sqrt_tpp = SCOPEIT((RecpSqrtTPP<float, float>(rem)), EW_RSQRT);
      auto mul_norm_tpp =
          SCOPEIT_REF((MulNormTPP<float, T>(rem, C, C, C)), EW_MUL);
      auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(rem, C)), EW_COPY);

      if (norm_type == "left" || norm_type == "right")
        recp_tpp(degs[nn * bn], norm[nn * bn]);
      else if (norm_type == "both")
        recp_sqrt_tpp(degs[nn * bn], norm[nn * bn]);
      cvt_f32_tpp(in[nn * bn], in_f32[nn * bn]);
      mul_norm_tpp(norm[nn * bn], in_f32[nn * bn], out[nn * bn]);
    }
    /**/
  }
}

return {t_out, t_norm};

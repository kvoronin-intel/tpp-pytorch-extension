
RECORD_FUNCTION("rgcn_norm_bwd", std::vector<c10::IValue>());

int i = 0;

auto t_grad_out = inputs[i++].contiguous();
auto t_norm = inputs[i++];

auto grad_out_sizes = t_grad_out.sizes();
auto N = grad_out_sizes[0];
auto K = grad_out_sizes[1];
auto bn = align;
auto nn = N / bn;
auto rem = N % bn;

at::Tensor t_grad_out_f32 = t_grad_out;
if (t_grad_out.dtype() == at::kBFloat16)
  t_grad_out_f32 = at::empty({N, K});

auto t_grad_in = t_grad_out.new_empty({N, K});

DECL_VLA_PTR_PT(T, grad_out, [bn][K], t_grad_out);
DECL_VLA_PTR_PT(T, grad_in, [bn][K], t_grad_in);
DECL_VLA_PTR_PT(float, grad_out_f32, [bn][K], t_grad_out_f32);
DECL_VLA_PTR_PT(float, norm, [bn], t_norm);

auto mul_norm_tpp = SCOPEIT((MulNormTPP<float, T>(bn, K, K, K)), EW_MUL);
auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(bn, K)), EW_COPY);

{
  RECORD_SCOPE(rgdnorm, {t_grad_out, t_grad_in});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < nn; n++) {
      cvt_f32_tpp(grad_out[n][0], grad_out_f32[n][0]);
      mul_norm_tpp(norm[n], grad_out_f32[n][0], grad_in[n][0]);
    }
    if (rem > 0) {
      DECL_VLA_PTR_PT(T, grad_out, [K], t_grad_out);
      DECL_VLA_PTR_PT(T, grad_in, [K], t_grad_in);
      DECL_VLA_PTR_PT(float, grad_out_f32, [K], t_grad_out_f32);
      DECL_VLA_PTR_PT(float, norm, [1], t_norm);

      auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(rem, K)), EW_COPY);
      auto mul_norm_tpp = SCOPEIT((MulNormTPP<float, T>(rem, K, K, K)), EW_MUL);

      cvt_f32_tpp(grad_out[nn * bn], grad_out_f32[nn * bn]);
      mul_norm_tpp(norm[nn * bn], grad_out_f32[nn * bn], grad_in[nn * bn]);
    }
  }
}

return t_grad_in;

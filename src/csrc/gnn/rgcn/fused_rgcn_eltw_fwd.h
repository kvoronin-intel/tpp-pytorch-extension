
#define gettid() ((int)syscall(SYS_gettid))

RECORD_FUNCTION("rgcn_eltw_fwd", std::vector<c10::IValue>());

int i = 0;

if (self_loop) {
  auto t_in = inputs[i++];
  auto t_in_dst = inputs[i++];
  auto t_wt = inputs[i++];
  at::Tensor t_bias = at::empty(0);
  if (apply_bias)
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

  auto t_out = t_in.new_empty({N, K});
  at::Tensor t_out_f32;
  if (t_out.dtype() == at::kFloat)
    t_out_f32 = t_out;
  else
    t_out_f32 = at::empty({N, K});

  at::Tensor t_in_f32;
  if (t_in.dtype() == at::kFloat)
    t_in_f32 = t_in;
  else
    t_in_f32 = at::empty({N, K});

  int rd = (bk + 15) / 16;

  at::Tensor t_relu_mask = at::empty({N, nk * rd}, at::kShort);
  at::Tensor t_dp_mask = at::empty({N, nk * rd}, at::kShort);

  DECL_VLA_PTR_PT(T, in, [bn][nk][bk], t_in);
  DECL_VLA_PTR_PT(T, in_dst, [bn][nc][bcp], t_in_dst);
  DECL_VLA_PTR_PT(T, wt_V, [nc][bcp * bk], t_wt_V);
  DECL_VLA_PTR_PT(float, bias, [bk], t_bias);
  DECL_VLA_PTR_PT(T, out, [bn][nk][bk], t_out);
  DECL_VLA_PTR_PT(float, out_f32, [bn][nk][bk], t_out_f32);
  DECL_VLA_PTR_PT(float, in_f32, [bn][nk][bk], t_in_f32);
  DECL_VLA_PTR_PT(short, relu_mask, [bn][nk][rd], t_relu_mask);
  DECL_VLA_PTR_PT(short, dp_mask, [bn][nk][rd], t_dp_mask);

  auto relu_fwd_tpp =
      SCOPEIT(ReLUFwdTPP<float>(bn, bk, nk * bk, nk * bk, true), ACT);
  auto dropout_fwd_tpp =
      SCOPEIT((DropOutFwdTPP<float, T>(bn, bk, nk * bk, nk * bk, p)), DROPOUT);
  auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bn, bk, K, K)), EW_COPY);
  auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(bn, bk, K, K)), EW_COPY);
  auto add_tpp = SCOPEIT((AddTPP<float, float>(bn, bk, K, K)), EW_ADD);

  if (apply_bias) {
    auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
        bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));
    auto cpy_bias_tpp = SCOPEIT(CpyBiasTPP<float>(bn, bk, K), BIAS);

    RECORD_SCOPE(rgewo_gemm, {t_in, t_wt_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < nn; n++) {
        for (int k = 0; k < nk; k++) {
          cpy_bias_tpp(bias[k], out_f32[n][0][k]);
          brgemm_tpp(in_dst[n][0][0], wt_V[k][0], out_f32[n][0][k], nc);
          cvt_f32_tpp(in[n][0][k], in_f32[n][0][k]);
          add_tpp(in_f32[n][0][k], out_f32[n][0][k], out_f32[n][0][k]);
          if (act == "relu") {
            relu_fwd_tpp(
                out_f32[n][0][k], out_f32[n][0][k], relu_mask[n][0][k]);
          }
          if (p > 0 && training) {
            dropout_fwd_tpp(
                out_f32[n][0][k],
                (void*)get_rng_state(),
                out[n][0][k],
                dp_mask[n][0][k]);
          } else
            cvt_tpp(out_f32[n][0][k], out[n][0][k]);
        }
      }
      if (rem > 0) {
        DECL_VLA_PTR_PT(T, in, [nk][bk], t_in);
        DECL_VLA_PTR_PT(T, in_dst, [nc][bcp], t_in_dst);
        DECL_VLA_PTR_PT(T, out, [nk][bk], t_out);
        DECL_VLA_PTR_PT(float, out_f32, [nk][bk], t_out_f32);
        DECL_VLA_PTR_PT(float, in_f32, [nk][bk], t_in_f32);
        DECL_VLA_PTR_PT(short, relu_mask, [nk][rd], t_relu_mask);
        DECL_VLA_PTR_PT(short, dp_mask, [nk][rd], t_dp_mask);

        auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
            rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 1.0, 0, nc)));
        auto cpy_bias_tpp = SCOPEIT(CpyBiasTPP<float>(1, bk, K), BIAS);
        auto relu_fwd_tpp =
            SCOPEIT(ReLUFwdTPP<float>(1, bk, nk * bk, nk * bk, true), ACT);
        auto dropout_fwd_tpp = SCOPEIT(
            (DropOutFwdTPP<float, T>(1, bk, nk * bk, nk * bk, p)), DROPOUT);
        auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(1, bk, K, K)), EW_COPY);
        auto cvt_f32_tpp =
            SCOPEIT((ConvertTPP<T, float>(1, bk, K, K)), EW_COPY);
        auto add_tpp = SCOPEIT((AddTPP<float, float>(1, bk, K, K)), EW_ADD);

        for (int k = 0; k < nk; k++) {
          for (int r = 0; r < rem; r++)
            cpy_bias_tpp(bias[k], out_f32[nn * bn + r][k]);
          brgemm_tpp(in_dst[nn * bn][0], wt_V[k][0], out_f32[nn * bn][k], nc);

          for (int r = 0; r < rem; r++) {
            cvt_f32_tpp(in[nn * bn + r][k], in_f32[nn * bn + r][k]);
            add_tpp(
                in_f32[nn * bn + r][k],
                out_f32[nn * bn + r][k],
                out_f32[nn * bn + r][k]);
            if (act == "relu") {
              relu_fwd_tpp(
                  out_f32[nn * bn + r][k],
                  out_f32[nn * bn + r][k],
                  relu_mask[nn * bn + r][k]);
            }
            if (p > 0 && training) {
              dropout_fwd_tpp(
                  out_f32[nn * bn + r][k],
                  (void*)get_rng_state(),
                  out[nn * bn + r][k],
                  dp_mask[nn * bn + r][k]);
            } else
              cvt_tpp(out_f32[nn * bn + r][k], out[nn * bn + r][k]);
          }
        }
      }
    }
  } else {
    auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
        bn, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));

    RECORD_SCOPE(rgewo_gemm, {t_in, t_wt_V});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for collapse(2)
      for (int n = 0; n < nn; n++) {
        for (int k = 0; k < nk; k++) {
          brgemm_tpp(in_dst[n][0][0], wt_V[k][0], out_f32[n][0][k], nc);
          cvt_f32_tpp(in[n][0][k], in_f32[n][0][k]);
          add_tpp(in_f32[n][0][k], out_f32[n][0][k], out_f32[n][0][k]);
          if (act == "relu") {
            relu_fwd_tpp(
                out_f32[n][0][k], out_f32[n][0][k], relu_mask[n][0][k]);
          }
          if (p > 0 && training) {
            dropout_fwd_tpp(
                out_f32[n][0][k],
                (void*)get_rng_state(),
                out[n][0][k],
                dp_mask[n][0][k]);
          } else
            cvt_tpp(out_f32[n][0][k], out[n][0][k]);
        }
      }
      if (rem > 0) {
        DECL_VLA_PTR_PT(T, in, [nk][bk], t_in);
        DECL_VLA_PTR_PT(T, in_dst, [nc][bcp], t_in_dst);
        DECL_VLA_PTR_PT(T, out, [nk][bk], t_out);
        DECL_VLA_PTR_PT(float, out_f32, [nk][bk], t_out_f32);
        DECL_VLA_PTR_PT(float, in_f32, [nk][bk], t_in_f32);
        DECL_VLA_PTR_PT(short, relu_mask, [nk][rd], t_relu_mask);
        DECL_VLA_PTR_PT(short, dp_mask, [nk][rd], t_dp_mask);

        auto brgemm_tpp = SCOPEITGEMM((BrgemmTPP<T, float>(
            rem, bk, bcp, bcp, bk * bcp, nc * bcp, bk, nk * bk, 0.0, 0, nc)));
        auto relu_fwd_tpp =
            SCOPEIT(ReLUFwdTPP<float>(1, bk, nk * bk, nk * bk, true), ACT);
        auto dropout_fwd_tpp = SCOPEIT(
            (DropOutFwdTPP<float, T>(1, bk, nk * bk, nk * bk, p)), DROPOUT);
        auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(1, bk, K, K)), EW_COPY);
        auto cvt_f32_tpp =
            SCOPEIT((ConvertTPP<T, float>(1, bk, K, K)), EW_COPY);
        auto add_tpp = SCOPEIT((AddTPP<float, float>(1, bk, K, K)), EW_ADD);

        for (int k = 0; k < nk; k++) {
          brgemm_tpp(in_dst[nn * bn][0], wt_V[k][0], out_f32[nn * bn][k], nc);

          for (int r = 0; r < rem; r++) {
            cvt_f32_tpp(in[nn * bn + r][k], in_f32[nn * bn + r][k]);
            add_tpp(
                in_f32[nn * bn + r][k],
                out_f32[nn * bn + r][k],
                out_f32[nn * bn + r][k]);
            if (act == "relu") {
              relu_fwd_tpp(
                  out_f32[nn * bn + r][k],
                  out_f32[nn * bn + r][k],
                  relu_mask[nn * bn + r][k]);
            }
            if (p > 0 && training) {
              dropout_fwd_tpp(
                  out_f32[nn * bn + r][k],
                  (void*)get_rng_state(),
                  out[nn * bn + r][k],
                  dp_mask[nn * bn + r][k]);
            } else
              cvt_tpp(out_f32[nn * bn + r][k], out[nn * bn + r][k]);
          }
        }
      }
    }
  }
  return {t_out, t_relu_mask, t_dp_mask};
} else {
  auto t_in = inputs[i++];
  at::Tensor t_bias = at::empty(0);
  if (apply_bias)
    t_bias = inputs[i++];

  auto in_sizes = t_in.sizes();
  auto N = in_sizes[0];
  auto C = in_sizes[1];
  auto bn = align;
  auto nn = N / bn;
  auto rem = N % bn;

  at::Tensor t_in_f32;
  if (t_in.dtype() == at::kFloat)
    t_in_f32 = t_in;
  else
    t_in_f32 = at::empty({N, C});

  int rd = (C + 15) / 16;

  at::Tensor t_relu_mask = at::empty({N, rd}, at::kShort);
  at::Tensor t_dp_mask = at::empty({N, rd}, at::kShort);

  DECL_VLA_PTR_PT(T, in, [bn][C], t_in);
  DECL_VLA_PTR_PT(float, in_f32, [bn][C], t_in_f32);
  DECL_VLA_PTR_PT(float, bias, [C], t_bias);
  DECL_VLA_PTR_PT(short, relu_mask, [bn][rd], t_relu_mask);
  DECL_VLA_PTR_PT(short, dp_mask, [bn][rd], t_dp_mask);

  auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(bn, C)), EW_COPY);
  auto relu_fwd_tpp = SCOPEIT(ReLUFwdTPP<float>(bn, C, true), ACT);
  auto dropout_fwd_tpp = SCOPEIT((DropOutFwdTPP<float, T>(bn, C, p)), DROPOUT);
  auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(bn, C)), EW_COPY);

  if (apply_bias) {
    RECORD_SCOPE(rgo_eltw, {t_in});
    {
      auto add_bias_tpp = SCOPEIT(AddBiasTPP<float>(bn, C), BIAS);
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++) {
        cvt_f32_tpp(in[n][0], in_f32[n][0]);
        add_bias_tpp(bias[0], in_f32[n][0]);
        if (act == "relu") {
          relu_fwd_tpp(in_f32[n][0], in_f32[n][0], relu_mask[n][0]);
        }
        if (p > 0 && training) {
          dropout_fwd_tpp(
              in_f32[n][0], (void*)get_rng_state(), in[n][0], dp_mask[n][0]);
        } else
          cvt_tpp(in_f32[n][0], in[n][0]);
      }
      if (rem > 0) {
        DECL_VLA_PTR_PT(T, in, [C], t_in);
        DECL_VLA_PTR_PT(float, in_f32, [C], t_in_f32);
        DECL_VLA_PTR_PT(short, relu_mask, [rd], t_relu_mask);
        DECL_VLA_PTR_PT(short, dp_mask, [rd], t_dp_mask);

        auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(rem, C)), EW_COPY);
        auto add_bias_tpp = SCOPEIT(AddBiasTPP<float>(rem, C), BIAS);
        auto relu_fwd_tpp = SCOPEIT(ReLUFwdTPP<float>(rem, C, true), ACT);
        auto dropout_fwd_tpp =
            SCOPEIT((DropOutFwdTPP<float, T>(rem, C, p)), DROPOUT);
        auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(rem, C)), EW_COPY);

        cvt_f32_tpp(in[nn * bn], in_f32[nn * bn]);
        add_bias_tpp(bias[0], in_f32[nn * bn]);
        if (act == "relu") {
          relu_fwd_tpp(in_f32[nn * bn], in_f32[nn * bn], relu_mask[nn * bn]);
        }
        if (p > 0 && training) {
          dropout_fwd_tpp(
              in_f32[nn * bn],
              (void*)get_rng_state(),
              in[nn * bn],
              dp_mask[nn * bn]);
        } else
          cvt_tpp(in_f32[nn * bn], in[nn * bn]);
      }
    }
  } else {
    RECORD_SCOPE(rgo_eltw, {t_in});
    {
      RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
      for (int n = 0; n < nn; n++) {
        cvt_f32_tpp(in[n][0], in_f32[n][0]);
        if (act == "relu") {
          relu_fwd_tpp(in_f32[n][0], in_f32[n][0], relu_mask[n][0]);
        }
        if (p > 0 && training) {
          dropout_fwd_tpp(
              in_f32[n][0], (void*)get_rng_state(), in[n][0], dp_mask[n][0]);
        } else
          cvt_tpp(in_f32[n][0], in[n][0]);
      }
      if (rem > 0) {
        DECL_VLA_PTR_PT(T, in, [C], t_in);
        DECL_VLA_PTR_PT(float, in_f32, [C], t_in_f32);
        DECL_VLA_PTR_PT(short, relu_mask, [rd], t_relu_mask);
        DECL_VLA_PTR_PT(short, dp_mask, [rd], t_dp_mask);

        auto cvt_f32_tpp = SCOPEIT((ConvertTPP<T, float>(rem, C)), EW_COPY);
        auto relu_fwd_tpp = SCOPEIT(ReLUFwdTPP<float>(rem, C, true), ACT);
        auto dropout_fwd_tpp =
            SCOPEIT((DropOutFwdTPP<float, T>(rem, C, p)), DROPOUT);
        auto cvt_tpp = SCOPEIT((ConvertTPP<float, T>(rem, C)), EW_COPY);

        cvt_f32_tpp(in[nn * bn], in_f32[nn * bn]);
        if (act == "relu") {
          relu_fwd_tpp(in_f32[nn * bn], in_f32[nn * bn], relu_mask[nn * bn]);
        }
        if (p > 0 && training) {
          dropout_fwd_tpp(
              in_f32[nn * bn],
              (void*)get_rng_state(),
              in[nn * bn],
              dp_mask[nn * bn]);
        } else
          cvt_tpp(in_f32[nn * bn], in[nn * bn]);
      }
    }
  }

  return {t_in, t_relu_mask, t_dp_mask};
}

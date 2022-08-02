#if 1
{
  int i = 0;
  at::Tensor t_in = inputs[i++];
  at::Tensor t_idx = inputs[i++];

  auto in_sizes = t_in.sizes();
  auto C = in_sizes[1];
  auto Nout = t_idx.sizes()[0];
  const int bc = align;
  auto nc = C / align;

  auto t_out = t_in.new_empty({Nout, C});

  DECL_VLA_PTR_PT(T, in, [nc][bc], t_in);
  DECL_VLA_PTR_PT(T, out, [nc][bc], t_out);
  DECL_VLA_PTR_PT(int64_t, idx, [1], t_idx);

#pragma omp parallel for
  for (int n = 0; n < Nout; n++) {
    int64_t i = idx[n][0];
    for (int c = 0; c < nc; c++) {
#pragma omp simd
      for (int b = 0; b < bc; b++) {
        out[n][c][b] = in[i][c][b];
      }
    }
  }
  return t_out;
}
#else
{
  int i = 0;
  at::Tensor t_in = inputs[i++];
  at::Tensor t_idx = inputs[i++];

  auto in_sizes = t_in.sizes();
  auto C = in_sizes[1];
  auto Nout = t_idx.sizes()[0];
  auto bc = align;
  auto nc = C / align;
  auto bn = align;
  auto nn = Nout / align;
  auto rem = Nout % align;

  auto t_out = t_in.new_empty({Nout, C});

  DECL_VLA_PTR_PT(T, in, [nc][bc], t_in);
  DECL_VLA_PTR_PT(T, out, [bn][nc][bc], t_out);
  DECL_VLA_PTR_PT(int64_t, idx, [bn], t_idx);

#pragma omp parallel for
  for (int n = 0; n < nn; n++) {
    std::sort(&idx[n][0], &idx[n][bn]);
    for (int bb = 0; bb < bn; bb++) {
      int i = idx[n][bb];
#pragma GCC unroll 4
      for (int c = 0; c < nc; c++) {
#pragma omp simd
        for (int b = 0; b < bc; b++) {
          out[n][bb][c][b] = in[i][c][b];
        }
      }
    }
  }
  if (rem > 0) {
    DECL_VLA_PTR_PT(T, out, [nc][bc], t_out);
    DECL_VLA_PTR_PT(int64_t, idx, [1], t_idx);
    for (int bb = 0; bb < rem; bb++) {
      int i = idx[nn * bn + bb][0];
      for (int c = 0; c < nc; c++) {
#pragma omp simd
        for (int b = 0; b < bc; b++) {
          out[nn * bn + bb][c][b] = in[i][c][b];
        }
      }
    }
  }

  return t_out;
}
#endif

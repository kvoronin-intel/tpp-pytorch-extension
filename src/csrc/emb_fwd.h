auto t_wt = inputs[0];
auto t_inp = inputs[1];

int N = t_inp.size(0);
int nn = N / alignN;
int bn = alignN;
int E = t_wt.size(1);
// int nk = E/alignE;
// int bk = alignE;
int rem = N % alignN;

auto t_out = t_wt.new_empty({N, E});

// auto  wt = GetVLAPtr<T>( t_wt, { nk, bk});
auto wt = GetVLAPtr<T>(t_wt, {E});
auto inp = GetVLAPtr<Tind>(t_inp, {bn});
auto out = GetVLAPtr<T>(t_out, {bn, E});

auto emb_fwd_tpp = SCOPEIT((EmbeddingFwdTPP<T, Tind, T>(bn, E, E, E)), ROW_GT);
// auto emb_fwd_tpp = EmbeddingFwdTPP<T, T, Tind>(bn, bk, E, E);

{
  RECORD_SCOPE(remb, {t_wt, t_inp});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());
#pragma omp parallel for
    for (int n = 0; n < nn; n++) {
#if 0
      for(int k=0; k<nk; k++) {
        emb_fwd_tpp(wt[0][k], inp[n], out[n][0][k]);
      }
#else
      emb_fwd_tpp(wt[0], inp[n], out[n][0]);
#endif
    }
  }
  if (rem > 0) {
    auto inp = GetVLAPtr<Tind>(t_inp, {0});
    // auto  out = GetVLAPtr<T>( t_out, { nk, bk});
    auto out = GetVLAPtr<T>(t_out, {E});
    auto emb_fwd_tpp =
        SCOPEIT((EmbeddingFwdTPP<T, Tind, T>(rem, E, E, E)), ROW_GT);
    // auto emb_fwd_tpp = EmbeddingFwdTPP<T, T, Tind>(rem, bk, E, E);

    // for(int k=0; k<nk; k++)
    //   emb_fwd_tpp(wt[N][k], inp[nn*bn], out[nn*bn][k]);
    emb_fwd_tpp(wt[0], inp[nn * bn], out[nn * bn]);
  }
}

return t_out;

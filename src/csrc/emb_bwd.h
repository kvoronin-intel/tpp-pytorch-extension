auto t_grad_out = inputs[0];
auto t_wt = inputs[1];
auto t_indices = inputs[2];
Tind* indices = t_indices.data_ptr<Tind>();

long numIndices = t_indices.size(0);
long maxInd = t_wt.size(0);
auto pydt = (maxInd < INT_MAX ? at::kInt : at::kLong);
auto t_scratch = at::empty({4 * numIndices}, pydt);

std::tuple<at::Tensor, at::Tensor, at::Tensor> ret;

if (maxInd < INT_MAX) {
  typedef int Tidx;
  std::pair<Tidx, Tidx>* scratch = (std::pair<Tidx, Tidx>*)t_scratch.data_ptr();
  init_scratch<Tind, Tidx>(scratch, indices, numIndices);
  ret = coalescing_preprocessing<Tidx>(numIndices, maxInd, scratch);
} else {
  typedef long Tidx;
  std::pair<Tidx, Tidx>* scratch = (std::pair<Tidx, Tidx>*)t_scratch.data_ptr();
  init_scratch<Tind, Tidx>(scratch, indices, numIndices);
  ret = coalescing_preprocessing<Tidx>(numIndices, maxInd, scratch);
}

at::Tensor t_uniqueIndices = std::get<0>(ret);
at::Tensor t_outputRowOffsets = std::get<1, at::Tensor>(ret);
at::Tensor t_outputRows = std::get<2>(ret);

auto N = t_uniqueIndices.size(0);
auto E = t_grad_out.size(1);

auto t_values = t_grad_out.new_empty({N, E});

DECL_VLA_PTR_PT(T, grad_out, [E], t_grad_out);
DECL_VLA_PTR_PT(T, values, [E], t_values);

int* input = t_outputRows.data_ptr<int>();
int* or_offsets = t_outputRowOffsets.data_ptr<int>();

auto emb_bwd_tpp = SCOPEIT((EmbeddingBwdTPP<T, int, T>(E)), EW_RED);

{
  RECORD_SCOPE(gremb, {t_grad_out});
  {
    RECORD_FUNCTION("parallel_for", std::vector<c10::IValue>());

#pragma omp parallel for
    for (int n = 0; n < N; n++) {
      auto start = or_offsets[n];
      auto end = or_offsets[n + 1];

      emb_bwd_tpp(grad_out[0], &input[start], values[n], end - start);
    }
  }
}

auto t_indices_ = t_uniqueIndices.view({1, -1});
auto t_grad_weight =
    at::_sparse_coo_tensor_unsafe(t_indices_, t_values, t_wt.sizes());
t_grad_weight._coalesced_(true);
// printf("gwc %d\n",t_grad_weight.is_coalesced());

return t_grad_weight;

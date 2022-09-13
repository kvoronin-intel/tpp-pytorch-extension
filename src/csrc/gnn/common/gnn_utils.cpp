#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdlib>

#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <iostream>
#include <mutex>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace pcl;
REGISTER_SCOPE(gather, "gather");
REGISTER_SCOPE(scatter, "scatter");

void lfsr_Xwide(unsigned int* rng_state, unsigned int* prng_out, int width) {
  int res = 16 - width;
  __mmask16 msk = res ? (1 << res) - 1 : 255;
  __m512i vrng_s0 = _mm512_loadu_epi32(rng_state);
  __m512i vrng_s1 = _mm512_loadu_epi32(rng_state + 16);
  __m512i vrng_s2 = _mm512_loadu_epi32(rng_state + 32);
  __m512i vrng_s3 = _mm512_loadu_epi32(rng_state + 48);

  __m512i vrng = _mm512_add_epi32(vrng_s3, vrng_s0);
  _mm512_mask_compressstoreu_epi32(prng_out, msk, vrng);

  __m512i vtmp0 = _mm512_slli_epi32(vrng_s1, 9);
  vrng_s2 = _mm512_xor_epi32(vrng_s2, vrng_s0);
  vrng_s3 = _mm512_xor_epi32(vrng_s3, vrng_s1);
  vrng_s1 = _mm512_xor_epi32(vrng_s1, vrng_s2);
  vrng_s0 = _mm512_xor_epi32(vrng_s0, vrng_s3);
  vrng_s2 = _mm512_xor_epi32(vrng_s2, vtmp0);
  vtmp0 = _mm512_slli_epi32(vrng_s3, 11);
  __m512i vtmp1 = _mm512_srli_epi32(vrng_s3, 21);
  vrng_s3 = _mm512_or_epi32(vtmp0, vtmp1);
  _mm512_storeu_epi32(rng_state, vrng_s0);
  _mm512_storeu_epi32(rng_state + 16, vrng_s1);
  _mm512_storeu_epi32(rng_state + 32, vrng_s2);
  _mm512_storeu_epi32(rng_state + 48, vrng_s3);
}

at::Tensor gather_features(const int alignN, std::vector<at::Tensor> inputs) {
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "gather.h"
  } else {
    typedef bfloat16 T;
#include "gather.h"
  }
}

void scatter_features(const int alignN, std::vector<at::Tensor> inputs) {
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "scatter.h"
  } else {
    typedef bfloat16 T;
#include "scatter.h"
  }
}

std::vector<at::Tensor> find_nodes(
    std::vector<at::Tensor> inputs,
    std::string ntype) {
  auto t_pnd_s_in = inputs[0];
  auto t_pnd_orig = inputs[1];
  auto t_srcnodes = inputs[2];
  auto t_lnodes = inputs[3];

  typedef long Ta;
  typedef int Ts;
#include "find_nodes.h"
}

std::vector<at::Tensor> map_nodes(std::vector<at::Tensor> inputs) {
  auto t_db = inputs[0];
  auto t_sn_orig = inputs[1];
  auto t_sn_batch = inputs[2];
  auto t_sn_part = inputs[3];

#include "map_nodes_simple_vec.h"
}

std::vector<at::Tensor> find_n_map_nodes(std::vector<at::Tensor> inputs) {
  auto t_db = inputs[0];
  auto t_pnd_solid = inputs[1];
  auto t_pnd_orig = inputs[2];
  auto t_srcnodes = inputs[3];
  auto t_lnodes = inputs[4];

  typedef long Ta;
  typedef int Ts;
#include "find_n_map_solid_nodes.h"
}

void cache_store_n(
    int N,
    int cp,
    long* hmap,
    int* rptr,
    long* nodes,
    int* age,
    int rval,
    int hval,
    at::Tensor t_in_f,
    at::Tensor t_out_f) {
  auto t_loc = at::empty({N}, at::kLong);
  long* loc = t_loc.data_ptr<long>();
#pragma omp parallel for
  for (int j = 0; j < N; j++)
    loc[j] = cp + j;

  int n = 0;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      if (rptr[loc[n + v]] != rval)
        hmap[n + v] = hval;
      rptr[loc[n + v]] = (int)nodes[n + v];
      age[loc[n + v]] = 0;
      hmap[nodes[n + v]] = loc[n + v];
    }
  }
  if (n < N) {
    int rem = N - n;
    for (int r = 0; r < rem; r++) {
      if (rptr[loc[n + r]] != rval)
        hmap[n + r] = hval;
      rptr[loc[n + r]] = (int)nodes[n + r];
      age[loc[n + r]] = 0;
      hmap[nodes[n + r]] = loc[n + r];
    }
  }
  std::vector<at::Tensor> in;
  in.push_back(t_in_f);
  in.push_back(t_loc);
  in.push_back(t_out_f);
  int alignN = N > 32 ? 32 : N;
  if (N > 0 && alignN > 0)
    scatter_features(alignN, in);
}

void cache_store(
    std::vector<at::Tensor> inputs,
    int cache_size,
    int hval,
    int rval) {
  auto t_hmap = inputs[0];
  auto t_rptr = inputs[1];
  auto t_age = inputs[2];
  auto t_nodes = inputs[3];
  auto t_st_feats = inputs[4];
  auto t_feats = inputs[5];
  auto t_sz_feats = inputs[6];
  auto t_feats_sz = inputs[7];
  auto t_cache_p = inputs[8];

  int* rptr = t_rptr.data_ptr<int>();
  long* hmap = t_hmap.data_ptr<long>();
  long* nodes = t_nodes.data_ptr<long>();
  int* age = t_age.data_ptr<int>();
  int* cp = t_cache_p.data_ptr<int>();

  auto N = t_nodes.size(0);
  if (N > 0) {
    int n = 0;
#pragma omp parallel for lastprivate(n)
    for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
#pragma omp simd
      for (int v = 0; v < 16; v++) {
        if (hmap[nodes[n + v]] != hval)
          rptr[n + v] = rval;
      }
    }
    if (n < N) {
      int rem = N - n;
      for (int r = 0; r < rem; r++) {
        if (hmap[nodes[n + r]] != hval)
          rptr[n + r] = rval;
      }
    }
    int size = cache_size - cp[0];
    if (size >= N) {
      cache_store_n(
          N, cp[0], hmap, rptr, nodes, age, rval, hval, t_feats, t_st_feats);
      cp[0] += N;
    } else {
      cache_store_n(
          size,
          cp[0],
          hmap,
          rptr,
          nodes,
          age,
          rval,
          hval,
          t_sz_feats,
          t_st_feats);
      int rem = N - size;
      cache_store_n(
          rem,
          0,
          hmap,
          rptr,
          nodes + size,
          age,
          rval,
          hval,
          t_feats_sz,
          t_st_feats);
      cp[0] = rem;
    }
  }
}

std::vector<at::Tensor> cache_load(
    std::vector<at::Tensor> inputs,
    int level,
    int minlife,
    int life) {
  auto t_hmap = inputs[0];
  auto t_oid = inputs[1];
  auto t_feats = inputs[2];
  at::Tensor t_age = at::empty(0, at::kInt);
  if (level > 0)
    t_age = inputs[3];

#include "cache_load.h"
}

void gather_n_store_offset(
    std::vector<at::Tensor> inputs,
    long offseti,
    long offsetv) {
  auto t_in = inputs[0];
  auto t_ind = inputs[1];
  auto t_out = inputs[2];

  auto N = t_ind.size(0);
  long* in = t_in.data_ptr<long>();
  long* ind = t_ind.data_ptr<long>();
  long* out = t_out.data_ptr<long>();

  ind += offseti;
  out += offsetv;

  int n = 0;
#pragma omp parallel for lastprivate(n)
  for (n = 0; n < ALIGNDOWN(N, 16); n += 16) {
#pragma omp simd
    for (int v = 0; v < 16; v++) {
      out[n + v] = in[ind[n + v]];
    }
  }
  if (n < N) {
    int rem = N - n;
    for (int r = 0; r < rem; r++) {
      out[n + r] = in[ind[n + r]];
    }
  }
}

std::vector<at::Tensor> node_sampling(
    std::vector<at::Tensor> inputs,
    const int hil,
    const int thres) {
  auto t_deg = inputs[0];
  auto t_xnbn = inputs[1];
  auto t_xrbn = inputs[2];

  if (t_deg.dtype() == at::kLong) {
    typedef long T;
#include "node_sampling.h"
  } else if (t_deg.dtype() == at::kInt) {
    typedef int T;
#include "node_sampling.h"
  } else if (t_deg.dtype() == at::kFloat) {
    typedef float T;
#include "node_sampling.h"
  }
}

void mapped_spmm(std::vector<at::Tensor> inputs, string op, string redop) {
  auto t_source = inputs[0];
  auto t_indptr = inputs[1];
  auto t_sind   = inputs[2];
  auto t_dind   = inputs[3];
  auto t_dest   = inputs[4];

  long *indptr = t_indptr.data_ptr<long>();
  long *sind   = t_sind.data_ptr<long>();
  long *dind   = t_dind.data_ptr<long>();

  auto N = t_indptr.size(0);
  auto F = t.dest.size(1);

  libxsmm_meltw_opreduce_vecs_flags opredop_flags;
  if(op == "add")
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_ADD;
  else if(op == "sub")
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_SUM;
  else if(op == "mul")
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_MUL;
  else if(op == "div")
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_DIV;
  else if(op == "copylhs") {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY;
  }
  else if(op == "copyrhs") {
    opredop_flags = LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OP_COPY;
  }

  if(op == "copylhs")
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | 
        LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN);
  else if(op == "copyrhs") {
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags | 
        LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIN_VECIDX);
    if(!has_idx)
      opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags |
          LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VECIDX);
  }
  else {
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags |
        LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_OPORDER_VECIDX_VECIN);
    if (has_idx) {
      opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags |
          LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_INDEXED_VEC);
    } else {
      opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags |
          LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_IMPLICIT_INDEXED_VEC);
    }
  }

  if(redop == "sum") 
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags |
        LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_SUM);
  else if(redop == "min")
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags |
        LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MIN);
  else if(redop == "max")
    opredop_flags = (libxsmm_meltw_opreduce_vecs_flags)(opredop_flags |
        LIBXSMM_MELTW_FLAG_OPREDUCE_VECS_REDOP_MAX);

  if(t_dest.dtype() == at::kFloat) {
    typedef float T;
    DECL_VLA_PTR_PT(T, dest, [F], t_dest);
    DECL_VLA_PTR_PT(T, source, [F], t_source);

    libxsmm_meltwfunction_opreduce_vecs_idx kernel = NULL;
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(
        F, &F, &F, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
        (sizeof(indptr) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32, opredop_flags);

#pragma omp parallel 
    {
#pragma omp for schedule(dynamic)
      for(int i=0; i<N-1; i++) {
        long row_start = indptr[i];
        long row_end = indptr[i+1];
        libxsmm_meltw_opreduce_vecs_idx_param params;
        params.n = row_end - row_start;
        params.indices = &sind[row_start];
        params.in_matrix = source[0];
        params.out_vec = dest[dind[i]];
        params.scale_vals = nullptr;
        params.in_matrix2 = NULL;
        params.indices2 = NULL;
        kernel(&params);
      }
    }
  }
  else if(t_dest.dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    DECL_VLA_PTR_PT(T, dest, [F], t_dest);
    DECL_VLA_PTR_PT(T, source, [F], t_source);

    libxsmm_meltwfunction_opreduce_vecs_idx kernel = NULL;
    kernel = libxsmm_dispatch_meltw_opreduce_vecs_idx(
        F, &F, &F, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16,
        (sizeof(indptr) == 8) ? LIBXSMM_DATATYPE_I64 : LIBXSMM_DATATYPE_I32, opredop_flags);

#pragma omp parallel 
    {
#pragma omp for schedule(dynamic)
      for(int i=0; i<N; i++) {
        long row_start = indptr[i];
        long row_end = indptr[i+1];
        libxsmm_meltw_opreduce_vecs_idx_param params;
        params.n = row_end - row_start;
        params.indices = &sind[row_start];
        params.in_matrix = source[0];
        params.out_vec = dest[dind[i]];
        params.scale_vals = NULL;
        params.in_matrix2 = NULL;
        params.indices2 = NULL;
        kernel(&params);
      }
    }
  }
}

void affinitize_cores(const int nthreads, const int num_workers) {
#pragma omp parallel
  {
    int mytid = omp_get_thread_num();
    cpu_set_t my_set;
    CPU_ZERO(&my_set);
    CPU_SET(num_workers + mytid, &my_set);

    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
#ifdef DEBUG
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
      perror("sched_getaffinity");
      assert(false);
    }
    long nproc = nthreads + num_workers;
    for (long i = 0; i < nproc; i++) {
      if (CPU_ISSET(i, &mask))
        printf("%d on core %ld\n", mytid, i);
    }
#endif
  }
}

REGISTER_SUBMODULE(_gnn_utils, m) {
  m.def("gather_features", &gather_features, "C++ Impl of feature gather");
  m.def("scatter_features", &scatter_features, "C++ Impl of feature scatter");
  m.def(
      "find_nodes",
      &find_nodes,
      "C++ Impl of func to gather halo & solid nodes");
  m.def(
      "map_nodes", &map_nodes, "C++ Impl of func to gather solid node mapping");
  m.def(
      "find_n_map_nodes",
      &find_n_map_nodes,
      "C++ Impl of func to gather solid nodes and get mapping");
  m.def(
      "cache_load",
      &cache_load,
      "C++ impl of func to gather features from cache");
  m.def(
      "cache_store",
      &cache_store,
      "C++ impl of func to scatter features to cache");
  m.def("node_sampling", &node_sampling, "C++ impl of func to sample nodes");
  m.def(
      "gather_n_store_offset",
      &gather_n_store_offset,
      "Gather and store long ints");
  m.def("affinitize_cores", &affinitize_cores, "Compute thread affinization");
}

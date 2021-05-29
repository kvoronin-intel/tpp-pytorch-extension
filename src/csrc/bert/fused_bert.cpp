
#include <ATen/record_function.h>
//#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>
#include "../init.h"
#include "../timing.h"
#include "../xsmm_functors.h"

using namespace pcl;
#include "../tensor_helper.h"

static int my_rank = guess_mpi_rank();

PassType globalPass = FWD;
ScopeType globalScope = st_other;
double debug_timers[MAX_THREADS][2][NUM_TIMERS];
double scope_timers[MAX_THREADS][NUM_SCOPES][NUM_TIMERS];
double master_scope_timers[NUM_SCOPES];
double scope_flops[MAX_THREADS][NUM_ALIGNED_SCOPES];
double master_debug_timers[2];

void reset_debug_timers() {
  master_debug_timers[0] = 0.0;
  master_debug_timers[1] = 0.0;
  for (int p = 0; p < NUM_SCOPES; p++) {
    master_scope_timers[p] = 0.0;
  }
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    for (int t = 0; t < NUM_TIMERS; t++) {
      debug_timers[tid][FWD][t] = 0.0;
      debug_timers[tid][BWD][t] = 0.0;
    }
    for (int p = 0; p < NUM_SCOPES; p++) {
      for (int t = 0; t < NUM_TIMERS; t++) {
        scope_timers[tid][p][t] = 0.0;
      }
      scope_flops[tid][p] = 0;
    }
  }
}

void print_debug_timers(int tid) {
  if (my_rank != 0)
    return;
  int max_threads = omp_get_max_threads();
  constexpr int maxlen = 5000;
  SafePrint<maxlen> printf;
  // printf("%-20s", "####");
  printf("### ##: %-11s: ", "#KEY#");
  for (int t = 0; t < LAST_TIMER; t++) {
    printf(" %7s", DebugTimerNames[t]);
  }
  printf(" %8s  %8s\n", "Total", "MTotal");
  for (int i = 0; i < max_threads; i++) {
    if (tid == -1 || tid == i) {
      for (int p = 0; p < 2; p++) {
        double total = 0.0;
        printf("TID %2d: %-11s: ", i, p == 1 ? "BWD" : "FWD");
        for (int t = 0; t < LAST_TIMER; t++) {
          printf(" %7.1f", debug_timers[i][p][t] * 1e3);
          total += debug_timers[i][p][t];
        }
        printf(" %8.1f  %8.1f\n", total * 1e3, master_debug_timers[p] * 1e3);
      }
      for (int p = 0; p < NUM_SCOPES; p++) {
        double total = 0.0;
        printf("TID %2d: %-11s: ", i, ScopeNames[p]);
        for (int t = 0; t < LAST_TIMER; t++) {
          printf(" %7.1f", scope_timers[i][p][t] * 1e3);
          total += scope_timers[i][p][t];
        }
        long t_flops = 0;
        for (int f = 0; f < max_threads; f++)
          t_flops += scope_flops[f][p];
        if (t_flops > 0.0) {
          printf(
              " %8.1f  %8.1f  %8.3f (%4.2f) %6.3f\n",
              total * 1e3,
              master_scope_timers[p] * 1e3,
              t_flops * 1e-9,
              t_flops * 100.0 / (scope_flops[i][p] * max_threads),
              t_flops * 1e-12 / scope_timers[i][p][BRGEMM]);
        } else {
          printf(" %8.1f  %8.1f\n", total * 1e3, master_scope_timers[p] * 1e3);
        }
      }
    }
  }
  printf.print();
}

#ifdef PURE_GEMM_TIME
template <typename Tin, typename Tout>
class TBrgemmExtTPP {
 public:
  TBrgemmExtTPP() {}
  TBrgemmExtTPP(
      long M,
      long N,
      long K,
      long str_a,
      long str_b,
      float beta = 1.0,
      XformTPP::XFORM_TYPE c_trans = XformTPP::XFORM_NONE_TPP,
      int a_trans = 0)
      : M(M),
        N(N),
        K(K),
        beta(beta),
        c_trans(c_trans),
        brgemm(),
        xform(),
        add() {
    // auto dt_in = XsmmDtype<Tin>();
    auto dt_out = XsmmDtype<Tout>();
    if (dt_out == LIBXSMM_DATATYPE_F32 && c_trans == XformTPP::XFORM_N2V_TPP)
      c_trans = XformTPP::XFORM_NONE_TPP;
    auto beta_ = beta;

    if (c_trans != XformTPP::XFORM_NONE_TPP) {
      beta_ = 0.0;
      xform = XformExtTPP<Tout>(M, N, c_trans);
    }
    brgemm = BrgemmTPP<Tin, Tout>(M, N, K, str_a, str_b, beta_, a_trans);
    if (beta_ != beta) {
      add = AddTPP<Tout, Tout>(M, N);
    }
    xform_type = c_trans == XformTPP::XFORM_N2V_TPP ? VNNI : XPOSE;
  }

  void operator()(Tin* A, Tin* B, Tout* C, long count) {
    if (c_trans == XformTPP::XFORM_NONE_TPP) {
      ScopedTimer _t(BRGEMM, globalPass, 2 * M * N * K * count);
      brgemm(A, B, C, count);
    } else {
      Tout tmp_C[M * N];
      {
        ScopedTimer _t(BRGEMM, globalPass, 2 * M * N * K * count);
        brgemm(A, B, tmp_C, count);
      }
      if (beta == 0.0) {
        ScopedTimer _t(xform_type, globalPass);
        xform(tmp_C, C);
      } else {
        Tout tmp[M * N];
        {
          ScopedTimer _t(xform_type, globalPass);
          xform(tmp_C, tmp);
        }
        {
          ScopedTimer _t(EW_ADD, globalPass);
          add(C, tmp, C);
        }
      }
    }
  }

  void ref(Tin* A, Tin* B, Tout* C, long count) {
    if (c_trans == XformTPP::XFORM_NONE_TPP) {
      ScopedTimer _t(BRGEMM, globalPass, 2 * M * N * K * count);
      brgemm.ref(A, B, C, count);
    } else {
      Tout tmp_C[M * N];
      {
        ScopedTimer _t(BRGEMM, globalPass, 2 * M * N * K * count);
        brgemm(A, B, tmp_C, count);
      }
      if (beta == 0.0) {
        ScopedTimer _t(xform_type, globalPass);
        xform.ref(tmp_C, C);
      } else {
        Tout tmp[M * N];
        {
          ScopedTimer _t(xform_type, globalPass);
          xform.ref(tmp_C, tmp);
        }
        {
          ScopedTimer _t(EW_ADD, globalPass);
          add.ref(C, tmp, C);
        }
      }
    }
  }

 private:
  long M, N, K;
  float beta;
  XformTPP::XFORM_TYPE c_trans;
  BrgemmTPP<Tin, Tout> brgemm;
  XformExtTPP<Tout> xform;
  AddTPP<Tout, Tout> add;
  DebugTimer xform_type;
};

#define BrgemmExtTPP TBrgemmExtTPP
#endif

template <typename T>
inline void omp_reduce_buf(
    int num_threads,
    int N,
    float** ptrs,
    T* buf,
    bool accumulate = false) {
  ScopedTimer _t(EW_RED, globalPass);
#pragma omp for
  for (int i = 0; i < N; i++) {
    float sum = 0.0;
    for (int j = 0; j < num_threads; j++) {
      sum += ptrs[j][i];
    }
    if (accumulate) {
      buf[i] += sum;
    } else {
      buf[i] = sum;
    }
  }
}

std::vector<at::Tensor> fused_self_attention_fwd(
    float p,
    std::vector<at::Tensor> inputs,
    bool training) {
  if (inputs[6].dtype() == at::kFloat) {
    typedef float T;
#include "fused_self_attention_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_self_attention_fwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_self_attention_bwd(
    float p,
    std::vector<at::Tensor> inputs) {
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_self_attention_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_self_attention_bwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_dense_dropout_layernorm_fwd(
    float p,
    float eps,
    std::vector<at::Tensor> inputs,
    bool training) {
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_dense_dropout_layernorm_fwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_dense_dropout_layernorm_bwd(
    float p,
    std::vector<at::Tensor> inputs) {
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_dense_dropout_layernorm_bwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_dense_gelu_fwd(
    at::Tensor t_in,
    at::Tensor t_wt,
    at::Tensor t_bias,
    bool training) {
  if (t_in.dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_gelu_fwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_dense_gelu_fwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_dense_gelu_bwd(
    at::Tensor t_grad_out,
    at::Tensor t_gelu_in,
    at::Tensor t_in,
    at::Tensor t_wt) {
  if (t_grad_out.dtype() == at::kFloat) {
    typedef float T;
#include "fused_dense_gelu_bwd_tmpl.h"
  } else {
    typedef bfloat16 T;
#include "fused_dense_gelu_bwd_tmpl.h"
  }
}

std::vector<at::Tensor> fused_embedding_layernorm_dropout_fwd(
    float p,
    float eps,
    long H,
    long pad_id,
    std::vector<at::Tensor>& inputs,
    bool training) {
  if (inputs[4].dtype() == at::kFloat && inputs[6].dtype() == at::kFloat) {
    typedef float T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kBFloat16 && inputs[6].dtype() == at::kFloat) {
    typedef bfloat16 T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kFloat && inputs[6].dtype() == at::kBFloat16) {
    typedef float T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else if (
      inputs[4].dtype() == at::kBFloat16 &&
      inputs[6].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_fwd_tmpl.h"
  } else {
    PCL_ASSERT(0, "Should not come here\n");
  }
}

std::vector<at::Tensor> fused_embedding_layernorm_dropout_bwd(
    float p,
    long pad_id,
    std::vector<at::Tensor>& inputs) {
  if (inputs[0].dtype() == at::kFloat && inputs[6].dtype() == at::kFloat) {
    typedef float T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat16 && inputs[6].dtype() == at::kFloat) {
    typedef bfloat16 T;
    typedef float ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kFloat && inputs[6].dtype() == at::kBFloat16) {
    typedef float T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else if (
      inputs[0].dtype() == at::kBFloat16 &&
      inputs[6].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    typedef bfloat16 ET;
#include "fused_embedding_layernorm_dropout_bwd_tmpl.h"
  } else {
    PCL_ASSERT(0, "Should not come here\n");
  }
}

REGISTER_SUBMODULE(_fused_bert, m) {
  m.def("print_debug_timers", &print_debug_timers, "print_debug_timers");
  m.def("reset_debug_timers", &reset_debug_timers, "reset_debug_timers");

  m.def(
      "fused_self_attention_fwd",
      &fused_self_attention_fwd,
      "Pcl BERT forward");
  m.def(
      "fused_self_attention_bwd",
      &fused_self_attention_bwd,
      "Pcl BERT backward");
  m.def(
      "fused_dense_dropout_layernorm_fwd",
      &fused_dense_dropout_layernorm_fwd,
      "Pcl BERT forward");
  m.def(
      "fused_dense_dropout_layernorm_bwd",
      &fused_dense_dropout_layernorm_bwd,
      "Pcl BERT forward");
  m.def("fused_dense_gelu_fwd", &fused_dense_gelu_fwd, "Pcl BERT forward");
  m.def("fused_dense_gelu_bwd", &fused_dense_gelu_bwd, "Pcl BERT forward");
  m.def(
      "fused_embedding_layernorm_dropout_fwd",
      &fused_embedding_layernorm_dropout_fwd,
      "Pcl BERT forward");
  m.def(
      "fused_embedding_layernorm_dropout_bwd",
      &fused_embedding_layernorm_dropout_bwd,
      "Pcl BERT backward");
}

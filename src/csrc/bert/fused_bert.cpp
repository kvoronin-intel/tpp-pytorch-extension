
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

REGISTER_SCOPE(b_emb, "b_emb");
REGISTER_SCOPE(q_gemm, "q_gemm");
REGISTER_SCOPE(k_gemm, "k_gemm");
REGISTER_SCOPE(v_gemm, "v_gemm");
REGISTER_SCOPE(a_gemm, "a_gemm");
REGISTER_SCOPE(c_gemm, "c_gemm");
REGISTER_SCOPE(o_gemm, "o_gemm");
REGISTER_SCOPE(i_gemm, "i_gemm");

REGISTER_SCOPE(db_emb, "db_emb");
REGISTER_SCOPE(diq_gemm, "diq_gemm");
REGISTER_SCOPE(dik_gemm, "dik_gemm");
REGISTER_SCOPE(div_gemm, "div_gemm");
REGISTER_SCOPE(dica_gemm, "dica_gemm");
REGISTER_SCOPE(dii_gemm, "dii_gemm");
REGISTER_SCOPE(dio_gemm, "dio_gemm");
REGISTER_SCOPE(dwqkv_gemm, "dwqkv_gemm");
REGISTER_SCOPE(dwq_gemm, "dwq_gemm");
REGISTER_SCOPE(dwk_gemm, "dwk_gemm");
REGISTER_SCOPE(dwv_gemm, "dwv_gemm");
REGISTER_SCOPE(dwa_gemm, "dwa_gemm");
REGISTER_SCOPE(dwc_gemm, "dwc_gemm");
REGISTER_SCOPE(dwi_gemm, "dwi_gemm");
REGISTER_SCOPE(dwo_gemm, "dwo_gemm");
REGISTER_SCOPE(dqkv_bias, "dqkv_bias");
REGISTER_SCOPE(di_bias, "di_bias");
REGISTER_SCOPE(do_bias, "do_bias");

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
      ScopedTimer _t(BRGEMM, 2 * M * N * K * count);
      brgemm(A, B, C, count);
    } else {
      Tout tmp_C[M * N];
      {
        ScopedTimer _t(BRGEMM, 2 * M * N * K * count);
        brgemm(A, B, tmp_C, count);
      }
      if (beta == 0.0) {
        ScopedTimer _t(xform_type);
        xform(tmp_C, C);
      } else {
        Tout tmp[M * N];
        {
          ScopedTimer _t(xform_type);
          xform(tmp_C, tmp);
        }
        {
          ScopedTimer _t(EW_ADD);
          add(C, tmp, C);
        }
      }
    }
  }

  void ref(Tin* A, Tin* B, Tout* C, long count) {
    if (c_trans == XformTPP::XFORM_NONE_TPP) {
      ScopedTimer _t(BRGEMM, 2 * M * N * K * count);
      brgemm.ref(A, B, C, count);
    } else {
      Tout tmp_C[M * N];
      {
        ScopedTimer _t(BRGEMM, 2 * M * N * K * count);
        brgemm(A, B, tmp_C, count);
      }
      if (beta == 0.0) {
        ScopedTimer _t(xform_type);
        xform.ref(tmp_C, C);
      } else {
        Tout tmp[M * N];
        {
          ScopedTimer _t(xform_type);
          xform.ref(tmp_C, tmp);
        }
        {
          ScopedTimer _t(EW_ADD);
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
  ScopedTimer _t(EW_RED);
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
  GlobalPass _gp(FWD);
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
  GlobalPass _gp(BWD);
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
  GlobalPass _gp(FWD);
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
  GlobalPass _gp(BWD);
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
  GlobalPass _gp(FWD);
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
  GlobalPass _gp(BWD);
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
  GlobalPass _gp(FWD);
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
  GlobalPass _gp(BWD);
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

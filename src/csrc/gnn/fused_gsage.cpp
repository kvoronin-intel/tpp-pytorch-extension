
#include <ATen/record_function.h>
#include <torch/extension.h>

#include <iostream>
#include <mutex>
#include <vector>
#include "../ext_tpp.h"
#include "../init.h"
#include "../timing.h"
#include "../xsmm_functors.h"

using namespace pcl;
#include "../tensor_helper.h"

static int my_rank = guess_mpi_rank();

REGISTER_SCOPE(go_gemm, "go_gemm");
REGISTER_SCOPE(gdi_gemm, "gdi_gemm");
REGISTER_SCOPE(gdw_gemm, "gdw_gemm");
REGISTER_SCOPE(gdbias, "gdbias");
REGISTER_SCOPE(go_dropout, "go_dropout");
REGISTER_SCOPE(gdo_dropout, "gdo_dropout");

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

template <typename Tin, typename Tout>
inline void omp_reduce_buf(
    int num_threads,
    int N,
    Tin** ptrs,
    Tout* buf,
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

std::vector<at::Tensor> fused_gsage_mlp_fwd(
    int align,
    float p,
    std::string act,
    bool res,
    bool training,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_gsage_mlp_flat_fwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_gsage_mlp_flat_fwd.h"
  }
}

std::vector<at::Tensor> fused_gsage_mlp_bwd(
    int align,
    float p,
    std::string act,
    bool res,
    std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "fused_gsage_mlp_flat_bwd.h"
  } else {
    typedef bfloat16 T;
#include "fused_gsage_mlp_flat_bwd.h"
  }
}

std::vector<at::Tensor> dropout_fwd(float p, at::Tensor inp, bool training) {
  GlobalPass _gp(FWD);
  if (inp.dtype() == at::kFloat) {
    typedef float T;
#include "dropout_fwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_fwd.h"
  }
}

at::Tensor dropout_bwd(float p, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "dropout_bwd.h"
  } else {
    typedef bfloat16 T;
#include "dropout_bwd.h"
  }
}

REGISTER_SUBMODULE(_fused_gsage, m) {
  m.def(
      "fused_gsage_mlp_fwd", &fused_gsage_mlp_fwd, "Pcl GraphSAGE MLP forward");
  m.def(
      "fused_gsage_mlp_bwd",
      &fused_gsage_mlp_bwd,
      "Pcl GraphSAGE MLP backward");
  m.def("dropout_fwd", &dropout_fwd, "Pcl Optimized Dropout FWD");
  m.def("dropout_bwd", &dropout_bwd, "Pcl Optimized Dropout BWD");
}

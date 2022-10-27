
#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cstdlib>

#include <sys/syscall.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <vector>
#include "ext_tpp.h"
#include "init.h"
#include "timing.h"
#include "xsmm_functors.h"

using namespace tpp;
#include "prep_radix_sort.h"
#include "radix_sort.h"
#include "tensor_helper.h"

REGISTER_SCOPE(remb, "remb");
REGISTER_SCOPE(gremb, "gremb");

at::Tensor emb_fwd(int alignN, std::vector<at::Tensor> inputs) {
  GlobalPass _gp(FWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
    if (inputs[1].dtype() == torch::kInt64) {
      typedef int64_t Tind;
#include "emb_fwd.h"
    } else if (inputs[1].dtype() == torch::kInt32) {
      typedef int Tind;
#include "emb_fwd.h"
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    if (inputs[1].dtype() == torch::kInt64) {
      typedef int64_t Tind;
#include "emb_fwd.h"
    } else if (inputs[1].dtype() == torch::kInt32) {
      typedef int Tind;
#include "emb_fwd.h"
    }
  }
}

at::Tensor emb_bwd(std::vector<at::Tensor> inputs) {
  GlobalPass _gp(BWD);
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
    if (inputs[2].dtype() == torch::kInt64) {
      typedef int64_t Tind;
      omp_set_num_threads(omp_get_max_threads());
#include "emb_bwd.h"
    } else if (inputs[2].dtype() == torch::kInt32) {
      typedef int Tind;
      omp_set_num_threads(omp_get_max_threads());
#include "emb_bwd.h"
    }
  } else if (inputs[0].dtype() == at::kBFloat16) {
    typedef bfloat16 T;
    if (inputs[2].dtype() == torch::kInt64) {
      typedef int64_t Tind;
      omp_set_num_threads(omp_get_max_threads());
#include "emb_bwd.h"
    } else if (inputs[2].dtype() == torch::kInt32) {
      typedef int Tind;
      omp_set_num_threads(omp_get_max_threads());
#include "emb_bwd.h"
    }
  }
}

REGISTER_SUBMODULE(_embedding, m) {
  m.def("emb_fwd", &emb_fwd, "Tpp Embedding forward");
  m.def("emb_bwd", &emb_bwd, "Tpp Embedding backward");
}

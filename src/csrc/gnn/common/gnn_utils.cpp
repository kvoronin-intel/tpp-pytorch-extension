#include <torch/extension.h>
#include <cstdlib>

#include <iostream>
#include <mutex>
#include <vector>
#include "init.h"
#include "utils.h"

typedef at::BFloat16 bfloat16;

at::Tensor gather_features(const int align, std::vector<at::Tensor> inputs) {
  if (inputs[0].dtype() == at::kFloat) {
    typedef float T;
#include "gnn_utils.h"
  } else {
    typedef bfloat16 T;
#include "gnn_utils.h"
  }
}

void affinitize_cores(const int nthreads, const int num_workers) {
#pragma omp parallel
  {
    int mytid = omp_get_thread_num();
    cpu_set_t my_set, mask;
    CPU_ZERO(&my_set);
    CPU_SET(num_workers + mytid, &my_set);

    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &mask) == -1) {
      perror("sched_getaffinity");
      assert(false);
    }
    long nproc = nthreads + num_workers;
    for (long i = 0; i < nproc; i++) {
      if (CPU_ISSET(i, &mask))
        printf("%d on core %ld\n", mytid, i);
    }
  }
}

REGISTER_SUBMODULE(_gnn_utils, m) {
  m.def("gather_features", &gather_features, "C++ Impl of feature gather");
  m.def("affinitize_cores", &affinitize_cores, "Compute thread affinization");
}

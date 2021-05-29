#include <immintrin.h>
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>

#include "init.h"
#include "utils.h"

thread_local unsigned int* rng_state = NULL;
thread_local struct drand48_data drng_state; // For non AVX512 version
void xsmm_manual_seed(unsigned int seed) {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);

    if (rng_state) {
      libxsmm_rng_destroy_extstate(rng_state);
      rng_state = NULL;
    }
    rng_state = libxsmm_rng_create_extstate(seed + tid);
    srand48_r(seed + tid, &drng_state);
  }
}

void init_libxsmm() {
  auto max_threads = omp_get_max_threads();
  PCL_ASSERT(
      max_threads <= MAX_THREADS,
      "Maximun %d threads supported, %d threads being used, please compile with increased  MAX_THREADS value\n",
      MAX_THREADS,
      max_threads);
  libxsmm_init();
  manual_seed(0);
}

REGISTER_SUBMODULE(_xsmm, m) {
  m.def("manual_seed", &xsmm_manual_seed, "Set libxsmm random seed");
  m.def("init_libxsmm", &init_libxsmm, "Initialize libxsmm");
}

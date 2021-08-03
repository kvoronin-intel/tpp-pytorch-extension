#include "init.h"
#include "timing.h"
#include "utils.h"

#ifdef _OPENMP
#pragma message "Using OpenMP"
#endif

std::vector<std::pair<std::string, submodule_init_func>> _submodule_list;

// Declared in timing.h
std::vector<Scope> _scope_list{Scope("Reserved")};
std::vector<Scope> _pass_list{Scope("OTH"),
                              Scope("FWD"),
                              Scope("BWD"),
                              Scope("UPD")};

double ifreq = 1.0 / getFreq();

PassType globalPass = OTH;
REGISTER_SCOPE(other, "other");
REGISTER_SCOPE(w_vnni, "w_vnni");
REGISTER_SCOPE(w_xpose, "w_xpose");
REGISTER_SCOPE(a_xpose, "a_xpose");
REGISTER_SCOPE(a_vnni, "a_vnni");
REGISTER_SCOPE(zero, "zero");
REGISTER_SCOPE(pad_act, "pad_act");
REGISTER_SCOPE(unpad_act, "unpad_act");

int globalScope = 0;

void reset_debug_timers() {
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    for (auto& scope : _pass_list) {
      if (scope.master_timer == 0.0)
        continue;
      for (int t = 0; t < NUM_TIMERS; t++) {
        scope.detailed_timers[tid][t] = 0.0;
      }
      scope.flops[tid][0] = 0;
    }
    for (auto& scope : _scope_list) {
      if (scope.master_timer == 0.0)
        continue;
      for (int t = 0; t < NUM_TIMERS; t++) {
        scope.detailed_timers[tid][t] = 0.0;
      }
      scope.flops[tid][0] = 0;
    }
  }
  for (auto& scope : _pass_list) {
    if (scope.master_timer == 0.0)
      continue;
    scope.master_timer = 0.0;
  }
  for (auto& scope : _scope_list) {
    if (scope.master_timer == 0.0)
      continue;
    scope.master_timer = 0.0;
  }
}

void print_debug_timers(int tid) {
  int my_rank = guess_mpi_rank();
  if (my_rank != 0)
    return;
  int max_threads = omp_get_max_threads();
  constexpr int maxlen = 10000;
  SafePrint<maxlen> printf;
  // printf("%-20s", "####");
  printf("### ##: %-11s: ", "#KEY#");
  for (int t = 0; t < LAST_TIMER; t++) {
    printf(" %7s", DebugTimerNames[t]);
  }
  printf(" %8s  %8s\n", "Total", "MTotal");
  for (int i = 0; i < max_threads; i++) {
    if (tid == -1 || tid == i) {
      auto print_scope = [&](const Scope& scope) {
        if (scope.master_timer == 0.0)
          return;
        double total = 0.0;
        printf("TID %2d: %-11s: ", i, scope.name.c_str());
        for (int t = 0; t < LAST_TIMER; t++) {
          printf(" %7.1f", scope.detailed_timers[i][t] * 1e3);
          total += scope.detailed_timers[i][t];
        }
        long t_flops = 0;
        for (int f = 0; f < max_threads; f++)
          t_flops += scope.flops[f][0];
        if (t_flops > 0.0) {
          printf(
              " %8.1f  %8.1f  %8.3f (%4.2f) %6.3f\n",
              total * 1e3,
              scope.master_timer * 1e3,
              t_flops * 1e-9,
              t_flops * 100.0 / (scope.flops[i][0] * max_threads),
              t_flops * 1e-12 / scope.detailed_timers[i][BRGEMM]);
        } else {
          printf(" %8.1f  %8.1f\n", total * 1e3, scope.master_timer * 1e3);
        }
      };
      for (auto& scope : _pass_list)
        print_scope(scope);
      for (auto& scope : _scope_list)
        print_scope(scope);
    }
  }
  printf.print();
}

static void init_submodules(pybind11::module& m) {
  for (auto& p : _submodule_list) {
    auto sm = m.def_submodule(p.first.c_str());
    auto module = py::handle(sm).cast<py::module>();
    p.second(module);
  }
}

// PYBIND11_MODULE(TORCH_MODULE_NAME, m) {
PYBIND11_MODULE(_C, m) {
  init_submodules(m);
  m.def("print_debug_timers", &print_debug_timers, "print_debug_timers");
  m.def("reset_debug_timers", &reset_debug_timers, "reset_debug_timers");
};

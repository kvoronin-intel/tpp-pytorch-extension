#include "init.h"
#include "utils.h"

#ifdef _OPENMP
#pragma message "Using OpenMP"
#endif

std::vector<std::pair<std::string, submodule_init_func>> _submodule_list;

double ifreq = 1.0 / getFreq();

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
};

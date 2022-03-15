#ifndef _BERT_TIMING_H_
#define _BERT_TIMING_H_

#include "utils.h"

enum DebugTimer {
  BRGEMM,
  XPOSE,
  DROPOUT,
  LAYER_NORM,
  SOFTMAX,
  ACT,
  BIAS,
  VNNI,
  EW_COPY,
  EW_ADD,
  EW_SCL,
  EW_RECP,
  EW_RECP_SQRT,
  EW_MUL,
  EW_ZERO,
  EW_RED,
  OPTIM,
  LAST_TIMER
};

inline const char* DebugTimerName(int t) {
  const char* names[] = {"BRGEMM",
                         "XPOSE",
                         "DROPOUT",
                         "LYR_NRM",
                         "SOFTMAX",
                         "ACT",
                         "BIAS",
                         "VNNI",
                         "COPY",
                         "ADD",
                         "SCALE",
                         "RECP",
                         "RECP_SQRT",
                         "MUL",
                         "ZERO",
                         "REDUCE",
                         "OPTIM",
                         "LAST_TIMER"};
  return names[t];
}

enum PassType { OTH, FWD, BWD, UPD };

extern PassType globalPass;
extern int globalScope;
constexpr int NUM_TIMERS = ((LAST_TIMER + 7) / 8) * 8;
extern double pass_timers[MAX_THREADS][3][NUM_TIMERS];
extern double master_pass_timers[3];
struct Scope {
  Scope(std::string const& name) : name(name) {}
  const std::string name;
  double master_timer;
  double detailed_timers[MAX_THREADS][NUM_TIMERS];
  double flops[MAX_THREADS][8];
};

// Defined in init.cpp
extern std::vector<Scope> _scope_list;
extern std::vector<Scope> _pass_list;
inline int register_scope(std::string name) {
  _scope_list.emplace_back(name);
  int idx = _scope_list.size() - 1;
  // printf("Registering %s scope @%d\n", name.c_str(), idx);
  return idx;
}

#define REGISTER_LOCAL_SCOPE(id, name) static int sc_##id = register_scope(name)
#define REGISTER_SCOPE(id, name) int sc_##id = register_scope(name)
#define USING_SCOPE(id) extern int sc_##id

class ScopedTimer {
 public:
  ScopedTimer(DebugTimer t, long f = 0) : type(t), flops(f), start(getTime()) {}
  ~ScopedTimer() {
    auto time = getTime() - start;
    int tid = omp_get_thread_num();
    auto& pass = _pass_list[globalPass];
    pass.detailed_timers[tid][type] += time;
    if (type == BRGEMM)
      pass.flops[tid][0] += flops;
    if (globalPass == 0 && tid == 0)
      pass.master_timer += time;

    auto& scope = _scope_list[globalScope];
    scope.detailed_timers[tid][type] += time;
    if (type == BRGEMM)
      scope.flops[tid][0] += flops;
    if (globalScope == 0 && tid == 0)
      scope.master_timer += time;
  }
  DebugTimer type;
  long flops;
  double start;
};

class GlobalScope {
 public:
  GlobalScope(int t) : oldScope(globalScope), start(getTime()) {
    PCL_ASSERT(t < (int)_scope_list.size(), "Invalid scope initialized");
    globalScope = t;
  }
  ~GlobalScope() {
    if (oldScope == 0) {
      // Record time only for outermost scope
      auto time = getTime() - start;
      auto& scope = _scope_list[globalScope];
      scope.master_timer += time;
    }
    globalScope = oldScope;
  }
  int oldScope;
  double start;
};

class GlobalPass {
 public:
  GlobalPass(PassType p) : oldPass(globalPass), start(getTime()) {
    globalPass = p;
  }
  ~GlobalPass() {
    if (oldPass == 0) {
      auto time = getTime() - start;
      auto& pass = _pass_list[globalPass];
      pass.master_timer += time;
    }
    globalPass = oldPass;
  }
  PassType oldPass;
  double start;
};

template <typename T, int impl = 0>
class ScopedTPP {
 public:
  ScopedTPP(T func, DebugTimer t) : func(std::move(func)), t(t) {}
  template <typename... Types>
  void operator()(Types... vars) {
    ScopedTimer _t(t);
    if (impl == 0) {
      func(vars...);
    } else if (impl == 1) {
      func.ref(vars...);
    } else {
      printf("invalid impl requested\n");
      exit(1);
    }
  }

 private:
  T func;
  DebugTimer t;
};

#if 1
// Keeping below two definitions for backward compatibility for now
#define SCOPEITGEMM SCOPEIT
#define SCOPEITGEMM2 SCOPEIT

#define SCOPEIT(f, ...) ScopedTPP<decltype(f), 0>(f, ##__VA_ARGS__)
#define SCOPEIT_REF(f, ...) ScopedTPP<decltype(f), 1>(f, ##__VA_ARGS__)
#else
#define SCOPEIT(f, t) f
#endif

#define RECORD_SCOPE(scope, ...) \
  GlobalScope gs_(sc_##scope);   \
  RECORD_FUNCTION(#scope, std::vector<c10::IValue>(__VA_ARGS__))
#endif //_BERT_TIMING_H_

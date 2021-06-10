#ifndef _PCL_UTILS_H_
#define _PCL_UTILS_H_

#include <ATen/record_function.h>
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_num_threads() 1
#define omp_get_thread_num() 0
#endif

#define MAX_THREADS 640
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
#define PCL_ASSERT(cond, x...) \
  do {                         \
    if (!(cond)) {             \
      printf(x);               \
      fflush(stdout);          \
      exit(1);                 \
    }                          \
  } while (0)

typedef at::BFloat16 bfloat16;
typedef at::Half half;

#define DECL_VLA_PTR(type, name, dims, ptr) type(*name) dims = (type(*) dims)ptr
#define DECL_VLA_PTR_PT(type, name, dims, t) \
  type(*name) dims = (type(*) dims)(t.data_ptr<type>())

extern double ifreq; // defined in init.cpp

// Defined in xsmm.cpp
extern thread_local unsigned int* rng_state;
extern thread_local struct drand48_data drng_state; // For non AVX512 version

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

inline double getFreq() {
  long long int s = rdtsc();
  sleep(1);
  long long int e = rdtsc();
  return (e - s) * 1.0;
}

inline double getTime() {
  return rdtsc() * ifreq;
}

inline int guess_mpi_rank() {
  const char* env_names[] = {
      "RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"};
  static int guessed_rank = -1;
  if (guessed_rank >= 0)
    return guessed_rank;
  for (int i = 0; i < 4; i++) {
    if (getenv(env_names[i]) != NULL) {
      int r = atoi(getenv(env_names[i]));
      if (r >= 0) {
        printf("My guessed rank = %d\n", r);
        guessed_rank = r;
        return guessed_rank;
      }
    }
  }
  guessed_rank = 0;
  return guessed_rank;
}

template <int maxlen>
class SafePrint {
 public:
  SafePrint() {}
  template <typename... Types>
  int operator()(Types... vars) {
    if (len < maxlen) {
      int l = snprintf(&buf[len], 2*maxlen - len, vars...);
      len += l;
      if (len > maxlen) {
        print();
      }
      return l;
    }
    return 0;
  }
  void print() {
    printf("%s", buf);
    len = 0;
  }

 private:
  char buf[2*maxlen];
  int len = 0;
};

#endif //_PCL_UTILS_H_

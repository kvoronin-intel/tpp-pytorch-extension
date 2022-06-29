#include <omp.h>
#include <algorithm>
#include <iostream>
#include <vector>

#include <stdio.h>

#include <unistd.h>
#include <cstdlib>
#include "radix_sort.h"

using namespace std;

#define COALESCING_PREPROCESSING_VERBOSE 0

class CoalescingPreprocessingThreadState {
 public:
  CoalescingPreprocessingThreadState() : numUniqueIndices(0) {}
  uint32_t numUniqueIndices;
  uint32_t dummy[15];
};

template <typename Tind, typename Tidx>
void init_scratch(
    std::pair<Tidx, Tidx>* scratch,
    Tind* indices,
    long numIndices) {
#pragma omp parallel for
  for (long i = 0; i < numIndices; i++) {
    scratch[i].first = indices[i];
    scratch[i].second = i;
  }
}

template <typename Tidx>
std::tuple<at::Tensor, at::Tensor, at::Tensor> coalescing_preprocessing(
    long indicesCount,
    long maxIndexValue,
    std::pair<Tidx, Tidx>* scratch) {
  unsigned long long t3 = __rdtsc();
  auto sortedIndexWithOutputRowPair = radix_sort_parallel(
      scratch, scratch + indicesCount, indicesCount, maxIndexValue);
  unsigned long long t4 = __rdtsc();

  // cout << "Sort: " << (t4-t3)*1e3/freq<< " ms" << endl;

  int maxThreads = omp_get_max_threads();

  unsigned long long t5 = __rdtsc();

  CoalescingPreprocessingThreadState preprocessingThreadState[maxThreads];

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    preprocessingThreadState[tid].numUniqueIndices = 0;
#pragma omp for schedule(static)
    for (int i = 1; i < indicesCount; i++) {
      if (sortedIndexWithOutputRowPair[i].first !=
          sortedIndexWithOutputRowPair[i - 1].first) {
        preprocessingThreadState[tid].numUniqueIndices++;
      }
    }
  }
  preprocessingThreadState[0].numUniqueIndices += 1;
  for (int i = 1; i < maxThreads; i++)
    preprocessingThreadState[i].numUniqueIndices +=
        preprocessingThreadState[i - 1].numUniqueIndices;

  long uniqueIndicesCount =
      preprocessingThreadState[maxThreads - 1].numUniqueIndices;

  at::Tensor t_uniqueIndices = at::empty({uniqueIndicesCount}, at::kLong);
  at::Tensor t_outputRowOffsets = at::empty({uniqueIndicesCount + 1}, at::kInt);
  at::Tensor t_outputRows = at::empty({indicesCount}, at::kInt);

  auto uniqueIndices = t_uniqueIndices.data_ptr<long>();
  auto outputRowOffsets = t_outputRowOffsets.data_ptr<int>();
  auto outputRows = t_outputRows.data_ptr<int>();

  uniqueIndices[0] = sortedIndexWithOutputRowPair[0].first;
  outputRowOffsets[0] = 0;
  outputRows[0] = sortedIndexWithOutputRowPair[0].second;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    auto* tstart =
        (tid == 0 ? uniqueIndices + 1
                  : uniqueIndices +
                 preprocessingThreadState[tid - 1].numUniqueIndices);
    auto* t_offs =
        (tid == 0 ? outputRowOffsets + 1
                  : outputRowOffsets +
                 preprocessingThreadState[tid - 1].numUniqueIndices);
#pragma omp for schedule(static)
    for (int i = 1; i < indicesCount; i++) {
      outputRows[i] = sortedIndexWithOutputRowPair[i].second;
      if (sortedIndexWithOutputRowPair[i].first !=
          sortedIndexWithOutputRowPair[i - 1].first) {
        *tstart = sortedIndexWithOutputRowPair[i].first;
        *t_offs = i;
        tstart++;
        t_offs++;
      }
    }
  }
  outputRowOffsets[uniqueIndicesCount] = indicesCount;

  return std::make_tuple(t_uniqueIndices, t_outputRowOffsets, t_outputRows);
}

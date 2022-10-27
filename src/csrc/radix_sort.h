#ifndef _TPP_RADIX_SORT_
#define _TPP_RADIX_SORT_

// #include "sort_headers.h"
#include <omp.h>
#include <limits>
#include <utility>

#ifndef BKT_BITS
#define BKT_BITS 11
#endif

template <typename Tind>
using Key_Value_Pair = std::pair<Tind, Tind>;

template <typename Tind>
Key_Value_Pair<Tind>* radix_sort_parallel(
    Key_Value_Pair<Tind>* inp_buf,
    Key_Value_Pair<Tind>* tmp_buf,
    int64_t elements_count,
    int64_t max_value) {
  constexpr int bkt_bits = BKT_BITS;
  constexpr int nbkts = (1 << bkt_bits);
  constexpr int bkt_mask = (nbkts - 1);

  int maxthreads = omp_get_max_threads();
  int histogram[nbkts * maxthreads], histogram_ps[nbkts * maxthreads + 1];
  if (max_value == 0)
    return inp_buf;
  int num_bits = 64;
  if (sizeof(Tind) == 8 && max_value > std::numeric_limits<int>::max()) {
    num_bits = sizeof(Tind) * 8 - __builtin_clzll(max_value);
  } else {
    num_bits = 32 - __builtin_clz((unsigned int)max_value);
  }

  int num_passes = (num_bits + bkt_bits - 1) / bkt_bits;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    int* local_histogram = &histogram[nbkts * tid];
    int* local_histogram_ps = &histogram_ps[nbkts * tid];
    int elements_count_4 = elements_count / 4 * 4;
    Key_Value_Pair<Tind>* input = inp_buf;
    Key_Value_Pair<Tind>* output = tmp_buf;

    for (unsigned int pass = 0; pass < (unsigned int)num_passes; pass++) {
#ifdef DEBUG
      unsigned long long t1 = __rdtsc();
#endif
      // Step 1: compute histogram
      // Reset histogram
      for (int i = 0; i < nbkts; i++)
        local_histogram[i] = 0;

#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        Tind val_1 = input[i].first;
        Tind val_2 = input[i + 1].first;
        Tind val_3 = input[i + 2].first;
        Tind val_4 = input[i + 3].first;

        local_histogram[(val_1 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_2 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_3 >> (pass * bkt_bits)) & bkt_mask]++;
        local_histogram[(val_4 >> (pass * bkt_bits)) & bkt_mask]++;
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          Tind val = input[i].first;
          local_histogram[(val >> (pass * bkt_bits)) & bkt_mask]++;
        }
      }
#pragma omp barrier
#ifdef DEBUG
      unsigned long long t11 = __rdtsc();
#endif
      // Step 2: prefix sum
      if (tid == 0) {
        int sum = 0, prev_sum = 0;
        for (int bins = 0; bins < nbkts; bins++)
          for (int t = 0; t < nthreads; t++) {
            sum += histogram[t * nbkts + bins];
            histogram_ps[t * nbkts + bins] = prev_sum;
            prev_sum = sum;
          }
        histogram_ps[nbkts * nthreads] = prev_sum;
        if (prev_sum != elements_count) {
          printf("Error1!\n");
          exit(123);
        }
      }
#pragma omp barrier
#ifdef DEBUG
      unsigned long long t12 = __rdtsc();
#endif

      // Step 3: scatter
#pragma omp for schedule(static)
      for (int64_t i = 0; i < elements_count_4; i += 4) {
        Tind val_1 = input[i].first;
        Tind val_2 = input[i + 1].first;
        Tind val_3 = input[i + 2].first;
        Tind val_4 = input[i + 3].first;
        Tind bin_1 = (val_1 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_2 = (val_2 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_3 = (val_3 >> (pass * bkt_bits)) & bkt_mask;
        Tind bin_4 = (val_4 >> (pass * bkt_bits)) & bkt_mask;
        int pos;
        pos = local_histogram_ps[bin_1]++;
        output[pos] = input[i];
        pos = local_histogram_ps[bin_2]++;
        output[pos] = input[i + 1];
        pos = local_histogram_ps[bin_3]++;
        output[pos] = input[i + 2];
        pos = local_histogram_ps[bin_4]++;
        output[pos] = input[i + 3];
      }
      if (tid == (nthreads - 1)) {
        for (int64_t i = elements_count_4; i < elements_count; i++) {
          Tind val = input[i].first;
          int pos = local_histogram_ps[(val >> (pass * bkt_bits)) & bkt_mask]++;
          output[pos] = input[i];
        }
      }

      Key_Value_Pair<Tind>* temp = input;
      input = output;
      output = temp;
#pragma omp barrier
#ifdef DEBUG
      unsigned long long t2 = __rdtsc();
#endif
#ifdef DEBUG_TIME
      if (tid == 0)
        printf(
            "pass = %d  time = %8lld  %8lld  %8lld %8lld\n",
            pass,
            (t2 - t1) / 1000,
            (t11 - t1) / 1000,
            (t12 - t11) / 1000,
            (t2 - t12) / 1000);
#endif
    }
  }
  return (num_passes % 2 == 0 ? inp_buf : tmp_buf);
}

#endif

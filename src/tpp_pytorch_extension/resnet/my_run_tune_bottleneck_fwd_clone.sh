#!/bin/bash

export LD_PRELOAD=/usr/lib64/libstdc++.so.6:$LD_PRELOAD
export LIBXSMM_ROOT=/home/kvoronin/work/libxsmm/libxsmm

export CC=gcc

#. /swtools/glibc/glibc_vars.sh

python --version
gcc --version
#exit

: '
OMP_NUM_THREADS=28 KMP_AFFINITY=compact,granularity=fine,1,0 IOMP_PREFIX=/swtools/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64 \
OMP_NUM_THREADS=28 srun python test_conv.py
'
  python -u bottleneck_tuning_fwd_driver.py $@ #--use-bf16-opt
#gdb --args  python -u bottleneck_tuning_fwd_driver.py $@ #--use-bf16-opt
exit

set -e
set -o pipefail

export OMP_NUM_THREADS=28
# Functional
#gdb --args python test_bottleneck_ext.py $@ #--use-bf16-opt
#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
#for i in 1 2 3 4 5 6 7 8 9 10
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
  echo "run $i"
  python -u bottleneck_tuning_fwd_driver.py $@ #--use-bf16-opt
done
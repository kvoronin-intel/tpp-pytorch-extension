#!/bin/bash

# Source a python environment with torch

export LD_PRELOAD=/usr/lib64/libstdc++.so.6:$LD_PRELOAD
export LIBXSMM_ROOT=/home/kvoronin/work/libxsmm/libxsmm
export LIBXSMM_TARGET=SPR

export CC=gcc

#. /swtools/glibc/glibc_vars.sh

python --version
gcc --version
#exit

export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=/swtools/intel/oneapi/compiler/latest/linux/compiler/lib/intel64/libiomp5.so:$LD_PRELOAD

export LD_LIBRARY_PATH=/home/kvoronin/work/libxsmm/libxsmm/lib:$LD_LIBRARY_PATH

#export OMP_NUM_THREADS=28
# Functional
#gdb --args python test_conv_ext.py
#gdb --args python test_conv_ext.py --test-module ext_tpp #--use-bf16-opt
#python test_conv_ext.py --test-module ext_tpp #--use-bf16-opt
#gdb --args python test_conv_ext.py $@ #--use-bf16-opt
#            python -u test_conv_ext.py $@ #--use-bf16-opt
python -u test_conv_ext.py  $@

#python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 16 14 14 1024 2048 2 1  --tuning-params 1 1  1 0 1  0 0 0  1 8 2 --tuning-string A{R:8}C{C:2}dbef --niters 1 --niters-warmup 0
# fails with free()
#gdb --args python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 64 --basic-sizes 56 56 56 64 64 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1 --niters-warmup 0 

exit

set -e
set -o pipefail

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
  echo "run $i"
  python -u test_conv_ext.py $@ #--use-bf16-opt
done

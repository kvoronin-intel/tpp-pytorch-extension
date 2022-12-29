#!/bin/bash

export LD_PRELOAD=/usr/lib64/libstdc++.so.6:$LD_PRELOAD
export LIBXSMM_ROOT=/home/kvoronin/work/libxsmm/libxsmm
export LIBXSMM_TARGET=SPR

export CC=gcc

python --version
gcc --version
#exit

# Functional
#python test_bn.py #--use-bf16-opt
#python test_bn.py --use-bf16-opt
#gdb --args python test_bn.py
#gdb --args python test_bn_ext.py --test-module ext_tpp #--use-bf16-opt
#gdb --args  python test_bn_ext.py $@
            python -u test_bn_ext.py $@
#python test_conv.py --use-bf16-opt --with-perf
exit

set -e
set -o pipefail

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
do
  echo "run $i"
  python -u test_bn_ext.py $@
done

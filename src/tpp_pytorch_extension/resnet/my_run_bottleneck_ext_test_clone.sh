#!/bin/bash

# Source a python environment with torch

#export PATH=/nfs/site/proj/mkl/mirror/NN/tools/gcc_installed/el7/gnu_8.3.0/bin:/nfs/site/proj/mkl/mirror/NN/tools/ninja/lnx:$PATH
#export LIBXSMM_ROOT=/nfs/site/proj/mkl/project/kvoronin/various_libs/libxsmm/libxsmm

#export PATH=/swtools/gcc/11.2.0/bin:$PATH
export PATH=/swtools/gcc/8.3.0/bin:$PATH
export LD_LIBRARY_PATH=/swtools/gcc/8.3.0/lib64:$LD_LIBRARY_PATH
#/swtools/ninja/latest/bin
export LIBXSMM_ROOT=/nfs_home/kvoronin/work/libxsmm_gcc3/libxsmm

export CC=gcc

#. /swtools/glibc/glibc_vars.sh

python --version
gcc --version
#exit

#export OMP_NUM_THREADS=28
#export GOMP_CPU_AFFINITY="0-27"
#gdb --args python test_conv.py
#export LD_PRELOAD=/swtools/jemalloc/lib/libjemalloc.so

#export CONDA_PREFIX=/data/nfs_home/kvoronin/work/pcl_pt/cnn/env_yml_check/cnn/.cenv
#export CONDA_PREFIX=/data/nfs_home/kvoronin/anaconda/envs/mypython376env
#export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so
#export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so
#export LD_PRELOAD=/usr/lib64/libtcmalloc.so.4
#export IOMP_PREFIX=/swtools/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64
#python test_conv.py

#source ../BatchNorm/env_clone.sh
: '
OMP_NUM_THREADS=28 KMP_AFFINITY=compact,granularity=fine,1,0 IOMP_PREFIX=/swtools/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64 \
OMP_NUM_THREADS=28 srun python test_conv.py
'

set -e
set -o pipefail

export OMP_NUM_THREADS=28
# Functional
#gdb --args python test_bottleneck_ext.py $@ #--use-bf16-opt
#for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30
for i in 1 2 3 4 5 6 7 8 9 10
do
  echo "run $i"
  python -u test_bottleneck_ext.py $@ #--use-bf16-opt
done
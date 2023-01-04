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

export GOMP_CPU_AFFINITY=0-55
export OMP_NUM_THREADS=56
#export GOMP_CPU_AFFINITY=0-27
#export OMP_NUM_THREADS=28

#GOMP_CPU_AFFINITY=4-15 taskset -c 4-15 python -c "import psutil, torch; print(psutil.Process().cpu_affinity())"
#exit
# locate libtorch.so and getting libtorch.so from .../ported_envs/...
#ldd libtorch.so # shows that it calls libgomp
#exit

#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes 56 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters 1 --niters-warmup 1

niters=1000
niters_warmup=100
#extra=""
extra="--scale-only"

#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 64 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp                --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 64 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp                --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 1024 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp                --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 1024 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#exit
#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp                --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 

#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 1024 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#exit

#python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 512 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#python -u test_bn_ext.py --test-module ext_tpp                --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 512 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
#exit

# Full
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 64 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 64 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 64 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 64 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 64 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 64 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 256 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 56 56 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 128 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 512 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 128 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 128 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 128 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 28 28 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 1024 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 14 14 512 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 512 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 2048 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 2048 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 512 1 0 0 0 1 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 512 1 0 0 1 0 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 2048 1 1 0 0 0 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 512 1 0 0 0 1 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 512 1 0 0 1 0 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $OMP_NUM_THREADS 7 7 2048 1 1 0 0 0 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra


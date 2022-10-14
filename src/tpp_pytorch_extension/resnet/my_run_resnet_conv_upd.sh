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

 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 7 7 512 2048 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefcdb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 7 7 512 512 1 3 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Adbcef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 7 7 2048 512 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefcdb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 7 7 512 2048 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefcdb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 7 7 512 512 1 3 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Adbcef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 7 7 2048 512 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefcdb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 7 7 512 2048 1 1 --tuning-params 1 1  0 0 1  1 0 0  0 1 1 --tuning-string Aefdbc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 512 512 2 3 --tuning-params 1 0  1 0 1  1 0 0  0 1 1 --tuning-string abEfdc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 1024 512 1 1 --tuning-params 1 1  0 0 1  1 0 0  0 1 1 --tuning-string Aefdbc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 1024 2048 2 1 --tuning-params 1 1  1 0 1  1 0 0  0 1 1 --tuning-string Aefdbc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 1024 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 256 1 3 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Abcdef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 1024 256 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 1024 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 256 1 3 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Abcdef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 1024 256 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 1024 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 256 1 3 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Abcdef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 1024 256 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 1024 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 256 1 3 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Abcdef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 1024 256 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 1024 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 256 1 3 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Abcdef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 1024 256 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefdcb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 14 14 256 1024 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefbcd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 256 256 2 3 --tuning-params 1 0  1 1 1  0 0 0  0 1 1 --tuning-string abEfcd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 512 256 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefbcd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 512 1024 2 1 --tuning-params 1 1  1 1 1  0 0 0  0 1 1 --tuning-string Aefbcd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 128 512 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 128 128 1 3 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Acdbef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 512 128 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 128 512 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 128 128 1 3 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Acdbef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 512 128 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 128 512 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 128 128 1 3 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Acdbef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 512 128 1 1 --tuning-params 1 1  0 1 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 28 28 128 512 1 1 --tuning-params 1 1  0 0 1  1 0 0  0 1 1 --tuning-string Aefcdb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 128 128 2 3 --tuning-params 1 0  1 0 1  1 0 0  0 1 1 --tuning-string abEFdc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 256 128 1 1 --tuning-params 1 1  0 0 1  1 0 0  0 1 1 --tuning-string Aefcdb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 256 512 2 1 --tuning-params 1 1  1 0 1  1 0 0  0 1 1 --tuning-string Aefcdb --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 64 256 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefdbc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 64 64 1 3 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Acbdef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 256 64 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefdbc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 64 256 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefdbc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 64 64 1 3 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Acbdef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 256 64 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefdbc --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 64 256 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 64 64 1 3 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Acdbef --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 64 64 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
 python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --with-bwd --perf-bwd-w --bc 32 --bk 32 --basic-sizes 28 56 56 64 256 1 1 --tuning-params 1 1  0 0 1  0 0 0  0 1 1 --tuning-string Aefcbd --niters 1000 --niters-warmup 100 
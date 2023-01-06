#!/bin/bash

export OMP_NUM_THREADS=56

niters=1000
niters_warmup=100
extra=""
#extra="--scale-only"

mb=56

# Full
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 64 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 64 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 256 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 256 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 64 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 64 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 256 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 64 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 64 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 256 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 56 56 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 128 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 512 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 128 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 128 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 128 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 128 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 512 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 28 28 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 1024 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 256 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 1024 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 14 14 512 1 0 0 0 1 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 512 1 0 0 1 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 2048 0 0 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 2048 1 1 0 0 0 --tuning-string-ncp Ab --tuning-string-cp a --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 512 1 0 0 0 1 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 512 1 0 0 1 0 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 2048 1 1 0 0 0 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 512 1 0 0 0 1 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 512 1 0 0 1 0 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_bn_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --basic-sizes $mb 7 7 2048 1 1 0 0 0 --tuning-string-ncp AB --tuning-string-cp A --niters $niters --niters-warmup $niters_warmup $extra


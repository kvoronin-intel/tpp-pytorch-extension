#!/bin/bash

export OMP_NUM_THREADS=56

echo "Warning: The tuning paramaters used for tests below can be out-dated. To update them, one needs to dump the strings from one resnet training iteration (with VERBOSE)"

niters=1000
niters_warmup=100
extra=""
#extra="--preallocated-output"

mb=56

python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 64 64 1 1 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 64 64 1 3 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 64 256 1 1 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 64 256 1 1 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 256 64 1 1 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 64 64 1 3 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 64 256 1 1 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 256 64 1 1 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 64 64 1 3 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 64 256 1 1 --tuning-params 4 1 1 1 1 0 --tuning-string Afgbdced --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 256 128 1 1 --tuning-params 1 1 1 1 1 0 --tuning-string Afgbedc --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 128 128 2 3 --tuning-params 1 1 1 1 1 0 --tuning-string Afgbedc --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 56 56 256 512 2 1 --tuning-params 1 1 1 1 1 0 --tuning-string Afgbedc --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 128 512 1 1 --tuning-params 1 1 1 1 1 0 --tuning-string Afgbedc --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 512 128 1 1 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 128 128 1 3 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 128 512 1 1 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 512 128 1 1 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 128 128 1 3 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 128 512 1 1 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 512 128 1 1 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 128 128 1 3 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 128 512 1 1 --tuning-params 7 1 1 1 1 0 --tuning-string Afgbdecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 512 256 1 1 --tuning-params 1 1 1 1 1 0 --tuning-string Afgbcdce --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 256 256 2 3 --tuning-params 1 1 1 1 1 0 --tuning-string Afgbcdce --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 28 28 512 1024 2 1 --tuning-params 1 1 1 8 2 1 --tuning-string Afgbcdce --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 1024 1 1 --tuning-params 1 1 1 8 2 0 --tuning-string Afgbcdce --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 1024 256 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 256 1 3 --tuning-params 1 1 1 1 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 1024 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 1024 256 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 256 1 3 --tuning-params 1 1 1 1 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 1024 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 1024 256 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 256 1 3 --tuning-params 1 1 1 1 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 1024 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 1024 256 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 256 1 3 --tuning-params 1 1 1 1 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 1024 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 1024 256 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 256 1 3 --tuning-params 1 1 1 1 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 256 1024 1 1 --tuning-params 1 1 1 2 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 1024 512 1 1 --tuning-params 1 1 1 8 2 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 512 512 2 3 --tuning-params 1 1 1 4 1 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 14 14 1024 2048 2 1 --tuning-params 1 1 1 4 7 1 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 7 7 512 2048 1 1 --tuning-params 1 1 1 8 7 0 --tuning-string Afgbcecd --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 7 7 2048 512 1 1 --tuning-params 1 1 1 1 7 0 --tuning-string ACfgbdec --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 7 7 512 512 1 3 --tuning-params 1 1 1 1 7 0 --tuning-string ACfgbdec --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 7 7 512 2048 1 1 --tuning-params 1 1 1 1 7 0 --tuning-string ACfgbdec --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 7 7 2048 512 1 1 --tuning-params 1 1 1 1 7 0 --tuning-string ACfgbdec --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 7 7 512 512 1 3 --tuning-params 1 1 1 1 7 0 --tuning-string ACfgbdec --niters $niters --niters-warmup $niters_warmup $extra 
python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --perf-fwd --bc 32 --bk 32 --basic-sizes $mb 7 7 512 2048 1 1 --tuning-params 1 1 1 1 7 0 --tuning-string ACfgbdec --niters $niters --niters-warmup $niters_warmup $extra 
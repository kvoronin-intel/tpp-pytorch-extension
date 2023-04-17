#!/bin/bash

echo "Usage: ./run_nompi_template.sh <create_start_checkpoint|run_bf16_training> <batch_size>"

export LIBXSMM_ROOT=#<path_to_libxsmm>
export LD_LIBRARY_PATH=#<path_to_libxsmm>/lib:<path_to_libxsmm_dnn>/lib:$LD_LIBRARY_PATH

if [[ "$1" == "create_start_checkpoint" ]]; then
    opts=" --pad-input-for-bf16-ref \
            --use-ref-conv --use-ref-pool --use-ref-bn --use-ref-fc "
elif [[ "$1" == "run_bf16_training" ]]; then
    opts="  --use-bottleneck-tpp \
            --use-ext-bottleneck \
            --use-hardcoded-tunings \
            --channel-block-size 32 \
            --use-phys3x3-padding \
            --use-optim splitsgd_bf16fb \
            --use-bf16 \
            --use-ref-fc \
            --use-ext-optim \
            --use-new-conv2d "
else
    echo "Wrong input argument, \$1 = $1 but supported options are `create_start_checkpoint` and `run_bf16_training`"
    exit
fi

batch_size=$2

OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:<path_to_libiomp5.so>:$LD_PRELOAD \
  python -u main.py /home/imagenet -b $batch_size --lr 0.1 --workers 0  \
    --seed 0 \
      --synthetic --iterations 100 --epochs 1 --perf-only \
        $opts

exit



#        --use-ref-fc \

#        --use-ext-optim \

#        --use-new-conv2d \

#        --validate-fwd \

#        --use-ext-bottleneck \
#        --use-hardcoded-tunings \

#        --use-optim sgd_bf16 \
#        --use-optim sgd_bf16fb \
#        --use-optim splitsgd_bf16fb \

#        --use-optim sgd_fb \


#        --use-bf16 \

#        --enable-profiling \

#        \
#              --use-ref-conv --use-ref-pool --use-ref-bn --use-ref-fc


#        --channel-block-size 32 \

#        --channel-block-size 64 \

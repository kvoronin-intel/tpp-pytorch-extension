#!/bin/bash

NUMA_ARGS=""
if command -v numactl >& /dev/null ; then
if [ "x$MPI_LOCALRANKID" != "x" ] ; then
  REAL_NUM_NUMA_NODES=`lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
  PPNUMA=$(( MPI_LOCALNRANKS / REAL_NUM_NUMA_NODES ))
  if [ $PPNUMA -eq 0 ] ; then 
    if [ "x$SINGLE_SOCKET_ONLY" == "x1" ] ; then 
      NUMA_ARGS="numactl -m 0 "
    fi
  else
    NUMARANK=$(( MPI_LOCALRANKID / PPNUMA ))
    NUMA_ARGS="$NUMA_ARGS $GDB_ARGS "
  fi
  NUM_RANKS=$PMI_SIZE
else
  NUMA_ARGS="numactl -m 0 "
  NUM_RANKS=1
fi
fi

if [ "x$1" == "x-gdb" ] ; then
GDB_ARGS="gdb --args "
shift
else
GDB_ARGS=""
fi

# set dataset
if test -z $dataset || ! test -d $dataset ; then
  if test -d ./SQUAD1 ; then
    dataset=./SQUAD1
  else
    echo "Unable to find SQUAD dataset path"
    exit 1
  fi
fi

$NUMA_ARGS $GDB_ARGS python -u run_squad.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $dataset/train-v1.1.json \
  --predict_file $dataset/dev-v1.1.json \
  --per_gpu_train_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  $@


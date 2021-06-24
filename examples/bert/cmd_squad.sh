#!/bin/sh

export SQUAD_DIR=/nfs_home/ddkalamk/bert/SQUAD1

if [ "x$MPI_LOCALRANKID" != "x" ] ; then
  NUMARANK=$MPI_LOCALRANKID
else
  NUMARANK=0
fi
if [ "x$1" == "x-gdb" ] ; then
GDB_ARGS="gdb --args "
shift
else
GDB_ARGS=""
fi

numactl -m ${NUMARANK} $GDB_ARGS python -u run_squad.py \
  --model_type bert \
  --model_name_or_path bert-large-uncased-whole-word-masking \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 24 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/ \
  $@


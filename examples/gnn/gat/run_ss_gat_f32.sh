



#!/bin/bash

#dl=8 #12
#export CCL_WORKER_COUNT=1
#bsz=$2
dl=2
for bsz in 1024 #10240
 do
  if [ "$1" == "undir" ]
  then
      echo "batch size = "$bsz
      mdir_ss=$1"_model_rank1_f32_optmlp_"$bsz"_dl_"$dl
      mkdir $mdir_ss
      ./run_gat_all.sh 38 $dl $mdir_ss $bsz |& tee $mdir_ss/$1"_log_ss_f32_papers_icx_gat"
  fi
done


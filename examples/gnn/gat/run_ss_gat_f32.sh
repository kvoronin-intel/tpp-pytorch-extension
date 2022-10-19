



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
      #./run_dist.sh -n $ranks -ppn 2 -f ~/hostfile -dl 4 ./run_f32_unidr.sh 4 $mdir |& tee $1"_log_"$ranks"_f32"
      ./run_gat.sh 38 $dl $mdir_ss $bsz |& tee $mdir_ss/$1"_log_ss_f32_papers_icx_gat"
      #./run_dist.sh -n $ranks -ppn 2  -dl $dl ./run_f32_undir_gat.sh 0 $dl $mdir_0c |& tee $mdir_0c/$1"_log_0c_"$ranks"_f32""_papers_icx_gat"
    #done
  fi
done


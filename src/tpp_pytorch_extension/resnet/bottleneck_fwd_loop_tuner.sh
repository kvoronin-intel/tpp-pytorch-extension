#! /bin/bash

source /swtools/gcc/latest/gcc_vars.sh

export OMP_NUM_THREADS=24
export GOMP_CPU_AFFINITY="0-23"

loop_names=()
loop_specs=()

#For now we don't block additionally K dim, handled by BF of BR dimension
#for i in 0 1; do
#  for j in 0 1; do
for i in 0; do
  for j in 0; do
    loop_names+=("conv1")
    #loop_specs+=("a_0_K,b_0_K,c_${i}_M,d_${j}_N,e_0_N,f_0_K,g_0_K")
    loop_specs+=("A_0_M,b_0_K,c_${i}_M,d_0_K,e_0_K,f_0_K,g_0_K")
  done
done

for i in ${!loop_specs[@]}; do
  ./loop_permute_generator "${loop_names[$i]}" "${loop_specs[$i]}"
done

cat *bench_configs.txt > uber_config.txt
awk '!seen[$0]++' uber_config.txt > tuner_config.txt
rm uber_config.txt

loopArray=()
nLoops=0

while IFS= read -r line || [[ "$line" ]]; do
  loopArray+=("$line")
  let "nLoops+=1"
done < tuner_config.txt

nBottlenecks=8

HBFS=("1")
KBFS=("1")

export OMP_NUM_THREADS=24
export GOMP_CPU_AFFINITY="0-23"
unset LIBXSMM_VERBOSE

for (( l = 0 ; l < $nBottlenecks ; l++)); do
  echo "l = $l"
  N=${OMP_NUM_THREADS}
  bc=32
  bk=32
  niters=20
  H=56
  W=56
  C=64
  K=64
  str=1
  CBFS=("1" "2")
  WBFS=("1" "2" "4")
  downsample="True"
  expansion=4
  eps="1e-7"

  if [ $l -eq 1 ]; then
    H=56
    W=56
    C=256
    K=64
    str=1
    CBFS=("1" "2")
    WBFS=("1" "2")
    downsample="False"
  fi

  if [ $l -eq 2 ]; then
    H=56
    W=56
    C=256
    K=128
    str=2
    CBFS=("1" "2")
    WBFS=("1" "2")
    downsample="True"
  fi

  if [ $l -eq 3 ]; then
    H=28
    W=28
    C=512
    K=128
    str=1
    CBFS=("1" "2")
    WBFS=("1" "2")
    downsample="False"
  fi

  if [ $l -eq 4 ]; then
    H=28
    W=28
    C=512
    K=256
    str=2
    CBFS=("1" "2" "8")
    WBFS=("1" "2")
    downsample="True"
  fi

  if [ $l -eq 5 ]; then
    H=14
    W=14
    C=1024
    K=256
    str=1
    CBFS=("1" "2" "8")
    WBFS=("1" "2")
    downsample="False"
  fi

  if [ $l -eq 6 ]; then
    H=14
    W=14
    C=1024
    K=512
    str=2
    CBFS=("1" "2" "8")
    WBFS=("1" "2")
    downsample="True"
  fi

  if [ $l -eq 7 ]; then
    H=7
    W=7
    C=2048
    K=512
    str=1
    CBFS=("1" "2" "4")
    WBFS=("1" "2")
    downsample="False"
  fi

  benchmark_out_name="${H}_${W}_${C}_${K}_${str}_${downsample}_clx_btlnk_bench_results"
  echo "Tuning bottlenecks..." > ${benchmark_out_name}
  common_runline="python -u tune_bottleneck_fwd_driver.py --test-module ext_bottleneck --ref-module pt_native --use-physical-3x3-padding --use-bf16-opt --with-validation "
  #benchmark_out_name="clx_btlnk_bench_results"
  #echo "Tuning bottlenecks..." >> ${benchmark_out_name}

  # uncomment for running without validation
  #common_runline="python -u tune_bottleneck_fwd_driver.py --test-module ext_bottleneck --ref-module pt_native --use-physical-3x3-padding --use-bf16-opt"

  tmp_input_file="${benchmark_out_name}_tmp_input_file"
  echo "${N} ${H} ${W} ${C} ${K} ${str} ${expansion} ${downsample} ${eps}" > $tmp_input_file

  nLoops=1 # while developing the script
  for (( j = 0 ; j < $nLoops ; j++)); do
    line=${loopArray[$j]}
    echo ${line}
    lowerline=$(echo ${line} | tr '[:upper:]' '[:lower:]')
    echo ${lowerline}
    KBFcount=$( echo ${lowerline} | tr -d -c 'c' | awk '{ print length; }' )
    echo "C count is ${KBFcount}"
    HBFcount=$( echo ${lowerline} | tr -d -c 'd' | awk '{ print length; }' )
    echo "D count is ${HBFcount}"

    h_in_gemm=1
    echo "bc = ${bc}, bk = ${bk}"
    
    #if [ $KBFcount -eq 2 ]; then
    #  if [ $HBFcount -eq 2 ]; then
        for cb in "${CBFS[@]}"; do
          for kb in "${KBFS[@]}"; do
            for wb in "${WBFS[@]}"; do
              for hb in "${HBFS[@]}"; do
                for fuse_stats in 1 0 ; do
                  block_sizes_line=" --block-sizes ${bc} ${bc} ${bk} ${bk} "
                  hw_blocks_subline="${hb} ${wb} ${hb} ${wb} ${hb} ${wb} ${hb} ${wb}"
                  ck_blocks_subline="${cb} ${kb} ${cb} ${kb} ${cb} ${kb} ${cb} ${kb}"
                  tuning_params_line=" --tuning-params ${hw_blocks_subline} ${ck_blocks_subline} ${h_in_gemm} ${h_in_gemm} ${h_in_gemm} ${h_in_gemm} ${fuse_stats}"
                  tuning_strings_line=" --tuning-strings ${line} ${line} ${line} ${line}"
                  #set -x
                  ${common_runline} ${block_sizes_line} ${tuning_params_line} --test-data-file $tmp_input_file ${tuning_strings_line} --niters ${niters} >> ${benchmark_out_name}
                  #set +x
                  #echo "Exiting"
                  #exit
                  break 5 # for developing the script
                done
              done
            done
          done
        done
    #  fi
    #fi

    : '
    if [ $KBFcount -eq 2 ]; then
      if [ $HBFcount -eq 2 ]; then
        for cb in "${CBFS[@]}"; do
          for kb in "${KBFS[@]}"; do
            for wb in "${WBFS[@]}"; do
              for hb in "${HBFS[@]}"; do
                export OMP_NUM_THREADS=24     
                export GOMP_CPU_AFFINITY="0-23"
                unset LIBXSMM_VERBOSE
                ${common_runline} ${line} ${N} ${H} ${W} ${C} ${K} ${str} ${str} ${bc} ${bk} ${hb} ${wb} ${cb} ${kb} ${niters}  >> ${benchmark_out_name}
              done
            done
          done
        done
      fi
    fi

    if [ $KBFcount -eq 2 ]; then
      if [ $HBFcount -eq 1 ]; then
        for cb in "${CBFS[@]}"; do
          for kb in "${KBFS[@]}"; do
            for wb in "${WBFS[@]}"; do
              hb=1
              export OMP_NUM_THREADS=24     
              export GOMP_CPU_AFFINITY="0-23"
              unset LIBXSMM_VERBOSE
              ./conv_fwd ${line} ${N} ${H} ${W} ${C} ${K} ${str} ${str} ${bc} ${bk} ${hb} ${wb} ${cb} ${kb} ${niters}  >> ${benchmark_out_name}
            done
          done
        done
      fi
    fi

    if [ $KBFcount -eq 1 ]; then
      if [ $HBFcount -eq 2 ]; then
        for cb in "${CBFS[@]}"; do
          for wb in "${WBFS[@]}"; do
            for hb in "${HBFS[@]}"; do
              kb=1
              export OMP_NUM_THREADS=24     
              export GOMP_CPU_AFFINITY="0-23"
              unset LIBXSMM_VERBOSE
              ./conv_fwd ${line} ${N} ${H} ${W} ${C} ${K} ${str} ${str} ${bc} ${bk} ${hb} ${wb} ${cb} ${kb} ${niters}  >> ${benchmark_out_name}
            done
          done
        done
      fi
    fi

    if [ $KBFcount -eq 1 ]; then
      if [ $HBFcount -eq 1 ]; then
        for cb in "${CBFS[@]}"; do
          for wb in "${WBFS[@]}"; do
            kb=1
            hb=1
            export OMP_NUM_THREADS=24     
            export GOMP_CPU_AFFINITY="0-23"
            unset LIBXSMM_VERBOSE
            ./conv_fwd ${line} ${N} ${H} ${W} ${C} ${K} ${str} ${str} ${bc} ${bk} ${hb} ${wb} ${cb} ${kb} ${niters}  >> ${benchmark_out_name}
          done
        done
      fi
    fi
    '
  done

  rm -f ${tmp_input_file}
done


#!/bin/bash

echo "This is by no means a full test suite but if something fails here, it should be a concern"
echo "The tests passed for now should be checked manually, using: grep ntests <log_of_the_script_run>"

today=$(date +%F)

function func.check_npassed()
{
  ntests=$1
  resfile=$2
  label="$3"

  echo -e "ntests = $ntests, resfile = $resfile, label = $label \n"
  npassed=$(grep -c PASSED $resfile)
  echo -e "Number of $label tests passed: $npassed  out of $ntests (hardcoded in the test script) \n"

  local local_error=0

  if [[ "$npassed" != "$ntests" ]] ; then
    echo -e "Error: #tests passed for $label is wrong! \n"
    local_error=1
  fi

  echo $local_error
}

function func.run_test()
{
  ntests=$1
  label="$2"
  scriptline="$3"
  local local_error=0

  echo -e "Running $label tests with scriptline = '$scriptline' \n"

  resfile=log_run_test_suite_$today_$label.log

  echo -e "Executing...\n"
  #set -x
  #OMP_NUM_THREADS=56 ./my_run_bottleneck_ext_test_clone.sh --help
  #$envline $scriptline 2>&1 | tee $resfile
  mycmd=($scriptline)
  "${mycmd[@]}"  2>&1 | tee $resfile
  #set +x

  local_error=$(func.check_npassed $ntests $resfile $label)

  echo $local_error

}

date

export OMP_NUM_THREADS=56

# TPP PT extension modules
echo "TPP PT extension modules"

test_error=$(func.run_test 16 "btlnk_bf16" "python -u test_bottleneck_ext.py --test-module ext_bottleneck --use-physical-3x3-padding --use-hardcoded-tunings --use-bf16-opt --channel-block-size 32 --use-bf16-opt --test-data-file resnet50_bottleneck_test_data_56thr.data")
echo -e "test_error = $test_error\n"
test_error=$(func.run_test 52 "conv_bf16" "python -u test_conv_ext.py --test-module ext_tpp --use-bf16-opt --bc 32 --bk 32 --use-hardcoded-tunings --niters 1  --niters-warmup 0  --with-bwd --test-data-file resnet50_conv_test_data_for_bottleneck_56thr.data")
echo -e "test_error = $test_error\n"
test_error=$(func.run_test 53 "bn_bf16" "python -u test_bn_ext.py --test-module ext_tpp --bc 32 --use-bf16-opt --niters 1  --niters-warmup 0  --test-data-file resnet50_bn_test_data_56thr.data")
echo -e "test_error = $test_error\n"

test_error=$(func.run_test 16 "btlnk_fp32" "python -u test_bottleneck_ext.py --test-module ext_bottleneck --use-physical-3x3-padding --use-hardcoded-tunings --use-bf16-opt --channel-block-size 32 --test-data-file resnet50_bottleneck_test_data_56thr.data")
echo -e "test_error = $test_error\n"
test_error=$(func.run_test 52 "conv_fp32" "python -u test_conv_ext.py --test-module ext_tpp --bc 32 --bk 32 --use-hardcoded-tunings --niters 1  --niters-warmup 0  --with-bwd --test-data-file resnet50_conv_test_data_for_bottleneck_56thr.data")
echo -e "test_error = $test_error\n"
test_error=$(func.run_test 53 "bn_fp32" "python -u test_bn_ext.py --test-module ext_tpp --bc 32 --niters 1  --niters-warmup 0  --test-data-file resnet50_bn_test_data_56thr.data")
echo -e "test_error = $test_error\n"

echo "TPP CNN modules"

test_error=$(func.run_test 16 "btlnk_bf16_cnn" "python -u test_bottleneck_ext.py --test-module tpp_bottleneck --use-physical-3x3-padding --use-bf16-opt --channel-block-size 32 --use-bf16-opt --test-data-file resnet50_bottleneck_test_data_56thr.data")
echo -e "test_error = $test_error\n"
#exit
test_error=$(func.run_test 52 "conv_bf16_cnn" "python -u test_conv_ext.py --test-module cnn_tpp --use-bf16-opt --bc 32 --bk 32 --niters 1  --niters-warmup 0  --with-bwd --test-data-file resnet50_conv_test_data_for_bottleneck_56thr.data")
echo -e "test_error = $test_error\n"
test_error=$(func.run_test 53 "bn_bf16_cnn" "python -u test_bn_ext.py --test-module cnn_tpp --bc 32 --use-bf16-opt --niters 1  --niters-warmup 0  --test-data-file resnet50_bn_test_data_56thr.data")
echo -e "test_error = $test_error\n"

#exit

test_error=$(func.run_test 16 "btlnk_fp32_cnn" "python -u test_bottleneck_ext.py --test-module tpp_bottleneck --use-physical-3x3-padding --use-bf16-opt --channel-block-size 32 --test-data-file resnet50_bottleneck_test_data_56thr.data")
echo -e "test_error = $test_error\n"
test_error=$(func.run_test 52 "conv_fp32_cnn" "python -u test_conv_ext.py --test-module cnn_tpp --bc 32 --bk 32 --niters 1  --niters-warmup 0  --with-bwd --test-data-file resnet50_conv_test_data_for_bottleneck_56thr.data")
echo -e "test_error = $test_error\n"
test_error=$(func.run_test 53 "bn_fp32_cnn" "python -u test_bn_ext.py --test-module cnn_tpp --bc 32 --niters 1  --niters-warmup 0  --test-data-file resnet50_bn_test_data_56thr.data")
echo -e "test_error = $test_error\n"

#echo "test_error = $test_error"
date
exit


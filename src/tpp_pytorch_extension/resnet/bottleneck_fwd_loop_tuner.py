import subprocess
import os

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

import torch
import numpy as np
import time

from tune_bottleneck_fwd_driver import run_test_bottleneck

def getSizeOfNestedList(listOfElem):
    ''' Get number of elements in a nested list'''
    count = 0
    # Iterate over the list
    for elem in listOfElem:
        # Check if type of element is list
        if type(elem) == list:
            # Again call this function to get the size of this element
            count += getSizeOfNestedList(elem)
        else:
            count += 1
    return count

# A helper
def for_recursive(number_of_loops, range_list, execute_function, current_index=0, iter_list = [], **kwargs):
    #print("range_list = ", range_list)
    if iter_list == []:
        iter_list = [0]*number_of_loops

    if current_index == number_of_loops-1:
        for iter_list[current_index] in range_list[current_index]:
            execute_function(iter_list, **kwargs)
    else:
        for iter_list[current_index] in range_list[current_index]:
            for_recursive(number_of_loops, iter_list = iter_list, range_list = range_list,  current_index = current_index+1, execute_function = execute_function, **kwargs)

# Usage
#def do_whatever(index_list):
#    return print(index_list)
#for_recursive(range_list = [range(0,3), range(0,3) , range(1,3)], execute_function = do_whatever , number_of_loops=3)

def do_whatever(index_list):
    return print(index_list)

#for_recursive(range_list = [range(0,3), range(0,3) , range(1,3)], execute_function = do_whatever , number_of_loops=3)
for_recursive(range_list = [range(0,1), range(0,2), range(0,1)], execute_function = do_whatever, number_of_loops=3)

#exit()

def xbf_tester(index_list, cbfs=None, kbfs=None, hbfs=None, wbfs=None, hbfequal=None, wbfequal=None, hs_in_gemm=None, fuse_stats=None, has_downsample=None, loop_string=None,
                nhwck_params=None, bs=None, stride=None, eps=None, expansion=None, use_physical_3x3_padding=True, use_groupnorm=False, pack_input=None,
                opt_dtype=torch.float, ref_dtype=torch.float, with_perf=True, with_validation=True, test_module='ext_bottleneck', ref_module='pt_native', niters=20): # **kwargs):
    #for key, value in kwargs.items():
    #    print("The value of {} is {}".format(key, value))
    print("dbg: index_list = ", index_list)

    #run_test_bottleneck(N, H, W, inc, outc, bc_conv1, bc_conv2, bc_conv3, bk_conv3, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm, opt_dtype, ref_dtype, with_perf, with_validation, test_module, ref_module, tuning_params, tuning_strings, niters):
    """
        [h1_block, w1_block, h2_block, w2_block, h3_block, w3_block, h4_block, w4_block,
         c1_block, k1_block, c2_block, k2_block, c3_block, k3_block, c4_block, k4_block,
         h1_in_gemm, h2_in_gemm, h3_in_gemm, h4_in_gemm,
         pack_input_for_1x1_strided,
         fuse_stats ] = tuning_params
    """
    # defaults
    h_and_w_blocks = [1, 1, 1, 1, 1, 1, 1, 1]
    c_and_k_blocks = [1, 1, 1, 1, 1, 1, 1, 1]

    if cbfs is not None:
        c_blocks = [ cbfs[i][index_list[j]] for (i,j) in zip([0, 1, 2, 3], [0, 1, 2, 3])]
    else:
        c_blocks = [1, 1, 1, 1]
    #c_blocks = [] if cbfs is not None else [1, 1, 1, 1]
    print("dbg: c_blocks = ", c_blocks)

    if kbfs is not None:
        k_blocks = [ kbfs[i][index_list[j]] for (i,j) in zip([0, 1, 2, 3], [4, 5, 6, 7])]
    else:
        k_blocks = [1, 1, 1, 1]
    print("dbg: k_blocks = ", k_blocks)

    print("dbg: hbfs = ", hbfs)
    #print("dbg: zip for h = ", zip([0, 1, 2, 3], [8, 9, 10, 11]))

    if hbfs is not None:
        if hbfequal == True:
            h_blocks = [ hbfs[0][index_list[8]] ] * 4
        else:
            h_blocks = [ hbfs[i][index_list[j]] for (i,j) in zip([0, 1, 2, 3], [8, 9, 10, 11])]
    else:
        h_blocks = [1, 1, 1, 1]
    print("dbg: h_blocks = ", h_blocks)

    if wbfs is not None:
        if wbfequal == True:
            w_blocks = [ wbfs[0][index_list[12]] ] * 4
        else:
            w_blocks = [ wbfs[i][index_list[j]] for (i,j) in zip([0, 1, 2, 3], [12, 13, 14, 15])]
    else:
        w_blocks = [1, 1, 1, 1]
    print("dbg: w_blocks = ", w_blocks)

    # intersperse blocks for c and k, and also h and w
    c_and_k_blocks = [item for sublist in zip(c_blocks, k_blocks) for item in sublist]
    h_and_w_blocks = [item for sublist in zip(h_blocks, w_blocks) for item in sublist]

    print("dbg: c_and_k_blocks = ", c_and_k_blocks)
    print("dbg: h_and_w_blocks = ", h_and_w_blocks)
    
    tuning_params = [ *h_and_w_blocks,
                      *c_and_k_blocks,
                      *hs_in_gemm,
                       pack_input,
                       fuse_stats
                    ]
    
    tuning_strings = [ loop_string, loop_string, loop_string, loop_string ]

    
    run_test_bottleneck(*nhwck_params, bs, bs, bs, bs, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm,
                         opt_dtype, ref_dtype, with_perf, with_validation, test_module, ref_module,
                         tuning_params, tuning_strings, niters)
    
    return print(index_list)

#for_recursive(range_list = [range(0,1), range(0,2) , range(0,1)], execute_function = xbf_tester, number_of_loops=3)

loop_names = []
loop_specs = []

for i in range(1,2):
    for j in range(1,2):
        #for k in range(1,2):
        #    loop_names.append('bottleneck_first')
        #    loop_specs.append('A_0_M,' + 'b_0_K,' + ('c_' + str(i) + '_K,') + ('d_' + str(j) +'_K,') + ('e_' + str(k) +'_K,') + 'f_0_K,' + 'g_0_K')
        #loop_names.append('bottleneck_middle')
        #loop_specs.append('A_0_M,' + 'b_0_K,' + ('c_' + str(i) + '_K,') + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')
        loop_names.append('bottleneck_notlast')
        loop_specs.append('A_0_M,' + 'b_0_K,' + ('c_' + str(i) + '_K,') + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')
        loop_names.append('bottleneck_last')
        loop_specs.append('A_0_M,' + 'b_0_K,' + ('c_' + str(i) + '_M,') + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')

print("dbg: loop_names: ", loop_names)
print("dbg: loop_specs: ", loop_specs)

for i in range(len(loop_names)):
    cmd = './loop_permute_generator ' + loop_names[i] + ' ' + loop_specs[i] # extra optional arg for output file
    print("i, cmd to execute = ", i, cmd)
    os.system(cmd) # returns the exit status

#exit()

loop_lines = {} #'bottleneck_first': [], 'bottleneck_middle': [], 'bottleneck_last': []}
nLoops = {}

for name in loop_names:
    print("dbg: loop_name = ", name)
    cmd = 'cat ' + name + '_bench_configs.txt > uber_config.txt'
    print("dbg: ", cmd)
    os.system(cmd) # returns the exit status
    cmd = "awk '!seen[$0]++' uber_config.txt > tuner_config.txt"
    print("dbg: ", cmd)
    os.system(cmd) # returns the exit status
    cmd = 'rm uber_config.txt'
    print("dbg: ", cmd)
    os.system(cmd) # returns the exit status

#cat *bench_configs.txt > uber_config.txt
#awk '!seen[$0]++' uber_config.txt > tuner_config.txt
#rm uber_config.txt

#exit()

    with open("tuner_config.txt",'r') as f:
        loop_lines[name] = f.read().splitlines()
        nLoops[name] = len(loop_lines[name])
        print("dbg: total number of loop lines for name = ", name, nLoops[name])

#exit()

nBottlenecks = 8 #8

# fixed defaults for all bottlenecks (non-tunable)
expansion=4
eps=1e-7

nthreads=int(os.getenv("OMP_NUM_THREADS"))
print("dbg: nthreads = ", nthreads)

for l in [2]: #range(nBottlenecks):
    #file_path = 'bottleneck_' + str(l) + '_tuning_dbg_0705.txt'
    #sys.stdout = open(file_path, "w")

    # common parameters (potentially overwrite-able)
    N=nthreads
    bs=32
    #bc=32
    #bk=32
    niters=20
    hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
    #pack_input=0 # is an option only for a partivcular 1x1 strided convs

    print("l = ", l)

    if l == 0:
          base_sizes = [N, 56, 56, 64, 64]
          stride=1
          CBFS=None #[[1]]
          WBFS=[[1, 2]] #WBFS=[[1, 2]]
          HBFS=[[4, 7]] # 14
          KBFS=None #[[1]]
          hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
          pack_input = 0
          downsample=True
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 1:
          base_sizes = [N, 56, 56, 256, 64]
          stride=1
          CBFS=None #[[1]]
          WBFS=[[1, 2]] #WBFS=[[1, 2]]
          HBFS=[[4, 7]] # 14
          KBFS=None #[[1]]
          hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
          pack_input = 0
          downsample=False
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 2:
          base_sizes = [N, 56, 56, 256, 128]
          stride=2
          CBFS=None #[[1]]
          WBFS=[[1, 2], [1], [1], [1]] #[[1, 2], [1], [1], [1]]
          HBFS=[[4, 7], [4, 7], [4], [4]]
          KBFS=None #[[1]]
          hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
          pack_input = 0
          #dont pack_input for 1x1 strided, 28 W should be enough
          downsample=True
          config_name='bottleneck_notlast'
          hbfequal = False
          wbfequal = False
    elif l == 3:
          base_sizes = [N, 28, 28, 512, 128]
          stride=1
          CBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=[[7]]
          KBFS=[[1], [1], [1], [4]] # [[1], [1], [1], [1,4]] #if extended
          hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
          pack_input = 0
          downsample=False
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 4:
          base_sizes = [N, 28, 28, 512, 256]
          stride=2
          CBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=None #[[1]]
          KBFS=None #[[1]] # ??? if extended potentially
          hs_in_gemm = [1, 2, 1, 2] #h in gemm 2 for all except the first one and 3x3+str2
          #1x1 with stride 2 should
          pack_input = 1
          downsample=True
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 5:
          base_sizes = [N, 14, 14, 1024, 256]
          stride=1
          CBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=None #[[1]]
          KBFS=None #[[1]] # potentially, KBFS for the last one
          hs_in_gemm = [2, 2, 2, 2] #h in gemm 2 for all except the first one and 3x3+str2
          pack_input = 0
          downsample=False
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 6:
          base_sizes = [N, 14, 14, 1024, 512]
          stride=2
          CBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=None #[[1]]
          KBFS=None #[[1]] # potentially, KBFS for the last one
          hs_in_gemm = [2, 1, 7, 7]
          pack_input=1
          downsample=True
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 7:
          base_sizes = [N, 7, 7, 1024, 512]
          stride=1
          CBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=None #[[1]]
          KBFS=None #[[1]] # potentially, KBFS for the last one
          hs_in_gemm=[7, 7, 7, 7]
          downsample=False
          config_name='bottleneck_last'
          hbfequal = True
          wbfequal = True
    else:
          print("Error: index l does not mactch any existing cases, l = ", l)
          exit()

    ncombinations = 0
    nlines = 0

    for line in loop_lines[config_name]:
        if   config_name != 'bottleneck_last' and not line.startswith('Afgb'):
            continue
        elif config_name == 'bottleneck_last' and not (line.startswith('ACfgb') or line.startswith('Afgb') or line.startswith('CAfgb')):
            continue
        else:
            print("line = ", line)
            CBFcount = line.count('b') #$( echo ${lowerline} | tr -d -c 'c' | awk '{ print length; }' )
            print("CBFcount = ", CBFcount)
            KBFcount = line.count('c') #$( echo ${lowerline} | tr -d -c 'c' | awk '{ print length; }' )
            print("KBFcount = ", KBFcount)
            #echo "C count is ${KBFcount}"
            HBFcount = line.count('d') #$( echo ${lowerline} | tr -d -c 'd' | awk '{ print length; }' )
            print("HBFcount = ", HBFcount)
            WBFcount = line.count('e') #$( echo ${lowerline} | tr -d -c 'd' | awk '{ print length; }' )
            print("WBFcount = ", WBFcount)
            nlines = nlines + 1

        #for_recursive(range_list = [range(0,3), range(0,3) , range(1,3)], execute_function = do_whatever , number_of_loops=3)
        for fuse_stats in range(0,2):
            nbfloops = 0

            # CBFS and WBFS are different from KBFS and HBFS as there are not explicitly used as blocking factors in the loop strings
            if CBFS is None: # or CBFcount == 1
                use_cbfs = None
                #nbfloops = nbfloops + 1
            else:
                use_cbfs = CBFS if len(CBFS) == 4 else [CBFS[0]]*4
                #nbfloops = nbfloops + 4
            nbfloops = nbfloops + 4

            if KBFcount == 1 or KBFS is None:
                use_kbfs = None
                #nbfloops = nbfloops + 1
            else:
                use_kbfs = KBFS if len(KBFS) == 4 else [KBFS[0]]*4
                #nbfloops = nbfloops + 4
            nbfloops = nbfloops + 4

            if HBFcount == 1 or HBFS is None:
                use_hbfs = None
                #nbfloops = nbfloops + 1
            else:
                use_hbfs = HBFS if len(HBFS) == 4 else [HBFS[0]]*4
                #nbfloops = nbfloops + 4
            nbfloops = nbfloops + 4

            # CBFS and WBFS are different from KBFS and HBFS as there are not explicitly used as blocking factors in the loop strings
            if WBFS is None: # or WBFcount == 1
                use_wbfs = None
                #nbfloops = nbfloops + 1
            else:
                use_wbfs = WBFS if len(WBFS) == 4 else [WBFS[0]]*4
                #nbfloops = nbfloops + 4
            nbfloops = nbfloops + 4

            #use_cbfs = None if CBFcount == 1 else CBFS
            #use_kbfs = None if KBFcount == 1 else KBFS
            #use_hbfs = None if HBFcount == 1 else HBFS
            #use_wbfs = None if WBFcount == 1 else WBFS

            if CBFS is not None and len(CBFS) != 1 and len(CBFS) != 4:
                print("Error: CBFS as a list must have either 1 or 4 elements (sub-lists)")
            if KBFS is not None and len(KBFS) != 1 and len(KBFS) != 4:
                print("Error: KBFS as a list must have either 1 or 4 elements (sub-lists)")
            if HBFS is not None and len(HBFS) != 1 and len(HBFS) != 4:
                print("Error: HBFS as a list must have either 1 or 4 elements (sub-lists)")
            if WBFS is not None and len(WBFS) != 1 and len(WBFS) != 4:
                print("Error: WBFS as a list must have either 1 or 4 elements (sub-lists)")

            cf_ranges = [range(0, len(i)) for i in use_cbfs] if use_cbfs is not None else [range(0,1)]*4
            kf_ranges = [range(0, len(i)) for i in use_kbfs] if use_kbfs is not None else [range(0,1)]*4
            #hf_ranges = [range(0, len(i)) for i in use_hbfs] if use_hbfs is not None else [range(0,1)]*4
            if use_hbfs is not None:
                if hbfequal == True:
                    hf_ranges = [range(0, len(use_hbfs[0])), range(0,1), range(0,1), range(0,1)]
                else:
                    hf_ranges = [range(0, len(i)) for i in use_hbfs]
            else:
                hf_ranges = [range(0,1)]*4

            #wf_ranges = [range(0, len(i)) for i in use_wbfs] if use_wbfs is not None else [range(0,1)]*4
            if use_wbfs is not None:
                if wbfequal == True:
                    wf_ranges = [range(0, len(use_wbfs[0])), range(0,1), range(0,1), range(0,1)]
                else:
                    wf_ranges = [range(0, len(i)) for i in use_wbfs]
            else:
                wf_ranges = [range(0,1)]*4

            #cf_ranges = [range(0, len(CBFS[0])), range(0, len(CBFS[1])), range(0, len(CBFS[2])), range(0, len(CBFS[3]))] if len(CBFS) == 4 else [range(0, len(CBFS[1]))] * 4
            #kf_ranges = [range(0, len(KBFS[0])), range(0, len(KBFS[1])), range(0, len(KBFS[2])), range(0, len(KBFS[3]))] if len(KBFS) == 4 else [range(0, len(KBFS[1]))] * 4
            #hf_ranges = [range(0, len(HBFS[0])), range(0, len(HBFS[1])), range(0, len(HBFS[2])), range(0, len(HBFS[3]))] if len(HBFS) == 4 else [range(0, len(HBFS[1]))] * 4
            #wf_ranges = [range(0, len(WBFS[0])), range(0, len(WBFS[1])), range(0, len(WBFS[2])), range(0, len(WBFS[3]))] if len(WBFS) == 4 else [range(0, len(WBFS[1]))] * 4

            if downsample == False:
                cf_ranges[3] = range(0,1)
                kf_ranges[3] = range(0,1)
                hf_ranges[3] = range(0,1)
                wf_ranges[3] = range(0,1)

            for_recursive(range_list = [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges], execute_function = xbf_tester, number_of_loops = nbfloops,
                          cbfs=use_cbfs, wbfs=use_wbfs, hbfs=use_hbfs, kbfs=use_kbfs, hbfequal=hbfequal, wbfequal=wbfequal, hs_in_gemm = hs_in_gemm, fuse_stats = fuse_stats, bs = bs,
                          nhwck_params=base_sizes, stride = stride, eps = eps, expansion = expansion, has_downsample=downsample,
                          opt_dtype = torch.bfloat16,
                          loop_string=line,
                          pack_input=pack_input
                          )
            #ncombinations = ncombinations + sum([len[rangevar] for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
            print("range_list = ", [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges])
            print("its tmp list = ", [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
            print("its reduce product = ", reduce(operator.mul, [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]]))
            ncombinations = ncombinations + reduce(operator.mul, [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
            print("")
#  common_runline="python -u tune_bottleneck_fwd_driver.py --test-module ext_bottleneck --ref-module pt_native --use-physical-3x3-padding --use-bf16-opt --with-validation "
#                  block_sizes_line=" --block-sizes ${bc} ${bc} ${bk} ${bk} "
#                  hw_blocks_subline="${hb} ${wb} ${hb} ${wb} ${hb} ${wb} ${hb} ${wb}"
#                  ck_blocks_subline="${cb} ${kb} ${cb} ${kb} ${cb} ${kb} ${cb} ${kb}"
#                  tuning_params_line=" --tuning-params ${hw_blocks_subline} ${ck_blocks_subline} ${h_in_gemm} ${h_in_gemm} ${h_in_gemm} ${h_in_gemm} ${fuse_stats}"
#                  tuning_strings_line=" --tuning-strings ${line} ${line} ${line} ${line}"
#                  ${common_runline} ${block_sizes_line} ${tuning_params_line} --test-data-file $tmp_input_file ${tuning_strings_line} --niters ${niters} >> ${benchmark_out_name}

#    break # while developing
    print("l, config_name, nlines, ncombinations = ", l, config_name, nlines, ncombinations)
exit()

"""
source /swtools/gcc/latest/gcc_vars.sh

export OMP_NUM_THREADS=24
export GOMP_CPU_AFFINITY="0-23"

loop_names
loop_specs=()

#For now we don't block additionally K dim, handled by BF of BR dimension
for i in 0 1; do
  for j in 0 1; do
#for i in 0; do
  #for j in 0; do
    loop_names+=("conv1")
    loop_specs+=("A_0_M,b_0_K,c_${i}_K,d_${j}_K,e_0_K,f_0_K,g_0_K")
    loop_specs+=("A_0_M,b_0_K,c_${i}_M,d_${j}_K,e_0_K,f_0_K,g_0_K") for the last bottleneck
    #loop_specs+=("A_0_M,b_0_K,c_${i}_M,d_0_K,e_0_K,f_0_K,g_0_K")
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

nBottlenecks=1

HBFS=("1")
KBFS=("1")

export OMP_NUM_THREADS=24
export GOMP_CPU_AFFINITY="0-23"
unset LIBXSMM_VERBOSE

expansion=4
eps="1e-7"

for (( l = 0 ; l < $nBottlenecks ; l++)); do
  echo "l = $l"
  N=${OMP_NUM_THREADS}
  bc=32
  bk=32
  niters=20
  if [ $l -eq 0 ]; then
    H=56
    W=56
    C=64
    K=64
    str=1
    CBFS=("1")
    WBFS=("1" "2")
    HBFS=("4" "7") # 14
    KBFS=("1")
    downsample="True"
    h_in_gemm=1
  fi

  if [ $l -eq 1 ]; then
    H=56
    W=56
    C=256
    K=64
    str=1
    CBFS=("1")
    WBFS=("1" "2")
    HBFS=("4" "7") # 14
    KBFS=("1")
    downsample="False"
  fi

  if [ $l -eq 2 ]; then
    H=56
    W=56
    C=256
    K=128
    str=2
    CBFS=("1")
    WBFS=("1") for all but the first one, for the first one 1 and "2")
    HBFS=("4" "7") and maybe for the last two, no 7 
    KBFS=("1")
    dont pack_input for 1x1 strided, 28 W should be enough
    downsample="True"
  fi

  if [ $l -eq 3 ]; then
    H=28
    W=28
    C=512
    K=128
    str=1
    CBFS=("1")
    WBFS=("1")
    HBFS=("7")
    KBFS=("1") but for the last one KBFS = 1, 4 if extended
    downsample="False"
  fi

  if [ $l -eq 4 ]; then
    H=28
    W=28
    C=512
    K=256
    str=2
    CBFS=("1")
    WBFS=("1")
    HBFS=("1") 
    KBFS=("1") # potentially, KBFS
    h in gemm 2 for all except the first one and 3x3+str2
    #1x1 with stride 2 should 
    pack_input for 1x1 strided
    downsample="True"
  fi

  if [ $l -eq 5 ]; then
    H=14
    W=14
    C=1024
    K=256
    str=1
    CBFS=("1")
    WBFS=("1")
    HBFS=("1")
    KBFS=("1") # potentially, KBFS for the last one
    h_in_gemm=2
    downsample="False"
  fi

  if [ $l -eq 6 ]; then
    H=14
    W=14
    C=1024
    K=512
    str=2
    CBFS=("1")
    WBFS=("1")
    HBFS=("1")
    KBFS=("1") # potentially, KBFS for the last one
    h_in_gemm = 2 for the first one
    h_in_gemm=1 for the middle one
    h_in_gemm=7 for the last one
    downsample="True"
  fi

  # Should have AC and CA
  if [ $l -eq 7 ]; then
    H=7
    W=7
    C=2048
    K=512
    str=1
    CBFS=("1")
    WBFS=("1")
    HBFS=("1")
    KBFS=("1") # potentially, KBFS for the last one
    h_in_gemm=7
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

  nloops_actual=1
  #nLoops=1 # while developing the script
  for (( j = 0 ; j < $nLoops ; j++)); do
    line=${loopArray[$j]}
    #echo ${line}
    lowerline=$(echo ${line} | tr '[:upper:]' '[:lower:]')
    #echo ${lowerline}
    KBFcount=$( echo ${lowerline} | tr -d -c 'c' | awk '{ print length; }' )
    #echo "C count is ${KBFcount}"
    HBFcount=$( echo ${lowerline} | tr -d -c 'd' | awk '{ print length; }' )
    #echo "D count is ${HBFcount}"

    h_in_gemm=1
    #echo "bc = ${bc}, bk = ${bk}"
    
    #if [ $KBFcount -eq 2 ]; then
      if [ $HBFcount -eq 2 ]; then
        for cb in "${CBFS[@]}"; do
          for kb in "${KBFS[@]}"; do
            for wb in "${WBFS[@]}"; do
              for hb in "${HBFS[@]}"; do
                for fuse_stats in 1 ; do
    if [[ $line == Afgb* ]]
    then
      #echo "line counted = ${line}"
      let "nloops_actual+=1"
    else
      #echo "line skipped = ${line}"
      continue
    fi
                  : '  
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
                  '
                  #break 5 # for developing the script
                done
              done
            done
          done
        done
      fi
    #fi
 
      if [ $HBFcount -eq 1 ]; then
        for cb in "${CBFS[@]}"; do
          for kb in "${KBFS[@]}"; do
            for wb in "${WBFS[@]}"; do
              #for hb in "${HBFS[@]}"; do
              for hb in 1 ; do
                for fuse_stats in 1 ; do
    if [[ $line == Afgb* ]]
    then
      #echo "line counted = ${line}"
      let "nloops_actual+=1"
    else
      #echo "line skipped = ${line}"
      continue
    fi 
                  : ' 
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
                  #break 5 # for developing the script
                  '
                done
              done
            done
          done
        done
      fi

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
  echo "nloops_actual = $nloops_actual"
  rm -f ${tmp_input_file}
done

"""

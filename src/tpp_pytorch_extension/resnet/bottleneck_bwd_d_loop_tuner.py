import subprocess
import os
import sys

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

import torch
import numpy as np
import time

from bottleneck_tuning_bwd_d_driver import run_test_bottleneck

script_version='v2'

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
#for_recursive(range_list = [range(0,3), range(0,3), range(1,3)], execute_function = do_whatever , number_of_loops=3)
#for_recursive(range_list = [range(0,3), range(0,3), range(1,3)], execute_function = do_whatever , number_of_loops=3)
#for_recursive(range_list = [range(0,1), range(0,2), range(0,1)], execute_function = do_whatever, number_of_loops=3)
#exit()

def xbf_tester(index_list, cbfs=None, kbfs=None, hbfs=None, wbfs=None, hbfequal=None, wbfequal=None, hs_in_gemm=None, has_downsample=None, loop_string=None,
                nhwck_params=None, bs=None, stride=None, eps=None, expansion=None, use_physical_3x3_padding=True, use_groupnorm=False,
                opt_dtype=torch.float, ref_dtype=torch.float, with_perf=True, with_validation=True, test_module='ext_bottleneck', ref_module='pt_native', niters=20): # **kwargs):
    print("dbg: index_list = ", index_list)

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
                      *hs_in_gemm
                    ]

    tuning_strings = [ loop_string, loop_string, loop_string, loop_string ]

    #print("run_test_bottleneck is called", flush=True)
    #exit()
    run_test_bottleneck(*nhwck_params, bs, bs, bs, bs, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm,
                         opt_dtype, ref_dtype, with_perf, with_validation, test_module, ref_module,
                         tuning_params, tuning_strings, niters)
    
    return print(index_list)

#for_recursive(range_list = [range(0,1), range(0,2) , range(0,1)], execute_function = xbf_tester, number_of_loops=3)

loop_names = []
loop_specs = []

for i in range(1,2):
    for j in range(1,2):
        loop_names.append('bottleneck_notlast')
        # This was for fwd but we need to swap C and K loops = b and c letters
        # loop_specs.append('A_0_M,' + 'b_0_K,' + ('c_' + str(i) + '_K,') + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')
        loop_specs.append('A_0_M,' + ('b_' + str(i) + '_K,') + 'c_0_K,' + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')
        loop_names.append('bottleneck_last')
        # This was for fwd but we need to swap C and K loops = b and c letters
        #loop_specs.append('A_0_M,' + 'b_0_K,' + ('c_' + str(i) + '_M,') + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')
        loop_specs.append('A_0_M,' + ('b_' + str(i) + '_M,') + 'c_0_K,' + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K,' + 'g_0_K')

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

for l in range(nBottlenecks):
    #file_path = 'bottleneck_' + str(l) + '_tuning_dbg.txt'
    #sys.stdout = open(file_path, "w")

    # common parameters (potentially overwrite-able)
    N=nthreads
    bs=32
    #bc=32
    #bk=32
    niters=400
    #hs_in_gemm=[1, 1, 1, 1] # 1 per each conv

    print("l = ", l)

    if l == 0:
          base_sizes = [N, 56, 56, 64, 64]
          stride=1
          KBFS=None #[[1]]
          WBFS=[[1, 2]] #WBFS=[[1, 2]]
          HBFS=[[4, 7]] # 14
          CBFS=None #[[1]]
          hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
          downsample=True
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 1:
          base_sizes = [N, 56, 56, 256, 64]
          stride=1
          KBFS=None #[[1]]
          WBFS=[[1, 2]] #WBFS=[[1, 2]]
          HBFS=[[4, 7]] # 14
          CBFS=None #[[1]]
          hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
          downsample=False
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 2:
          base_sizes = [N, 56, 56, 256, 128]
          stride=2
          KBFS=None #[[1]]
          WBFS=[[1, 2], [1], [1], [1]] #[[1, 2], [1], [1], [1]]
          HBFS=[[4, 7], [4, 7], [4], [4]]
          CBFS=None #[[1]]
          hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
          downsample=True
          config_name='bottleneck_notlast'
          hbfequal = False
          wbfequal = False
    elif l == 3:
          base_sizes = [N, 28, 28, 512, 128]
          stride=1
          KBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=[[7]]
          CBFS=[[1], [1], [2, 4], [1]] # [[1], [1], [1], [1,4]] #if extended
          hs_in_gemm=[1, 1, 1, 1] # 1 per each conv
          downsample=False
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 4:
          base_sizes = [N, 28, 28, 512, 256]
          stride=2
          KBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=None #[[1]]
          CBFS=[[1], [1], [2, 4, 8], [2, 4, 8]] #None #[[1]] # ??? if extended potentially
          hs_in_gemm = [1, 1, 2, 2] #h in gemm 2 for all except the first one and 3x3+str2
          downsample=True
          config_name='bottleneck_notlast'
          hbfequal = False
          wbfequal = True
    elif l == 5:
          base_sizes = [N, 14, 14, 1024, 256]
          stride=1
          KBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=None #[[1]]
          CBFS=[[2, 4, 8], [1], [2, 4, 8], [1]] #None #[[1]] # potentially, KBFS for the last one
          hs_in_gemm = [2, 2, 2, 2] #h in gemm 2 for all except the first one and 3x3+str2
          downsample=False
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 6:
          base_sizes = [N, 14, 14, 1024, 512]
          stride=2
          KBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=None #[[1]]
          CBFS=[[4, 8], [4, 8], [4, 8, 16], [4, 8, 16] ] #None #[[1]] # potentially, KBFS for the last one
          hs_in_gemm = [2, 1, 7, 7]
          downsample=True
          config_name='bottleneck_notlast'
          hbfequal = True
          wbfequal = True
    elif l == 7:
          base_sizes = [N, 7, 7, 2048, 512]
          stride=1
          KBFS=None #[[1]]
          WBFS=None #[[1]]
          HBFS=None #[[1]]
          CBFS=[[2, 4, 8], [2, 4, 8], [2, 4, 8, 16], [1]] #None #[[1]] # potentially, KBFS for the last one
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
        if   config_name != 'bottleneck_last' and not line.startswith('Afgc'):
            continue
        elif config_name == 'bottleneck_last' and not (line.startswith('ABfgc') or line.startswith('Afgc') or line.startswith('BAfgc')):
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

            if CBFcount > 1 and CBFS == None:
                continue
            elif KBFcount > 1 and KBFS == None:
                continue
            elif HBFcount > 1 and HBFS == None:
                continue
            elif WBFcount > 1 and WBFS == None:
                continue

        if stride == 2 and pack_input_var == 0 and hs_in_gemm[3] != 1:
            continue
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

        if CBFS is not None and len(CBFS) != 1 and len(CBFS) != 4:
            print("Error: CBFS as a list must have either 1 or 4 elements (sub-lists)")
        if KBFS is not None and len(KBFS) != 1 and len(KBFS) != 4:
            print("Error: KBFS as a list must have either 1 or 4 elements (sub-lists)")
        if HBFS is not None and len(HBFS) != 1 and len(HBFS) != 4:
            print("Error: HBFS as a list must have either 1 or 4 elements (sub-lists)")
        if WBFS is not None and len(WBFS) != 1 and len(WBFS) != 4:
            print("Error: WBFS as a list must have either 1 or 4 elements (sub-lists)")

        #print("dbg: use_kbfs = ", use_kbfs, KBFS)
        #if use_kbfs is not None:
        #    for i in use_kbfs:
        #        print("dbg: i, len(i) in kbfs = ", i, len(i))

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

        if downsample == False:
            cf_ranges[3] = range(0,1)
            kf_ranges[3] = range(0,1)
            hf_ranges[3] = range(0,1)
            wf_ranges[3] = range(0,1)

        print("range_list = ", [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges])

        for_recursive(range_list = [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges], execute_function = xbf_tester, number_of_loops = nbfloops,
                      cbfs=use_cbfs, wbfs=use_wbfs, hbfs=use_hbfs, kbfs=use_kbfs, hbfequal=hbfequal, wbfequal=wbfequal, hs_in_gemm = hs_in_gemm, bs = bs,
                      nhwck_params=base_sizes, stride = stride, eps = eps, expansion = expansion, has_downsample=downsample, niters=niters,
                      opt_dtype = torch.bfloat16,
                      loop_string=line )
        #ncombinations = ncombinations + sum([len[rangevar] for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
        #print("range_list = ", [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges])
        #print("its tmp list = ", [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
        #print("its reduce product = ", reduce(operator.mul, [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]]))
        ncombinations = ncombinations + reduce(operator.mul, [len(rangevar) for rangevar in [*cf_ranges, *kf_ranges, *hf_ranges, *wf_ranges]])
        print("")
    print("script version, l, config_name, nlines, ncombinations = ", script_version, l, config_name, nlines, ncombinations)
exit()


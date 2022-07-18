import subprocess
import os
import sys

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

import torch
import numpy as np
import time

from bottleneck_tuning_bwd_w_driver import run_test_bottleneck

script_version='v4'

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

def xbf_tester(index_list, pbfs=None, pbfequal=None, pbfequal_c1c3c4=None, has_downsample=None, loop_string_c1c3c4=None, loop_string_c2=None, use_nchw_formats = None,
                nhwck_params=None, bs=None, stride=None, eps=None, expansion=None, use_physical_3x3_padding=True, use_groupnorm=False,
                pack_input_upfront=None, fuse_upd_transposes=None, use_f32_wt_reduction_and_external_wt_vnni=None,
                acc_nw=None, par_over_h=None, compute_full_wt_output_block=None,
                hybrid=None, n_img_teams=None, n_ofm_teams=None,
                opt_dtype=torch.float, ref_dtype=torch.float, with_perf=True, with_validation=True, test_module='ext_bottleneck', ref_module='pt_native', niters=20): # **kwargs):
    #print("dbg: index_list = ", index_list)

    # defaults
    p_blocks = [1, 1, 1, 1]

    #print("dbg: pbfs = ", pbfs)
    if pbfs is not None:
        if pbfequal == True:
            p_blocks = [ pbfs[0][index_list[0]] ] * 4
        else:
            p_blocks = [ pbfs[i][index_list[j]] for (i,j) in zip([0, 1, 2, 3], [0, 1, 2, 3])]
    else:
        p_blocks = [1, 1, 1, 1]
    #print("dbg: p_blocks = ", p_blocks)

    tuning_params = [ *p_blocks,
                      *use_nchw_formats,
                      pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                      acc_nw, par_over_h, compute_full_wt_output_block,
                      hybrid, n_img_teams, n_ofm_teams
                    ]

    tuning_strings = [ loop_string_c1c3c4, loop_string_c2, loop_string_c1c3c4, loop_string_c1c3c4 ]

    print("dbg: tuning_params  = ", tuning_params)
    print("dbg: tuning_strings = ", tuning_strings)

    #print("run_test_bottleneck is called", flush=True)
    #exit()
    
    run_test_bottleneck(*nhwck_params, bs, bs, bs, bs, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm,
                         opt_dtype, ref_dtype, with_perf, with_validation, test_module, ref_module,
                         tuning_params, tuning_strings, niters)
    
    #exit()
    return print(index_list)

#for_recursive(range_list = [range(0,1), range(0,2) , range(0,1)], execute_function = xbf_tester, number_of_loops=3)

loop_names = []
loop_specs = []

for i in range(1,2):
    for j in range(1,2):
        loop_names.append('bottleneck_nchw_1x1')
        #loop_specs.append('A_0_M,' + 'b_0_K,' + 'c_0_K,' + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K')
        loop_specs.append('A_0_M,' + 'b_0_K,' + 'c_0_K,' + 'd_0_K,' + 'e_0_K,' + 'f_0_K')
        loop_names.append('bottleneck_nchw_3x3') # will use different skipping condition potentially than 1x1
        #loop_specs.append('A_0_M,' + 'b_0_K,' + 'c_0_K,' + ('d_' + str(j) +'_K,') + 'e_0_K,' + 'f_0_K')
        loop_specs.append('A_0_M,' + 'b_0_K,' + 'c_0_K,' + 'd_0_K,' + 'e_0_K,' + 'f_0_K')
        loop_names.append('bottleneck_chwn')
        loop_specs.append('A_0_M,' + 'B_0_M,' + 'c_0_K,' + 'd_0_K,' + 'e_0_M,' + 'f_0_M')
        loop_names.append('bottleneck_hybrid') # only for 7x7? # AC is a must with 2d parallelization A{C:X} + C{R:Y}
        loop_specs.append('A_0_M,' + 'b_0_M,' + 'C_0_K,' + ('d_' + str(j) +'_K,') + 'e_0_M,' + 'f_0_M')

print("dbg: loop_names: ", loop_names)
print("dbg: loop_specs: ", loop_specs)

for i in range(len(loop_names)):
    cmd = './loop_permute_generator ' + loop_names[i] + ' ' + loop_specs[i] # extra optional arg for output file
    print("i, cmd to execute = ", i, cmd)
    os.system(cmd) # returns the exit status

#exit()

loop_lines = {}
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

nBottlenecks = 8 #6 #8

# fixed defaults for all bottlenecks (non-tunable)
expansion=4
eps=1e-7

nthreads=int(os.getenv("OMP_NUM_THREADS"))
print("dbg: nthreads = ", nthreads)

for l in [6, 7]: #[2, 3, 4, 5]: #range(nBottlenecks):
    #file_path = 'bottleneck_' + str(l) + '_tuning_dbg.txt'
    #sys.stdout = open(file_path, "w")

    # common parameters (potentially overwrite-able)
    N=nthreads
    bs=32
    #bc=32
    #bk=32
    niters=10

    fuse_upd_transposes_limit=1
    use_f32_wt_reduction_and_external_wt_vnni_limit=1
    acc_nw_limit=1
    par_over_h_limit=1
    compute_full_wt_output_block_limit=1
    hybrid=0
    n_img_teams=1
    n_ofm_teams=1
    pbfequal_c1c3c4=True
    print("l = ", l)

    if l == 0:
          base_sizes = [N, 56, 56, 64, 64]
          stride=1
          downsample=True
          CBFS=None #[[1]]
          KBFS=None #[[1]]
          PBFS=[[2, 4], [1], [2, 4], [2, 4]] # ???
          pack_input_upfront_limit=0
          config_name_c1c3c4='bottleneck_nchw_1x1'
          config_name_c2='bottleneck_nchw_3x3'
          pbfequal=True
    elif l == 1:
          base_sizes = [N, 56, 56, 256, 64]
          stride=1
          downsample=False
          KBFS=None #[[1]]
          CBFS=None #[[1]]
          PBFS=[[2, 4], [1], [2, 4], [2, 4]] # ???
          pack_input_upfront_limit=0
          config_name_c1c3c4='bottleneck_nchw_1x1'
          config_name_c2='bottleneck_nchw_3x3'
          pbfequal=True
    elif l == 2:
          base_sizes = [N, 56, 56, 256, 128]
          stride=2
          downsample=True
          KBFS=None #[[1]]
          CBFS=None #[[1]]
          PBFS=[[2, 4], [1], [1], [1]] # ???
          pack_input_upfront_limit=1
          config_name_c1c3c4='bottleneck_nchw_1x1'
          config_name_c2='bottleneck_chwn'
          pbfequal = False
    elif l == 3:
          base_sizes = [N, 28, 28, 512, 128]
          stride=1
          downsample=False
          KBFS=None #[[1]]
          CBFS=None #[[1]]
          PBFS=None # # ???
          pack_input_upfront_limit=0
          config_name_c1c3c4='bottleneck_nchw_1x1'
          config_name_c2='bottleneck_nchw_3x3'
          pbfequal = True
    elif l == 4:
          base_sizes = [N, 28, 28, 512, 256]
          stride=2
          downsample=True
          KBFS=None #[[1]]
          CBFS=None
          PBFS=None # # ???
          pack_input_upfront_limit=1
          config_name_c1c3c4='bottleneck_nchw_1x1'
          config_name_c2='bottleneck_chwn'
          pbfequal = True
    elif l == 5:
          base_sizes = [N, 14, 14, 1024, 256]
          stride=1
          downsample=False
          KBFS=None #[[1]]
          CBFS=None #[[1]]
          PBFS=None #[[1]]
          pack_input_upfront_limit=0
          config_name_c1c3c4='bottleneck_nchw_1x1'
          config_name_c2='bottleneck_nchw_3x3'
          pbfequal = True
    elif l == 6:
          base_sizes = [N, 14, 14, 1024, 512]
          stride=2
          downsample=True
          KBFS=None #[[1]]
          CBFS=None #[[1]]
          PBFS=None #[[1]]
          pack_input_upfront_limit=1
          config_name_c1c3c4='bottleneck_nchw_1x1'
          config_name_c2='bottleneck_chwn'
          pbfequal = True
    elif l == 7:
          base_sizes = [N, 7, 7, 2048, 512]
          stride=1
          downsample=False
          KBFS=None #[[1]]
          CBFS=None #[[1]]
          PBFS=None #[[1]]
          pack_input_upfront_limit=0
          config_name_c1c3c4='bottleneck_nchw_1x1'
          config_name_c2='bottleneck_nchw_3x3'
          pbfequal = True
    else:
          print("Error: index l does not mactch any existing cases, l = ", l)
          exit()

    ncombinations = 0
    nlinecombs = 0

    nlines_c1c3c4 = 0
    nlines_c2 = 0

    first_non_skipped_c1c3c4_line = None

    use_nchw_formats = [1, 1, 1, 1]

    if config_name_c2 == 'bottleneck_chwn':
        use_nchw_formats[1] = 0

    for line_c1c3c4 in loop_lines[config_name_c1c3c4]:
        # Simple restrictions are here, but more could be applied below
        if   config_name_c1c3c4 == 'bottleneck_nchw_1x1' and not line_c1c3c4.startswith('Aef'):
            continue

        PBFcount_c1c3c4 = line_c1c3c4.count('d')

        if PBFcount_c1c3c4 > 1 and (PBFS == None or (len(PBFS) == 1 and PBFS[0] == [1]) or (len(PBFS) == 4 and PBFS[0] == [1] and PBFS[2] == 1 and PBFS[3] == 1)):
            continue

        if first_non_skipped_c1c3c4_line is None:
            first_non_skipped_c1c3c4_line = line_c1c3c4

        nlines_c1c3c4 = nlines_c1c3c4 + 1

        # For strided 1x1 convolutions, specifically, pack_input must be allowed to be 1
        if stride == 2:
            range_for_pack = range(1, 2)
        else:
            range_for_pack = range(0, pack_input_upfront_limit + 1)
        for pack_input_upfront in range_for_pack:
            for fuse_upd_transposes in range (0, fuse_upd_transposes_limit + 1):

                # Only while debugging
                if stride == 2 and pack_input_upfront == 1 and fuse_upd_transposes == 1:
                    continue

                for use_f32_wt_reduction_and_external_wt_vnni in range (0, use_f32_wt_reduction_and_external_wt_vnni_limit + 1):

                    for line_c2 in loop_lines[config_name_c2]:

                        # Simple restrictions are here, but more could be applied below
                        if config_name_c2 == 'bottleneck_nchw_3x3' and (not line_c2.startswith('A') or not line_c2.endswith('ef')):
                            continue
                        elif config_name_c2 == 'bottleneck_chwn' and not line_c2.lower().startswith('abef'):
                            continue

                        PBFcount_c2 = line_c2.count('d')

                        if PBFcount_c2 > 1 and (PBFS == None or (len(PBFS) == 1 and PBFS[0] == [1]) or (len(PBFS) == 4 and PBFS[1] == [1])):
                            continue

                        if line_c1c3c4 == first_non_skipped_c1c3c4_line and pack_input_upfront == 0 and fuse_upd_transposes == 0 and use_f32_wt_reduction_and_external_wt_vnni == 0:
                            nlines_c2 = nlines_c2 + 1

                        nbfloops = 0

                        if (PBFcount_c1c3c4 == 1 and PBFcount_c2 == 1) or PBFS is None:
                            use_pbfs = None
                        else:
                            use_pbfs = PBFS if len(PBFS) == 4 else [PBFS[0]]*4
                            if PBFcount_c1c3c4 == 1:
                                use_pbfs[0] = [1]
                                use_pbfs[2] = [1]
                                use_pbfs[3] = [1]
                                pbfequal = False
                            if PBFcount_c2 == 1:
                                use_pbfs[1] = [1]
                                pbfequal = False
                        nbfloops = nbfloops + 4

                        if PBFS is not None and len(PBFS) != 1 and len(PBFS) != 4:
                            print("Error: PBFS as a list must have either 1 or 4 elements (sub-lists)")

                        #pf_ranges = [range(0, len(i)) for i in use_pbfs] if use_pbfs is not None else [range(0,1)]*4
                        if use_pbfs is not None:
                            if pbfequal == True:
                                pf_ranges = [range(0, len(use_pbfs[0])), range(0,1), range(0,1), range(0,1)]
                            elif pbfequal_c1c3c4 == True:
                                pf_ranges = [range(0, len(use_pbfs[0])), range(0,1), range(0, len(use_pbfs[0])), range(0, len(use_pbfs[0]))]
                            else:
                                pf_ranges = [range(0, len(i)) for i in use_pbfs]
                        else:
                            pf_ranges = [range(0,1)]*4

                        if downsample == False:
                            pf_ranges[3] = range(0,1)

                        if pack_input_upfront == 0 and fuse_upd_transposes == 0 and use_f32_wt_reduction_and_external_wt_vnni == 0:
                            print("line_c1c3c4 = ", line_c1c3c4)
                            print("line_c2     = ", line_c2)
                            print("PBFcount_c1c3c4 = ", PBFcount_c1c3c4)
                            print("PBFcount_c2     = ", PBFcount_c2)
                            print("range_list = ", [*pf_ranges])

                        for acc_nw in range(0, acc_nw_limit + 1):
                            # For 3x3 s2 chwn -> acc_nw = 0: put into the cpp code
                            # ...
                            if acc_nw == 1 and not line_c2.lower().startswith('abefd'):
                                continue
                            for par_over_h in range(0, par_over_h_limit + 1):
                                if par_over_h == 1 and not "C" in line_c2:
                                    continue
                                # Do we need to have necessarily "c" if par_over_h == 0?
                                # Just while debugging, disabling the option compute_full_wt_output_block = 1
                                for compute_full_wt_output_block in range(0,1): #range(0, compute_full_wt_output_block_limit + 1):
                                    if compute_full_wt_output_block == 1 and par_over_h != 0:
                                        continue

                                    if pack_input_upfront == 0 and fuse_upd_transposes == 0 and use_f32_wt_reduction_and_external_wt_vnni and acc_nw == 0 and par_over_h == 0 and compute_full_wt_output_block == 0:
                                        nlinecombs = nlinecombs + 1

                                    for_recursive(range_list = [*pf_ranges], execute_function = xbf_tester, number_of_loops = nbfloops,
                                                  pbfs=use_pbfs, pbfequal=pbfequal, pbfequal_c1c3c4=pbfequal_c1c3c4, bs = bs,
                                                  nhwck_params=base_sizes, stride = stride, eps = eps, expansion = expansion, has_downsample=downsample, niters=niters,
                                                  pack_input_upfront=pack_input_upfront, fuse_upd_transposes=fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni=use_f32_wt_reduction_and_external_wt_vnni_limit,
                                                  acc_nw=acc_nw, par_over_h=par_over_h, compute_full_wt_output_block=compute_full_wt_output_block,
                                                  hybrid=hybrid, n_img_teams=n_img_teams, n_ofm_teams=n_ofm_teams,
                                                  opt_dtype = torch.bfloat16,
                                                  loop_string_c1c3c4=line_c1c3c4, loop_string_c2=line_c2, use_nchw_formats=use_nchw_formats )
                                    ncombinations = ncombinations + reduce(operator.mul, [len(rangevar) for rangevar in [*pf_ranges]])
                                    print("")
    print("script version, l, config_names, nlines_c1c3c4, nlines_c2, nlinecombs, ncombinations = ", script_version, l, config_name_c1c3c4, config_name_c2, nlines_c1c3c4, nlines_c2, nlinecombs, ncombinations)
exit()


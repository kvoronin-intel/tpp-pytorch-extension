import subprocess
import os
import sys

from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

import torch
import numpy as np
import time

from bottleneck_tuning_bwd_w_driver import run_test_bottleneck

script_version='v1'

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

def xbf_tester(index_list, has_downsample=None, loop_string=None, use_nchw_formats = None,
                nhwck_params=None, bs=None, stride=None, eps=None, expansion=None, use_physical_3x3_padding=True, use_groupnorm=False,
                pack_input_upfront=None, fuse_upd_transposes=None, use_f32_wt_reduction_and_external_wt_vnni=None,
                compute_full_wt_output_block=None,
                hybrid=None, n_img_teams=None, n_ofm_teams=None,
                opt_dtype=torch.float, ref_dtype=torch.float, with_perf=True, with_validation=True, test_module='ext_bottleneck', ref_module='pt_native', niters=20): # **kwargs):
    #print("dbg: index_list = ", index_list)

    # defaults
    p_blocks = [1, 1, 1, 1]

    #print("dbg: pbfs = ", pbfs)
    #print("dbg: p_blocks = ", p_blocks)

    acc_nw = 0
    par_over_h = 0

    tuning_params = [ *p_blocks,
                      *use_nchw_formats,
                      pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
                      acc_nw, par_over_h, compute_full_wt_output_block,
                      hybrid, n_img_teams, n_ofm_teams
                    ]

    tuning_strings = [ loop_string, loop_string, loop_string, loop_string ]

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

loop_names.append('bottleneck_hybrid') # only for 7x7? # AC is a must with 2d parallelization A{C:X} + C{R:Y}
loop_specs.append('A_0_M,' + 'b_0_K,' + 'C_0_M,' + 'd_0_K,' + 'e_0_K,' + 'f_0_K')

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

# teams_pairs = a set of pairs [n_img_teams, n_ofm_teams]
if nthreads == 28:
  teams_pairs = [[14, 2], [7, 4]]
elif nthreads == 56:
  teams_pairs = [[14, 4], [7, 8]]
elif nthreads == 52:
  teams_pairs = [[13, 4]]

for l in [7]:
    #file_path = 'bottleneck_' + str(l) + '_tuning_dbg.txt'
    #sys.stdout = open(file_path, "w")

    # common parameters (potentially overwrite-able)
    N=nthreads
    bs=32
    #bc=32
    #bk=32
    niters=10

    fuse_upd_transposes=0
    use_f32_wt_reduction_and_external_wt_vnni_limit=1
    #acc_nw = 0
    #par_over_h = 0
    # To not deal with the rule that n_img_teams must be 1 for compute_... = 1? which only targets K = 2048
    compute_full_wt_output_block_limit=0
    #hybrid=0
    #n_img_teams=1
    #n_ofm_teams=1
    print("l = ", l)

    if l == 7:
        base_sizes = [N, 7, 7, 2048, 512]
        stride=1
        downsample=False
        KBFS=None #[[1]]
        CBFS=None #[[1]]
        PBFS=None #[[1]]
        hybrid=1
        config_name='bottleneck_hybrid'
    else:
        print("Error: index l does not mactch any existing cases, l = ", l)
        exit()

    ncombinations = 0
    nlinecombs = 0

    nlines = 0

    use_nchw_formats = [1, 1, 1, 1]

    for line in loop_lines[config_name]:
        # Simple restrictions are here, but more could be applied below
        if config_name == 'bottleneck_hybrid' and not line.startswith('AC') and not line.startswith('CA'):
            continue

        nlines = nlines + 1

        # For strided 1x1 convolutions, specifically, pack_input must be allowed to be 1
        if stride == 2:
            range_for_pack = range(1, 2)
        else:
            range_for_pack = range(0, 1)

        for pack_input_upfront in range_for_pack:
            for use_f32_wt_reduction_and_external_wt_vnni in range (0, use_f32_wt_reduction_and_external_wt_vnni_limit + 1):

                pf_ranges = [range(0,1)]*4
                nbfloops = 4

                if downsample == False:
                    pf_ranges[3] = range(0,1)

                if pack_input_upfront == 0 and use_f32_wt_reduction_and_external_wt_vnni == 0:
                    print("line       = ", line)
                    print("range_list = ", [*pf_ranges])

                for teams_pair in teams_pairs:
                    [n_img_teams, n_ofm_teams] = teams_pair

                    # Necessary pre-processing for 2d parallelization
                    if config_name == 'bottleneck_hybrid':
                        # Caution: order of replace() calls matter here!
                        line_with_teams = line
                        line_with_teams = line_with_teams.replace('C', 'C{R:' + str(n_ofm_teams) + '}', 1)
                        line_with_teams = line_with_teams.replace('A', 'A{C:' + str(n_img_teams) + '}', 1)
                        print("modified with teams line = ", line_with_teams)

                    for compute_full_wt_output_block in range(0, compute_full_wt_output_block_limit + 1):

                        if pack_input_upfront == 0 and use_f32_wt_reduction_and_external_wt_vnni and compute_full_wt_output_block == 0:
                            nlinecombs = nlinecombs + 1

                        for_recursive(range_list = [*pf_ranges], execute_function = xbf_tester, number_of_loops = nbfloops,
                                      bs = bs,
                                      nhwck_params=base_sizes, stride = stride, eps = eps, expansion = expansion, has_downsample=downsample, niters=niters,
                                      pack_input_upfront=pack_input_upfront, fuse_upd_transposes=fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni=use_f32_wt_reduction_and_external_wt_vnni_limit,
                                      compute_full_wt_output_block=compute_full_wt_output_block,
                                      hybrid=hybrid, n_img_teams=n_img_teams, n_ofm_teams=n_ofm_teams,
                                      opt_dtype = torch.bfloat16,
                                      loop_string=line_with_teams, use_nchw_formats=use_nchw_formats )
                        ncombinations = ncombinations + reduce(operator.mul, [len(rangevar) for rangevar in [*pf_ranges]])
                        print("")
    print("script version, l, config_names, nlines, nlinecombs, ncombinations = ", script_version, l, config_name, nlines, nlinecombs, ncombinations)
exit()


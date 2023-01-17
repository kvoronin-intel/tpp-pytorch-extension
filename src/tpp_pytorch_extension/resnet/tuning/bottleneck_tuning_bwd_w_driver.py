import argparse
import torch
import numpy as np
import time

#import logging, sys
#logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

import tpp_pytorch_extension

from tpp_pytorch_extension._C import _xsmm as xsmm_cpp

from tpp_pytorch_extension._C import _bottleneck as bottleneck_cpp
import bottleneck as bottleneck_py
from tpp_pytorch_extension._C import _conv as conv_cpp
import conv as conv_py
from tpp_pytorch_extension._C import _batchnorm as batchnorm_cpp
import batchnorm as batchnorm_py

import pcl_cgbp
import pcl_cgbp_cpp

from pcl_cgbp import Bottleneck_base, BottleneckTPP, XsmmBatchNormTPP, XsmmGroupNormTPP, XsmmConv2dTPP

import blocked_layout
from blocked_layout import BlockedTensor, BlockedParameter

import test_utils
from test_utils import compare_weight_grads, compare_padded_tensors

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--test-module', default='tpp_bottleneck', type=str,
                    help='module to test against the reference', dest='test_module')
parser.add_argument('--ref-module', default='pt_native', type=str,
                    help='module to be used as reference', dest='ref_module')

parser.add_argument("--with-perf",       action="store_true", default=True, help='if true, measures performance additionally for the opt module', dest='with_perf')
parser.add_argument("--with-validation", action="store_true", default=False, help='if true, verifies functional corectness for the opt module', dest='with_validation')

parser.add_argument('--use-bf16-opt', action="store_true", default=False, dest='use_bf16_opt')
parser.add_argument('--use-bf16-ref', action="store_true", default=False, dest='use_bf16_ref')

parser.add_argument('--use-physical-3x3-padding', action="store_true", default=True, dest='use_physical_3x3_padding')

parser.add_argument('--use-groupnorm', action="store_true", default=False, dest='use_groupnorm')

parser.add_argument('--block-sizes', nargs="+", type=int, help='block sizes: bc_conv1, bc_conv2, bc_conv3, bk_conv3')

parser.add_argument('--tuning-params', nargs="+", default=None, type=int, help=' (use_nchw_format, fuse_upd_transposes, bf16_acc_nw, par_over_h_pixels, pack_input_upfront, use_intermediate_f32_wt_tensor, use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams, use_f32_wt_reduction_and_external_wt_vnni, compute_full_wt_output_block, pblock) x4 (for each conv) = 48 numbers')
parser.add_argument('--tuning-strings', nargs="+", default=None, type=str, help='conv1_string, conv2_string, conv3_string, conv4_string')

parser.add_argument('--test-data-file', default='resnet50_bottleneck_test_data_28thr.data', type=str,
                    help='file to read test input data from', dest='test_data_file')

parser.add_argument('--basic-sizes', nargs="+", default=None, type=int, help='N H W inc outc stride for the bottleneck')

parser.add_argument('--niters', type=int, default=10, help='number of timed iterations (warmup hardcoded)')
#import pdb

# When physical padding is on, rims can be nans
#torch.autograd.set_detect_anomaly(True)

def gn_init(m, zero_init=False):
    #assert isinstance(m, nn.GroupNorm) or isinstance(m, pcl_cgbp.nn_GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()

def run_test_bottleneck(N, H, W, inc, outc, bc_conv1, bc_conv2, bc_conv3, bk_conv3, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm, opt_dtype, ref_dtype, with_perf, with_validation, test_module, ref_module, tuning_params, tuning_strings, niters):
    time_start = time.time()
    #logging.info("debug: run_test_bottleneck called with N H W inc outc stride eps expansion has_downsample use_physical_3x3_padding use_groupnorm opt_dtype ref_dtype with_perf with_validation, test_module ref_module",
    #        N, H, W, inc, outc, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm, opt_dtype, ref_dtype, with_perf, with_validation, test_module, ref_module)
    print("info: run_test_bottleneck called with N H W inc outc bc_conv1 bc_conv2 bc_conv3 bk_conv3 stride eps expansion has_downsample use_physical_3x3_padding use_groupnorm opt_dtype ref_dtype with_perf with_validation, test_module ref_module niters",
            N, H, W, inc, outc, bc_conv1, bc_conv2, bc_conv3, bk_conv3, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm, opt_dtype, ref_dtype, with_perf, with_validation, test_module, ref_module, niters)
    channel_block_sizes = [bc_conv1, bc_conv2, bc_conv3, bk_conv3]

    tuning_params_count = 48
    if tuning_params is not None and len(tuning_params) != tuning_params_count:
        print("Wrong length of the tuning params (must be " + str(tuning_params_count) + " if present) = " + str(tuning_params) + " " + str(len(tuning_params)))
        exit()

    if with_perf and test_module != 'ext_bottleneck':
        print("Flag with_perf only works with the PCL PT extension bottleneck implementation (check by disabling this early error exit)")
        exit()

    if tuning_params is not None:
        if test_module != 'ext_bottleneck':
            print("Custom tuning params can only be used for ext_bottleneck test_module")
            exit()

        """
        [p1_block, p2_block, p3_block, p4_block,
          c1_use_nchw_format, c2_use_nchw_format, c3_use_nchw_format, c4_use_nchw_format,
          pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
          acc_nw, par_over_h_pixels, compute_full_wt_output_block,
          hybrid, n_img_teams, n_ofm_teams
          ] = tuning_params
        print("info: tuning params: p1_block, p2_block, p3_block, p4_block = ", p1_block, p2_block, p3_block, p4_block)
        print("info: tuning params: c1_use_nchw_format, c2_use_nchw_format, c3_use_nchw_format, c4_use_nchw_format = ", c1_use_nchw_format, c2_use_nchw_format, c3_use_nchw_format, c4_use_nchw_format)
        print("info: tuning params: pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni = ", pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni)
        print("info: tuning params: acc_nw, par_over_h_pixels, compute_full_wt_output_block = ", acc_nw, par_over_h_pixels, compute_full_wt_output_block)
        print("info: tuning params: hybrid, n_img_teams, n_ofm_teams = ", hybrid, n_img_teams, n_ofm_teams)
        """

        [c1_use_nchw_format, c1_fuse_upd_transposes, c1_bf16_acc_nw, c1_par_over_h_pixels, c1_pack_input_upfront, c1_use_intermediate_f32_wt_tensor,
          c1_use_hybrid_imgfm_parallelization, c1_n_img_teams, c1_n_ofm_teams, c1_use_f32_wt_reduction_and_external_wt_vnni, c1_compute_full_wt_output_block, c1_pblock,
          c2_use_nchw_format, c2_fuse_upd_transposes, c2_bf16_acc_nw, c2_par_over_h_pixels, c2_pack_input_upfront, c2_use_intermediate_f32_wt_tensor,
          c2_use_hybrid_imgfm_parallelization, c2_n_img_teams, c2_n_ofm_teams, c2_use_f32_wt_reduction_and_external_wt_vnni, c2_compute_full_wt_output_block, c2_pblock,
          c3_use_nchw_format, c3_fuse_upd_transposes, c3_bf16_acc_nw, c3_par_over_h_pixels, c3_pack_input_upfront, c3_use_intermediate_f32_wt_tensor,
          c3_use_hybrid_imgfm_parallelization, c3_n_img_teams, c3_n_ofm_teams, c3_use_f32_wt_reduction_and_external_wt_vnni, c3_compute_full_wt_output_block, c3_pblock,
          c4_use_nchw_format, c4_fuse_upd_transposes, c4_bf16_acc_nw, c4_par_over_h_pixels, c4_pack_input_upfront, c4_use_intermediate_f32_wt_tensor,
          c4_use_hybrid_imgfm_parallelization, c4_n_img_teams, c4_n_ofm_teams, c4_use_f32_wt_reduction_and_external_wt_vnni, c4_compute_full_wt_output_block, c4_pblock
         ] = tuning_params
        print("info: tuning params  (c1 part 1): use_nchw_format, fuse_upd_transposes, bf16_acc_nw, par_over_h_pixels, pack_input_upfront, use_intermediate_f32_wt_tensor = ", c1_use_nchw_format, c1_fuse_upd_transposes, c1_bf16_acc_nw, c1_par_over_h_pixels, c1_pack_input_upfront, c1_use_intermediate_f32_wt_tensor)
        print("info: tuning params: (c1 part 2): use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams, use_f32_wt_reduction_and_external_wt_vnni, compute_full_wt_output_block, pblock = ", c1_use_hybrid_imgfm_parallelization, c1_n_img_teams, c1_n_ofm_teams, c1_use_f32_wt_reduction_and_external_wt_vnni, c1_compute_full_wt_output_block, c1_pblock)
        print("info: tuning params  (c2 part 1): use_nchw_format, fuse_upd_transposes, bf16_acc_nw, par_over_h_pixels, pack_input_upfront, use_intermediate_f32_wt_tensor = ", c2_use_nchw_format, c2_fuse_upd_transposes, c2_bf16_acc_nw, c2_par_over_h_pixels, c2_pack_input_upfront, c2_use_intermediate_f32_wt_tensor)
        print("info: tuning params: (c2 part 2): use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams, use_f32_wt_reduction_and_external_wt_vnni, compute_full_wt_output_block, pblock = ", c2_use_hybrid_imgfm_parallelization, c2_n_img_teams, c2_n_ofm_teams, c2_use_f32_wt_reduction_and_external_wt_vnni, c2_compute_full_wt_output_block, c2_pblock)
        print("info: tuning params  (c3 part 1): use_nchw_format, fuse_upd_transposes, bf16_acc_nw, par_over_h_pixels, pack_input_upfront, use_intermediate_f32_wt_tensor = ", c3_use_nchw_format, c3_fuse_upd_transposes, c3_bf16_acc_nw, c3_par_over_h_pixels, c3_pack_input_upfront, c3_use_intermediate_f32_wt_tensor)
        print("info: tuning params: (c3 part 2): use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams, use_f32_wt_reduction_and_external_wt_vnni, compute_full_wt_output_block, pblock = ", c3_use_hybrid_imgfm_parallelization, c3_n_img_teams, c3_n_ofm_teams, c3_use_f32_wt_reduction_and_external_wt_vnni, c3_compute_full_wt_output_block, c3_pblock)
        print("info: tuning params  (c4 part 1): use_nchw_format, fuse_upd_transposes, bf16_acc_nw, par_over_h_pixels, pack_input_upfront, use_intermediate_f32_wt_tensor = ", c4_use_nchw_format, c4_fuse_upd_transposes, c4_bf16_acc_nw, c4_par_over_h_pixels, c4_pack_input_upfront, c4_use_intermediate_f32_wt_tensor)
        print("info: tuning params: (c4 part 2): use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams, use_f32_wt_reduction_and_external_wt_vnni, compute_full_wt_output_block, pblock = ", c4_use_hybrid_imgfm_parallelization, c4_n_img_teams, c4_n_ofm_teams, c4_use_f32_wt_reduction_and_external_wt_vnni, c4_compute_full_wt_output_block, c4_pblock)
    else:
        tuning_params = None
        print("info: tuning params are empty")

    if tuning_strings is not None and len(tuning_strings) != 4:
        print("Wrong length of the tuning strings (must be 4 if present) = ", tuning_strings, len(tuning_strings))
        exit()

    if tuning_strings is not None:
        if test_module != 'ext_bottleneck':
            print("Custom tuning strings can only be used for ext_bottleneck test_module")
            exit()
        [c1_loop_string, c2_loop_string, c3_loop_string, c4_loop_string] = tuning_strings
        print("info: tuning strings: c1_string c2_string c3_string c4_string = ", c1_loop_string, c2_loop_string, c3_loop_string, c4_loop_string)
    else:
        tuning_strings = None
        print("info: tuning strings are empty")

    #return

    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        print("One of the modules is using bf16 hence padding input channels to an even number")
        if inc % 2 != 0:
            inc = inc + 1

    if (stride != 1 or inc != outc * expansion) and has_downsample == False:
        print("For these stride, inc, outc and expansion has_downsample must be True", stride, inc, outc, expansion)
        exit()

    if [bc_conv1, bc_conv2, bc_conv3, bk_conv3] != [None, None, None, None] and test_module != 'ext_bottleneck':
        print("Custom block sizes can only be used for ext_bottleneck test_module")
        exit()

    if (stride == 1 and inc == outc * expansion) and has_downsample == True:
        print("Warning: For these values of stride, inc, outc and expansion has_downsample should be False (are you sure you want to have it as True) in Resnet")

    if has_downsample:
        if ref_module == 'pt_native':
            torch.manual_seed(0)
            torch_downsample1 = torch.nn.Conv2d(inc, outc * expansion, kernel_size=1, stride=stride, bias=False, dtype=ref_dtype)
            torch.manual_seed(0)
            if use_groupnorm:
                torch_downsample2 = torch.nn.GroupNorm(32, outc * expansion, eps, dtype=ref_dtype)
                gn_init(torch_downsample2)
            else:
                torch_downsample2 = torch.nn.BatchNorm2d(outc * expansion, eps, dtype=ref_dtype)
        elif ref_module == 'pt_tpp':
            torch.manual_seed(0)
            torch_downsample1 = pcl_cgbp.XsmmConv2dTPP(inc, outc * expansion, kernel_size=1, stride=stride, bias=False, dtype=ref_dtype)
            torch.manual_seed(0)
            if use_groupnorm:
                torch_downsample2 = pcl_cgbp.XsmmGroupNormTPP(32, outc * expansion, eps, dtype=ref_dtype)
                gn_init(torch_downsample2)
            else:
                torch_downsample2 = pcl_cgbp.XsmmBatchNormTPP(outc * expansion, eps, dtype=ref_dtype)

        if test_module == 'tpp_bottleneck' or test_module == 'pt_tpp':
            torch.manual_seed(0)
            opt_downsample1 = pcl_cgbp.XsmmConv2dTPP(inc, outc * expansion, kernel_size=1, stride=stride, bias=False, dtype=opt_dtype)
            torch.manual_seed(0)
            if use_groupnorm:
                opt_downsample2 = pcl_cgbp.XsmmGroupNormTPP(32, outc * expansion, eps, dtype=opt_dtype)
                gn_init(opt_downsample2)
            else:
                opt_downsample2 = pcl_cgbp.XsmmBatchNormTPP(outc * expansion, eps, dtype=opt_dtype)
        elif test_module == 'ext_bottleneck':
            if not use_physical_3x3_padding:
                print("Error, for test_module = ext_bottleneck only physical padding for 3x3 convolutions is supported")
                exit()
            torch.manual_seed(0)
            opt_downsample1 = conv_py.TPPConv2dTPP(inc, outc * expansion, kernel_size=1, stride=stride, bias=False, dtype=opt_dtype, bc=bc_conv1, bk=bk_conv3)
            torch.manual_seed(0)
            if use_groupnorm:
                print("For test_module = ext_bottleneck groupnorm has not been implemented")
                #opt_downsample2 = XsmmGroupNormTPP(32, outc * expansion, eps, dtype=opt_dtype)
                #gn_init(opt_downsample2)
            else:
                #opt_bn = batchnorm_py.TPPBatchNormTPP(C, opt_padding, eps=eps, track_running_stats=track_running_stats, relu=has_relu, eltwise=has_eltwise, dtype=opt_dtype)
                opt_downsample2 = batchnorm_py.TPPBatchNormTPP(outc * expansion, padding=[0, 0, 0, 0],eps=eps, dtype=opt_dtype, bc=bk_conv3)
    else:
        torch_downsample1 = None
        torch_downsample2 = None
        opt_downsample1   = None
        opt_downsample2   = None

    torch.manual_seed(0)
    if ref_module == 'pt_native':
        torch_bottleneck = pcl_cgbp.Bottleneck_base(inc, outc, eps, stride, torch_downsample1, torch_downsample2, use_ref_conv=True, use_ref_norm=True, use_groupnorm=use_groupnorm, dtype=ref_dtype)
    elif ref_module == 'pt_tpp':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        torch_bottleneck = pcl_cgbp.Bottleneck_base(inc, outc, eps, stride, torch_downsample1, torch_downsample2, use_ref_conv=False, use_ref_norm=False, use_groupnorm=use_groupnorm, dtype=ref_dtype)
    elif ref_module == 'tpp_bottleneck':
        print("Warning: not sure that tpp_bottleneck can be used as a ref_module in the standalone bottleneck test")
        torch_bottleneck = pcl_cgbp.BottleneckTPP(inc, outc, eps, stride, use_physical_3x3_padding, opt_downsample1, opt_downsample2, use_groupnorm=use_groupnorm, dtype=opt_dtype)
    else:
        print("ref_module not supported, ref_module = ", ref_module)
        exit()

    time_end = time.time()
    print("Setting up downsample and ref modules took (s) ", time_end - time_start)
    time_start = time.time()

    #logging.debug("Saving initialized PT-based bottleneck")
    print("Saving initialized PT-based bottleneck")
    torch.save(torch_bottleneck.state_dict(), 'checkpoint_ref_bottleneck.pth.tar')

    time_end = time.time()
    print("Saving ref module to the disc took (s) ", time_end - time_start)
    time_start = time.time()

    if test_module == 'ext_bottleneck':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        opt_bottleneck = bottleneck_py.BottleneckTPP(inc, outc, eps, stride, use_physical_3x3_padding, opt_downsample1, opt_downsample2, use_groupnorm=use_groupnorm, dtype=opt_dtype,
                                                     bc_conv1=bc_conv1, bc_conv2=bc_conv2, bc_conv3=bc_conv3, bk_conv3=bk_conv3)
    elif test_module == 'tpp_bottleneck':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        opt_bottleneck = pcl_cgbp.BottleneckTPP(inc, outc, eps, stride, use_physical_3x3_padding, opt_downsample1, opt_downsample2, use_groupnorm=use_groupnorm, dtype=opt_dtype)
    elif test_module == 'pt_tpp':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        opt_bottleneck = pcl_cgbp.Bottleneck_base(inc, outc, eps, stride, opt_downsample1, opt_downsample2, use_ref_conv=False, use_ref_norm=False, use_groupnorm=use_groupnorm, dtype=opt_dtype)
    else:
        print("test_module not supported, test_module = ", test_module)
        exit()

    time_end = time.time()
    print("Creating test module to the disc took (s) ", time_end - time_start)
    time_start = time.time()

    #logging.debug("Loading initialized bottleneck from a checkpoint checkpoint_ref_bottleneck.pth.tar")
    print("Loading initialized bottleneck from a checkpoint checkpoint_ref_bottleneck.pth.tar")
    checkpoint = torch.load('checkpoint_ref_bottleneck.pth.tar')
    opt_bottleneck.load_state_dict(checkpoint)

    time_end = time.time()
    print("Loading ref module into the test module to the disc took (s) ", time_end - time_start)
    time_start = time.time()

    torch.manual_seed(0)

    x = torch.randn(N, inc, H, W, requires_grad=True)
    #x = torch.ones_like(x, requires_grad=True)
    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        x_bf16 = x.to(torch.bfloat16)
        x      = x_bf16.to(torch.float)

    if opt_dtype == torch.bfloat16:
        opt_x_init = x_bf16
    else:
        opt_x_init = x

    if ref_dtype == torch.bfloat16:
        ref_x_init = x_bf16
    else:
        ref_x_init = x

    x1 = opt_x_init.clone().detach().requires_grad_()
    x1.retain_grad()
    x2 = ref_x_init.clone().detach().requires_grad_()

    time_end = time.time()
    print("Setting up input vectors took (s) ", time_end - time_start)
    time_start = time.time()

    basic_params_string = str(N) + " " + str(H) + " " + str(W) + " " + str(inc) + " " + str(outc) + " " + str(stride) + " " + str(has_downsample)
    print("Bottleneck tuning parameters ", " basic: " + basic_params_string + " channel bs: " + str(channel_block_sizes) + " tuning params: "+ str(tuning_params) + " tuning_strings: " + str(tuning_strings))

    #logging.debug("running opt bottleneck forward")
    print("running opt bottleneck forward")
    if test_module == 'ext_bottleneck':
        print("ext_bottleneck forward is called with tuning_params and tuning_strings")
        #dummy_tuning_timings = [0.0]*16
        dummy_tuning_timings = np.zeros(16, dtype=np.float32)
        y1 = opt_bottleneck(x1, tuning_params_w=tuning_params, tuning_strings_w=tuning_strings, tuning_timings_bwd=dummy_tuning_timings)
    else:
        y1 = opt_bottleneck(x1)

    z1 = y1.mean()
    #z1.backward(retain_graph=True)

    time_end = time.time()
    print("First forward took (s) ", time_end - time_start)
    time_start = time.time()

    if with_validation:
        rtol=1.5e-1
        atol=1e+0

        #logging.debug("running ref bottleneck forward")
        print("running ref bottleneck forward")
        y2 = torch_bottleneck(x2)

        z2 = y2.mean()
        #z2.backward(retain_graph=True)

        time_end = time.time()
        print("Reference forward took (s) ", time_end - time_start)
        time_start = time.time()

        print("z1=", z1)
        print("z2=", z2)
        #logging.info('z1=%s', z1)
        #logging.info('z2=%s', z2)

        # Y (Out)

        validation_check_failed1 = not compare_padded_tensors(y1.unblocked_tensor(), y2, "Y (Out)", rtol=rtol, atol=atol)

        z1.backward(retain_graph=True)
        z2.backward(retain_graph=True)

        # X gradient

        validation_check_failed2 = not compare_padded_tensors(x1.grad, x2.grad, "X Grad", rtol=rtol, atol=atol)

        # Weight gradients for batchnorms

        validation_check_failed3 = not compare_weight_grads( opt_bottleneck.bn3.weight.grad, torch_bottleneck.bn3.weight.grad, "bn3", rtol=rtol, atol=atol)
        validation_check_failed4 = not compare_weight_grads( opt_bottleneck.bn2.weight.grad, torch_bottleneck.bn2.weight.grad, "bn2", rtol=rtol, atol=atol)
        validation_check_failed5 = not compare_weight_grads( opt_bottleneck.bn1.weight.grad, torch_bottleneck.bn1.weight.grad, "bn1", rtol=rtol, atol=atol)
        if has_downsample:
            validation_check_failed6 = not compare_weight_grads( opt_bottleneck.downsample2.weight.grad, torch_bottleneck.downsample2.weight.grad, "bn4", rtol=rtol, atol=atol)
        else:
            validation_check_failed6 = False

        # Weight gradients for convs

        validation_check_failed7 = not compare_weight_grads( opt_bottleneck.conv3.weight.grad, torch_bottleneck.conv3.weight.grad, "conv3", rtol=rtol, atol=atol)
        validation_check_failed8 = not compare_weight_grads( opt_bottleneck.conv2.weight.grad, torch_bottleneck.conv2.weight.grad, "conv2", rtol=rtol, atol=atol)
        validation_check_failed9 = not compare_weight_grads( opt_bottleneck.conv1.weight.grad, torch_bottleneck.conv1.weight.grad, "conv1", rtol=rtol, atol=atol)
        if has_downsample:
            validation_check_failed10 = not compare_weight_grads( opt_bottleneck.downsample1.weight.grad, torch_bottleneck.downsample1.weight.grad, "conv4", rtol=rtol, atol=atol)
        else:
            validation_check_failed10 = False

        validation_checks_failed = validation_check_failed1 or validation_check_failed2 or validation_check_failed3 or validation_check_failed4 or validation_check_failed5 or validation_check_failed6 or validation_check_failed7 or validation_check_failed8 or validation_check_failed9 or validation_check_failed10

        print("validation_check_failed1 = ", validation_check_failed1)
        print("validation_check_failed2 = ", validation_check_failed2)
        print("validation_check_failed3 = ", validation_check_failed3)
        print("validation_check_failed4 = ", validation_check_failed4)
        print("validation_check_failed5 = ", validation_check_failed5)
        print("validation_check_failed6 = ", validation_check_failed6)
        print("validation_check_failed7 = ", validation_check_failed7)
        print("validation_check_failed8 = ", validation_check_failed8)
        print("validation_check_failed9 = ", validation_check_failed9)
        print("validation_check_failed10 = ", validation_check_failed10)

        #validation_checks_failed = validation_check_failed1
        if validation_checks_failed:
            print("Validation FAILED")
        else:
            print("Validation PASSED")

    #return
    #exit()

    if with_perf:

        bottleneck_cfg = opt_bottleneck.config

        blocked_input = opt_bottleneck.get_blocked_tensor(
            x1,
            opt_bottleneck.blocked_input_signature,
            [None, opt_bottleneck.conv1.Cblock, None, None],
        )

        time_end = time.time()
        print("Reference forward took (s) ", time_end - time_start)
        time_start = time.time()

        conv1_out = torch.randn(N, outc, H, W, requires_grad=False, dtype=opt_dtype)
        bn1_out   = torch.nn.functional.pad(conv1_out, [1, 1, 1, 1], mode='constant', value=0.0) #torch.randn(N, outc, H+2, W+2, requires_grad=False, dtype=opt_dtype)
        if stride == 1:
            conv2_out = bn1_out.clone()
        else:
            tmp = torch.randn(N, outc, H//stride, W//stride, requires_grad=False, dtype=opt_dtype)
            conv2_out = torch.nn.functional.pad(tmp, [1, 1, 1, 1], mode='constant', value=0.0)
        if stride == 1:
            bn2_out = conv1_out.clone()#torch.randn(N, outc, H, W, requires_grad=False, dtype=opt_dtype)
        else:
            bn2_out = torch.randn(N, outc, H//stride, W//stride, requires_grad=False, dtype=opt_dtype)
        conv3_out = torch.randn(N, outc*expansion, H//stride, W//stride, requires_grad=False, dtype=opt_dtype)
        bn3_out   = conv3_out.clone() #torch.randn_like(conv3_out)
        conv4_out = conv3_out.clone() #torch.randn_like(conv3_out)
        bn4_out   = conv3_out.clone() #torch.randn_like(conv3_out)

        bn1_relu_out = bn1_out.to(torch.uint8)
        bn2_relu_out = bn2_out.to(torch.uint8)
        bn3_relu_out = bn3_out.to(torch.uint8)
        bn4_relu_out = bn4_out.to(torch.uint8)

        bn1_scratch    = torch.empty(10, N, outc, requires_grad=False, dtype=torch.float32)
        bn2_scratch    = torch.empty(10, N, outc, requires_grad=False, dtype=torch.float32)
        bn3_scratch    = torch.empty(10, N, outc*expansion, requires_grad=False, dtype=torch.float32)
        bn4_scratch    = torch.empty(10, N, outc*expansion, requires_grad=False, dtype=torch.float32)

        artificial_tensors = []
        for t in [conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out]:
            blocked_t = opt_bottleneck.get_blocked_tensor(
                t,
                opt_bottleneck.blocked_input_signature,
                [None, opt_bottleneck.conv1.Cblock, None, None],
            )
            artificial_tensors.append(blocked_t)

        artificial_tensors += [bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
                              bn1_scratch, bn2_scratch, bn3_scratch, bn4_scratch]

        #artificial_tensors = [conv1_out]

        time_end = time.time()
        print("Creating artifical tensors for bwd took (s) ", time_end - time_start)

        print("y1.blocked_tensor() shape = ", y1.blocked_tensor().shape)
        print("y1.unblocked_tensor() shape = ", y1.unblocked_tensor().shape)

        if opt_bottleneck.has_residual_conv:
            inputs = [y1.blocked_tensor(), blocked_input,
                      opt_bottleneck.conv1.weight, opt_bottleneck.conv2.weight, opt_bottleneck.conv3.weight, opt_bottleneck.downsample1.weight,
                      opt_bottleneck.bn1.weight, opt_bottleneck.bn2.weight, opt_bottleneck.bn3.weight, opt_bottleneck.downsample2.weight,
                      opt_bottleneck.bn1.bias, opt_bottleneck.bn2.bias, opt_bottleneck.bn3.bias, opt_bottleneck.downsample2.bias,
                      opt_bottleneck.bn1.mean, opt_bottleneck.bn2.mean, opt_bottleneck.bn3.mean, opt_bottleneck.downsample2.mean,
                      opt_bottleneck.bn1.var, opt_bottleneck.bn2.var, opt_bottleneck.bn3.var, opt_bottleneck.downsample2.var, *artificial_tensors]
        else:
            inputs = [y1.blocked_tensor(), blocked_input,
                      opt_bottleneck.conv1.weight, opt_bottleneck.conv2.weight, opt_bottleneck.conv3.weight, opt_bottleneck.dummy_tensor,
                      opt_bottleneck.bn1.weight, opt_bottleneck.bn2.weight, opt_bottleneck.bn3.weight, opt_bottleneck.dummy_tensor,
                      opt_bottleneck.bn1.bias, opt_bottleneck.bn2.bias, opt_bottleneck.bn3.bias, opt_bottleneck.dummy_tensor,
                      opt_bottleneck.bn1.mean, opt_bottleneck.bn2.mean, opt_bottleneck.bn3.mean, opt_bottleneck.dummy_tensor,
                      opt_bottleneck.bn1.var, opt_bottleneck.bn2.var, opt_bottleneck.bn3.var, opt_bottleneck.dummy_tensor, *artificial_tensors]

        warmup_niter = 10
        #logging.info("warmup_niter = ", warmup_niter)
        print("warmup_niter = ", warmup_niter)

        #dummy_tuning_timings = [0.0] * 16
        dummy_tuning_timings = np.zeros(16, dtype=np.float32)
        time_start = time.time()

        print("type of bottleneck_cfg = ", type(bottleneck_cfg))
        print("type of inputs = ", type(inputs))
        print("tuning_params = ", type(tuning_params))
        print("tuning_strings = ", type(tuning_strings))
        print("dummy_tuning_timings = ", type(dummy_tuning_timings))

        for i in range(warmup_niter):
            if tuning_params is None or tuning_strings is None or len(tuning_params) == 0 or len(tuning_strings) == 0 or dummy_tuning_timings is None:
                bottleneck_cpp.bottleneck_bn_bwd(bottleneck_cfg, inputs)
            else:
                bottleneck_cpp.bottleneck_bn_bwd_w_ext(bottleneck_cfg, inputs, tuning_params, tuning_strings, dummy_tuning_timings)

        time_end = time.time()
        print("Warmup took (s) ", time_end - time_start)

        timed_niters = niters
        #logging.info("timed_niters = ", timed_niters)
        print("timed_niters = ", timed_niters)

        #tuning_timings = [0.0] * 16
        tuning_timings = np.zeros(16, dtype=np.float32)
        print("dbg:tuning_timings before: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        time_start = time.time()
        for i in range(timed_niters):
            if tuning_params is None or tuning_strings is None or len(tuning_params) == 0 or len(tuning_strings) == 0 or tuning_timings is None:
                bottleneck_cpp.bottleneck_bn_bwd(bottleneck_cfg, inputs)
            else:
                bottleneck_cpp.bottleneck_bn_bwd_w_ext(bottleneck_cfg, inputs, tuning_params, tuning_strings, tuning_timings)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / timed_niters

        print("dbg:tuning_timings after: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        print("Timed loop took (s) ", time_end - time_start)
        print("Final perf time: ", time_per_iter)
        gflop = bottleneck_cpp.bottleneck_bn_bwd_w_get_gflop(bottleneck_cfg)
        basic_params_string = str(N) + " " + str(H) + " " + str(W) + " " + str(inc) + " " + str(outc) + " " + str(stride) + " " + str(has_downsample)
        print("Final perf GFLOPs: ", str(gflop/time_per_iter) + " basic: " + basic_params_string + " channel bs: " + str(channel_block_sizes) + " tuning params: "+ str(tuning_params) + " tuning_strings: " + str(tuning_strings))

        # Checking the timings
        gflop_details = bottleneck_cpp.bottleneck_bn_bwd_w_get_gflop_details(bottleneck_cfg)
        #print("timings c1 gflop_c1 c2 gflop_c2 c3 gflop_c3 c4 gflop_c4: ", tuning_timings[0], gflop_details[0], tuning_timings[1], gflop_details[1], tuning_timings[2], gflop_details[2], tuning_timings[3], gflop_details[3])
        print("timings: c1 gflop_c1 gflops_c1: ", tuning_timings[8], gflop_details[0], gflop_details[0] / (tuning_timings[8] / timed_niters) if tuning_timings[8] != 0.0 else 0.0)
        print("timings: c2 gflop_c2 gflops_c2: ", tuning_timings[9], gflop_details[1], gflop_details[1] / (tuning_timings[9] / timed_niters) if tuning_timings[9] != 0.0 else 0.0)
        print("timings: c3 gflop_c3 gflops_c3: ", tuning_timings[10], gflop_details[2], gflop_details[2] / (tuning_timings[10] / timed_niters) if tuning_timings[10] != 0.0 else 0.0)
        print("timings: c4 gflop_c4 gflops_c4: ", tuning_timings[11], gflop_details[3], gflop_details[3] / (tuning_timings[11] / timed_niters) if tuning_timings[11] != 0.0 else 0.0)
        #print("timings b1 gflop_b1 b2 gflop_b2 b3 gflop_b3 b4 gflop_b4: ", tuning_timings[4], gflop_details[4], tuning_timings[5], gflop_details[5], tuning_timings[6], gflop_details[6], tuning_timings[7], gflop_details[7])
        print("timings: b1 gflop_b1 gflops_b1: ", tuning_timings[4], gflop_details[4], gflop_details[4] / (tuning_timings[4] / timed_niters) if tuning_timings[4] != 0.0 else 0.0)
        print("timings: b2 gflop_b2 gflops_b2: ", tuning_timings[5], gflop_details[5], gflop_details[5] / (tuning_timings[5] / timed_niters) if tuning_timings[5] != 0.0 else 0.0)
        print("timings: b3 gflop_b3 gflops_b3: ", tuning_timings[6], gflop_details[6], gflop_details[6] / (tuning_timings[6] / timed_niters) if tuning_timings[6] != 0.0 else 0.0)
        print("timings: b4 gflop_b4 gflops_b4: ", tuning_timings[7], gflop_details[7], gflop_details[7] / (tuning_timings[7] / timed_niters) if tuning_timings[7] != 0.0 else 0.0)

        sum_timings = tuning_timings[8] + tuning_timings[9] + tuning_timings[10] + tuning_timings[11] + tuning_timings[4] + tuning_timings[5] + tuning_timings[6] + tuning_timings[7]
        print("timing diff (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

        #print("Error: performance part is not implemented for this test!")
        #exit()

    return
    #exit()

def main():
    xsmm_cpp.init_libxsmm()
    #pcl_cgbp.init_libxsmm()

    #print("block_sizes = ", args.block_sizes, len(args.block_sizes))
    #print("tuning_params = ", args.tuning_params, len(args.tuning_params))

    #logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    opt_dtype = torch.float if not args.use_bf16_opt else torch.bfloat16
    ref_dtype = torch.float if not args.use_bf16_ref else torch.bfloat16

    if args.block_sizes is not None:
        [bc_conv1, bc_conv2, bc_conv3, bk_conv3] = args.block_sizes
    else:
        [bc_conv1, bc_conv2, bc_conv3, bk_conv3] = [None, None, None, None]

    #with open("resnet50_bottleneck_test_data_28thr.data") as f:
    #with open("resnet50_bottleneck_test_data_28thr_dbg.data") as f:
    #with open("resnet50_bottleneck_test_data_28thr_saved.data") as f:
    if args.basic_sizes is not None:
        if len(args.basic_sizes) != 6:
            print("Error: basic sizes must have exactly 6 elements if defined (N, H, W, inc, outc, stride)")
            exit()
        [N, H, W, inc, outc, stride] = args.basic_sizes
        eps = 1e-7
        expansion = 4
        #if (stride != 1 or inc != outc * expansion) and has_downsample == False:
        #    print("For these stride, inc, outc and expansion has_downsample must be True")
        #    exit()
        if (stride != 1 or inc != outc * expansion):
            has_downsample = True
        else:
            has_downsample = False
        run_test_bottleneck(N, H, W, inc, outc, bc_conv1, bc_conv2, bc_conv3, bk_conv3, stride, eps, expansion, has_downsample, args.use_physical_3x3_padding, args.use_groupnorm,
                            opt_dtype, ref_dtype, args.with_perf, args.with_validation, args.test_module, args.ref_module, args.tuning_params, args.tuning_strings, args.niters)
    else:
        with open(args.test_data_file) as f:
            contents = f.readlines()
            for line in contents:
                if line[0] == '#' or len(line) < 2:
                    continue
                preprocessed_line = " ".join(line.strip().split()) # to remove extra spaces in the input line
                [N, H, W, inc, outc, stride, expansion, has_downsample, eps] = list(preprocessed_line.split(" ")) #list(line.split(" "))
                has_downsample = False if has_downsample.strip() == 'False' else True
                string_list = list(preprocessed_line.split(" ")) #list(line.strip().split(" "))
                integer_map = map(int, string_list[:7])
                [N, H, W, inc, outc, stride, expansion] = list(integer_map)
                eps = float(eps)
                run_test_bottleneck(N, H, W, inc, outc, bc_conv1, bc_conv2, bc_conv3, bk_conv3, stride, eps, expansion, has_downsample, args.use_physical_3x3_padding, args.use_groupnorm,
                                    opt_dtype, ref_dtype, args.with_perf, args.with_validation, args.test_module, args.ref_module, args.tuning_params, args.tuning_strings, args.niters)
    exit()

    # Just a single size run
    inc=64
    outc=64
    stride=1
    eps = 1e-7

    N=28
    H=56
    W=56

    expansion = 4 # Fixed
    has_downsample = True

    run_test_bottleneck(N, H, W, inc, outc, stride, eps, expansion, has_downsample, args.use_physical_3x3_padding, args.use_groupnorm, opt_dtype, ref_dtype,
                        args.with_perf, args.with_validation, args.test_module, args.ref_module, args.tuning_params, args.tuning_strings, args.niters)

if __name__ == "__main__":
    args = parser.parse_args()
    main()

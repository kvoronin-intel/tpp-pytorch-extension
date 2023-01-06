import argparse
import time
import torch
#import pcl_cgbp
#import pcl_cgbp_cpp
import numpy as np

import tpp_pytorch_extension
from tpp_pytorch_extension._C import _batchnorm as batchnorm_cpp
from tpp_pytorch_extension.resnet import batchnorm as batchnorm_py

import pcl_cgbp
import pcl_cgbp_cpp

import test_utils
from test_utils import compare_weight_grads, compare_padded_tensors

"""
import sys, inspect

def print_classes():
    for name, obj in inspect.getmembers(sys.modules['tpp_pytorch_extension']):
        print(obj)
        #if inspect.isclass(obj):
        #    print(obj)

print_classes()

exit()
"""

#from batchnorm_py import TPPBatchNormTPP

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--test-module', default='cnn_tpp', type=str,
                    help='module to test against the reference', dest='test_module')

parser.add_argument("--with-perf", action="store_true", default=False, help='if true, measures performance additionally for the opt module', dest='with_perf')

parser.add_argument('--use-bf16-opt', action="store_true", default=False, dest='use_bf16_opt')
parser.add_argument('--use-bf16-ref', action="store_true", default=False, dest='use_bf16_ref')

parser.add_argument('--bc',  nargs='?', type=int)

parser.add_argument('--tuning-string-ncp', default=None, type=str, help='tuning string for ncp loop')
parser.add_argument('--tuning-string-cp', default=None, type=str, help='tuning string for cp loop')

parser.add_argument('--test-data-file', default='resnet50_bn_test_data_56thr.data', type=str,
                    help='file to read test input data from', dest='test_data_file')

parser.add_argument('--basic-sizes', nargs="+", default=None, type=int, help='N, H, W, C, has_relu, has_eltwise, track_running_stats, pad_in, pad_out for the bn')

parser.add_argument('--niters', type=int, default=100, help='number of timed iterations')
parser.add_argument('--niters-warmup', type=int, default=10, help='number of warmup iterations')

parser.add_argument("--perf-fwd", action="store_true", default=False, help='if true, runs forward perf', dest='perf_fwd')

parser.add_argument("--scale-only", action="store_true", default=False, help='if true, runs only scale part of the batchnorm in fwd perf', dest='scale_only')

torch.autograd.set_detect_anomaly(True)

def run_test_bn(N, H, W, C, bc, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, with_perf, test_module, tuning_string_ncp, tuning_string_cp, niters, niters_warmup, perf_fwd, scale_only):
    print("debug: run_test_bn called with N H W C bc, opt_padding has_relu has_eltwise track_running_stats opt_dtype ref_dtype with_perf test_module tuning_string_ncp tuning_string_cp niters niters_warmup perf_fwd scale_only",
            N, H, W, C, bc, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, with_perf, test_module, tuning_string_ncp, tuning_string_cp, niters, niters_warmup, perf_fwd, scale_only)

    eps=1.e-7

    #if (perf_fwd and perf_bwd_w) or (perf_fwd and perf_bwd_d) or (perf_bwd_d and perf_bwd_w):
    #    print("Error: only one of perf-fwd, perf-bwd-w and perf-bwd-d can be active")
    #    exit()

    if tuning_string_ncp is not None or tuning_string_cp is not None:
        if test_module != 'ext_tpp':
            print("Custom tuning strings can only be used for ext_tpp test_module")
            exit()
        ncp_loop_string = tuning_string_ncp
        cp_loop_string  = tuning_string_cp
        print("info: tuning string: ncp_string = ", ncp_loop_string)
        print("info: tuning string:  cp_string = ", cp_loop_string)
    else:
        tuning_string_ncp = None
        tuning_string_cp  = None
        print("info: tuning strings are empty")

    #opt_padding = [0, 0, 0, 0]
    if opt_padding != None and len(opt_padding) != 4:
        print("Error: padding should have four elements [pad_h_in, pad_w_in, pad_h_out, pad_w_out]")
        exit()

    if bc != None and (test_module != 'ext_tpp' and test_module != 'cnn_tpp'):
        print("Custom block sizes can only be used for ext_tpp and cnn_tpp test_modules")
        exit()

    if opt_padding != None:
        input_hw_padding  = [opt_padding[0], opt_padding[0], opt_padding[1], opt_padding[1]]
        output_hw_padding = [opt_padding[2], opt_padding[2], opt_padding[3], opt_padding[3]]
        print("input_hw_padding = ",  input_hw_padding)
        print("output_hw_padding = ", output_hw_padding)

    torch.manual_seed(0)
    if test_module == 'cnn_tpp':
        print("info: testing TPP module from CNN (pcl_cgbp)")
        if opt_padding != None and opt_padding != [0, 0, 0, 0]:
            print("Error: Python side of batchnorm in cnn_tpp does not support padding")
            exit()
        opt_bn = pcl_cgbp.XsmmBatchNormTPP(C, eps=eps, track_running_stats=track_running_stats, relu=has_relu, eltwise=has_eltwise, dtype=opt_dtype, bc=bc)
    elif test_module == 'ext_tpp':
        print("info: testing TPP module from extensions (tpp_pytorch_extension)")
        opt_bn = batchnorm_py.TPPBatchNormTPP(C, opt_padding, eps=eps, track_running_stats=track_running_stats, relu=has_relu, eltwise=has_eltwise, dtype=opt_dtype, bc=bc)
    else:
        print("test_module not supported, test_module = ", test_module)
        exit()
    torch.manual_seed(0)
    torch_bn   = torch.nn.BatchNorm2d(C, eps=eps, track_running_stats=track_running_stats, device=None, dtype=ref_dtype)
    torch_relu = torch.nn.ReLU()

    torch.manual_seed(0)
    weight = torch.randn(C, requires_grad=True)
    # Would be a mistake for bn since weight and bias remain fp32 even for bf16 activations
    #if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
    #    weight_bf16 = weight.to(torch.bfloat16)
    #    weight      = weight_bf16.to(torch.float)

    bias   = torch.randn(C, requires_grad=True)
    # Would be a mistake for bn since weight and bias remain fp32 even for bf16 activations
    #if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
    #    bias_bf16 = bias.to(torch.bfloat16)
    #    bias      = bias_bf16.to(torch.float)

    #x = torch.randn(N, C, H, W, requires_grad=True)
    #x = torch.randn(N, C, H, W, requires_grad=True) * 0.5 + 4.0
    #x = torch.ones_like(x, requires_grad=True)
    rates = torch.rand(N, C, H, W) * 5
    x = torch.poisson(rates)
    x.requires_grad_()
    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        x_bf16 = x.to(torch.bfloat16)
        x      = x_bf16.to(torch.float)

    x_add = torch.randn(N, C, H, W, requires_grad=True)
    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        x_add_bf16 = x_add.to(torch.bfloat16)
        x_add      = x_add_bf16.to(torch.float)

    if opt_dtype == torch.bfloat16:
        opt_x_init = x_bf16
        if has_eltwise == True:
            opt_x_add_init = x_add_bf16
    else:
        opt_x_init = x
        if has_eltwise == True:
            opt_x_add_init = x_add

    if opt_padding != None:
        opt_x_init = torch.nn.functional.pad(opt_x_init,             input_hw_padding, mode='constant', value=0.0)
        if has_eltwise == True:
            opt_x_add_init = torch.nn.functional.pad(opt_x_add_init, input_hw_padding, mode='constant', value=0.0)

    opt_weight_init = weight
    opt_bias_init = bias

    if ref_dtype == torch.bfloat16:
        ref_x_init = x_bf16
        if has_eltwise == True:
            ref_x_add_init = x_add_bf16
    else:
        ref_x_init = x
        if has_eltwise == True:
            ref_x_add_init = x_add

    ref_weight_init = weight
    ref_bias_init = bias

    x1 = opt_x_init.clone().detach().requires_grad_()
    x1.retain_grad()
    x2 = ref_x_init.clone().detach().requires_grad_()

    if has_eltwise == True:
        x1_add = opt_x_add_init.clone().detach().requires_grad_()
        x1_add.retain_grad()
        x2_add = ref_x_add_init.clone().detach().requires_grad_()

    opt_bn.weight.data   = opt_weight_init.clone() # blocked_layout.py is taking care of blocking
    #print("opt_conv_weight shape = ", opt_conv.weight.shape)
    #opt_bn.weight.block()
    torch_bn.weight.data = ref_weight_init.clone()

    opt_bn.bias.data   = opt_bias_init.clone()
    torch_bn.bias.data = ref_bias_init.clone()

     # should be available if setup script has been called (once implemented)
    bc = opt_bn.Cblock
    print("Info: bc = ", bc)

    """
    if x1.dim() == 4:
      #size = [x1.size(0), 1, x1.size(1), x1.size(2), x1.size(3)]
      opt_input_size = [x1.size(0), x1.size(1)//bc, bc, x1.size(2), x1.size(3)]
      #print("got here, size")
      #print(size)
      xp = x1.view(opt_input_size).permute([0,1,3,4,2]).contiguous()
      if has_eltwise == True:
          xp_add = x1_add.view(opt_input_size).permute([0,1,3,4,2]).contiguous()

    x1.retain_grad()
    if has_eltwise == True:
        xp_add.retain_grad()
    print("x2 (torch) size, xp size = ", x2.size(), xp.size())
    """

    #y1 = opt_conv(x1, x1_add)
    #y2 = relu(torch_conv(x2) + x2_add)
    dummy_tuning_timings = np.zeros(16, dtype=np.float32)
    if has_eltwise == True:
        #y1 = opt_bn(xp, xp_add)
        if perf_fwd:
            y1 = opt_bn(x1, x1_add, tuning_string_ncp=tuning_string_ncp, tuning_string_cp=tuning_string_cp, tuning_timings=dummy_tuning_timings)
        else:
            y1 = opt_bn(x1, x1_add)
    else:
        #y1 = opt_bn(xp)
        if perf_fwd:
            y1 = opt_bn(x1, tuning_string_ncp=tuning_string_ncp, tuning_string_cp=tuning_string_cp, tuning_timings=dummy_tuning_timings)
        else:
            y1 = opt_bn(x1)

    if has_relu == False and has_eltwise == False:
        y2 = torch_bn(x2)
    elif has_relu == True and has_eltwise == False:
        y2 = torch_relu(torch_bn(x2))
    elif has_relu == True and has_eltwise == True:
        y2 = torch_relu(torch_bn(x2) + x2_add)
    else:
        print("Not implemented, has_relu has_eltwise = ", has_relu, has_eltwise)
        exit()

    print("y1.shape = ", y1.shape)
    print("y2.shape = ", y2.shape)

    z1 = y1.mean()
    z2 = y2.mean()
    #z1 = y1.sum()
    #z2 = y2.sum()
    if opt_padding != None:
        y1_numel = y1.unblocked_tensor().numel() # if hasattr(y1,unblocked_tensor) else y1.numel() does not work as a check
        print("z1 with account for padding", z1 * y1_numel / y2.numel())
    else:
        y1_numel = y1.unblocked_tensor().numel() # if hasattr(y1,unblocked_tensor) else y1.numel() does not work as a check
        print("z1", z1)
    print("z2", z2)
    #z1.backward(retain_graph=True)
    z1.backward(retain_graph=True,gradient=torch.tensor(1.*y1_numel / y2.numel(), dtype=y1.dtype))
    z2.backward(retain_graph=True)
    """
    grad_scale=1000.0
    enhanced_gradient = torch.tensor(grad_scale, dtype=torch.float)
    z1.backward(retain_graph=True, gradient=enhanced_gradient)
    z2.backward(retain_graph=True, gradient=enhanced_gradient)
    #z1.backward(gradient=enhanced_gradient)
    #z2.backward(gradient=enhanced_gradient)
    """

    """
    opt_x_grad = x1.grad.to(torch.float)
    if has_eltwise:
        opt_x_add_grad = x1_add.grad.to(torch.float)
    opt_weight_grad = opt_bn.weight.grad.to(torch.float)
    opt_bias_grad = opt_bn.bias.grad.to(torch.float)
    opt_y_fp32 = y1.unblocked_tensor().to(torch.float)
    """

    """
    if opt_dtype == torch.bfloat16:
        opt_x_grad = x1.grad.to(torch.float)
        if has_eltwise:
            opt_x_add_grad = x1_add.grad.to(torch.float)
        opt_weight_grad = opt_bn.weight.grad.to(torch.float)
        opt_bias_grad = opt_bn.bias.grad.to(torch.float)
        opt_y_fp32 = y1.unblocked_tensor().to(torch.float)
    else:
        opt_x_grad = x1.grad
        if has_eltwise:
            opt_x_add_grad = x1_add.grad
        opt_y_fp32 = y1.unblocked_tensor()
    """

    """
    opt_weight_grad = opt_bn.weight.grad
    opt_bias_grad = opt_bn.bias.grad

    ref_x_grad = x2.grad.to(torch.float)
    if has_eltwise:
        ref_x_add_grad = x2_add.grad.to(torch.float)
    ref_y_fp32 = y2.to(torch.float)

    if opt_padding != None:
        ref_y_fp32 = torch.nn.functional.pad(ref_y_fp32,         output_hw_padding, mode='constant', value=0.0)
        ref_x_grad = torch.nn.functional.pad(ref_x_grad,         input_hw_padding,  mode='constant', value=0.0)
        if has_eltwise:
            ref_x_add_grad = torch.nn.functional.pad(ref_x_add_grad, input_hw_padding,  mode='constant', value=0.0)
    """

    """
    if ref_dtype == torch.bfloat16:
        ref_x_grad = x2.grad.to(torch.float)
        if has_eltwise:
            ref_x_add_grad = x2_add.grad.to(torch.float)
        ref_weight_grad = torch_bn.weight.grad.to(torch.float)
        ref_bias_grad = torch_bn.bias.grad.to(torch.float)
        ref_y_fp32 = y2.to(torch.float)
    else:
        ref_x_grad = x2.grad
        if has_eltwise:
            ref_x_add_grad = x2_add.grad
        ref_y_fp32 = y2
    """
    """
    ref_weight_grad = torch_bn.weight.grad
    ref_bias_grad = torch_bn.bias.grad
    """

    # Very loose tolerances to check only obvious errors
    if opt_dtype == torch.bfloat16:
        rtol=5e-2 #1.5e-1
        atol=1e-2 #1e+0
    else: # not checked
        rtol=1e-3 #1.5e-1
        atol=1e-3 #1e+0

    ignored_failures = 0

    validation_check_x_grad_failed = not compare_padded_tensors(x1.grad, x2.grad, "X Grad", W, input_hw_padding, rtol=rtol, atol=atol)
    if x2.grad.norm(p=float('inf')) < 1e-10 and validation_check_x_grad_failed:
        print("Warning: ignoring the failed tensor comparison results as the reference tensor is too small for x grad")
        ignored_failures = ignored_failures + 1
        validation_check_x_grad_failed = False

    # X_ADD gradient
    if has_eltwise == True:
        validation_check_xadd_grad_failed = not compare_padded_tensors(x1_add.grad, x2_add.grad, "X ADD Grad", W, input_hw_padding, rtol=rtol, atol=atol)
        if x2_add.grad.norm(p=float('inf')) < 1e-10 and validation_check_xadd_grad_failed:
            print("Warning: ignoring the failed tensor comparison results as the reference tensor is too small for x add grad")
            ignored_failures = ignored_failures + 1
            validation_check_xadd_grad_failed = False
    else:
        validation_check_xadd_grad_failed = False

    validation_check_weight_grad_failed = not compare_weight_grads( opt_bn.weight.grad, torch_bn.weight.grad, "Weight Grad", rtol=rtol, atol=atol)

    if torch_bn.weight.grad.norm(p=float('inf')) < 1e-9 and validation_check_weight_grad_failed:
        print("Warning: ignoring the failed tensor comparison results as the reference tensor is too small for weight grad")
        ignored_failures = ignored_failures + 1
        validation_check_weight_grad_failed = False

    # Bias
    validation_check_bias_grad_failed = not compare_padded_tensors(opt_bn.bias.grad, torch_bn.bias.grad, "Bias Grad", rtol=rtol, atol=atol)

    # Out (Y)
    validation_check_out_failed = not compare_padded_tensors(y1.unblocked_tensor(), y2, "Y", W, output_hw_padding, rtol=rtol, atol=atol)

    if track_running_stats == True:
        #print("(opt_bn.running_mean - torch_bn.running_mean).abs().norm(inf)                    = ", (opt_bn.running_mean - torch_bn.running_mean).norm(p=float('inf')))
        #print("(opt_bn.running_mean - torch_bn.running_mean).norm(2) / torch_bn.run_mean        = ", (opt_bn.running_mean - torch_bn.running_mean).norm(2) / torch_bn.running_mean.norm(2))
        #print("(opt_bn.running_var  - torch_bn.running_var ).abs().norm(inf)                    = ", (opt_bn.running_var  - torch_bn.running_var) .norm(p=float('inf')))
        #print("(opt_bn.running_var  - torch_bn.running_var ).norm(2) / torch_bn.run_var         = ", (opt_bn.running_var  - torch_bn.running_var ).norm(2) / torch_bn.running_var.norm(2))
        validation_check_runstats_failed = not compare_padded_tensors(opt_bn.running_mean, torch_bn.running_mean, "Run mean", rtol=rtol, atol=atol) or not compare_padded_tensors(opt_bn.running_var, torch_bn.running_var, "Run var", rtol=rtol, atol=atol)
    else:
        validation_check_runstats_failed = False

    validation_checks_failed = validation_check_x_grad_failed or validation_check_xadd_grad_failed or validation_check_weight_grad_failed or validation_check_bias_grad_failed or validation_check_out_failed
    if track_running_stats == True:
        validation_checks_failed = validation_checks_failed or validation_check_runstats_failed

    if validation_checks_failed:
        print("Validation FAILED, ignored_failures = ", ignored_failures)
        print("Details:")
        print("validation_check_x_grad_failed      = ", validation_check_x_grad_failed)
        print("validation_check_xadd_grad_failed   = ", validation_check_xadd_grad_failed)
        print("validation_check_weight_grad_failed = ", validation_check_weight_grad_failed)
        print("validation_check_bias_grad_failed   = ", validation_check_bias_grad_failed)
        print("validation_check_out_failed         = ", validation_check_out_failed)
        print("validation_check_runstats_failed    = ", validation_check_runstats_failed)
    else:
        print("Validation PASSED, ignored failures = ", ignored_failures)

    if perf_fwd:

        blocked_input = opt_bn.get_blocked_tensor(
            x1,
            opt_bn.blocked_input_signature,
            [None, bc, None, None],
        )
        if has_eltwise:
            blocked_input_add = opt_bn.get_blocked_tensor(x1_add, opt_bn.blocked_input_signature, [None, bc, None, None])
            inputs = [blocked_input, blocked_input_add, opt_bn.weight, opt_bn.bias, opt_bn.mean, opt_bn.var]
        else:
            inputs = [blocked_input, opt_bn.weight, opt_bn.bias, opt_bn.mean, opt_bn.var]

        warmup_niter = niters_warmup
        #logging.info("warmup_niter = ", warmup_niter)
        print("warmup_niter = ", warmup_niter)

        #time_end = time.time()
        #print("Reference forward took (s) ", time_end - time_start)

        #dummy_tuning_timings = [0.0] * 16
        dummy_tuning_timings = np.zeros(16, dtype=np.float32)
        time_start = time.time()

        training = not scale_only
        print("training = ", training)
        eps = 1e-7
        for i in range(warmup_niter):
            if tuning_string_ncp is None or tuning_string_cp is None or len(tuning_string_ncp) == 0 or len(tuning_string_cp) == 0 or dummy_tuning_timings is None:
                batchnorm_cpp.batchnorm_fwd(training, has_relu, has_eltwise, eps, opt_padding, inputs)
            else:
                #if preallocated_output:
                #    conv_cpp.conv_fwd_preallocated_output_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings, allocated_y_bf16)
                #else:
                    #conv_cpp.conv_fwd_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)
                #    conv_cpp.conv_fwd_as_fused_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)
                batchnorm_cpp.batchnorm_fwd_ext(training, has_relu, has_eltwise, eps, opt_padding, tuning_string_ncp, tuning_string_cp, dummy_tuning_timings, inputs)

        time_end = time.time()
        print("Warmup took (s) ", time_end - time_start)

        timed_niters = niters
        #logging.info("timed_niters = ", timed_niters)
        print("timed_niters = ", timed_niters)

        #tuning_timings = [0.0] * 16
        tuning_timings = np.zeros(16, dtype=np.float32)
        #print("tuning_timings before: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        batchnorm_cpp.batchnorm_resume_itt()

        time_start = time.time()
        for i in range(timed_niters):
            if tuning_string_ncp is None or tuning_string_cp is None or len(tuning_string_ncp) == 0 or len(tuning_string_cp) == 0 or tuning_timings is None:
                batchnorm_cpp.batchnorm_fwd(training, has_relu, has_eltwise, eps, opt_padding, inputs)
            else:
                #if preallocated_output:
                #    conv_cpp.conv_fwd_preallocated_output_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings, allocated_y_bf16)
                #else:
                    #conv_cpp.conv_fwd_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)
                #    conv_cpp.conv_fwd_as_fused_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)
                batchnorm_cpp.batchnorm_fwd_ext(training, has_relu, has_eltwise, eps, opt_padding, tuning_string_ncp, tuning_string_cp, tuning_timings, inputs)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / timed_niters

        #print("tuning_timings after: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        print("Timed loop took (s) ", time_end - time_start)
        print("Final perf time: ", time_per_iter)
        gflop = batchnorm_cpp.batchnorm_fwd_get_gflop(N, C, H, W)
        basic_params_string = str(N) + " " + str(H) + " " + str(W) + " " + str(C) + " " + str(has_relu) + " " + str(has_eltwise) + " " + str(track_running_stats)
        print("Final perf GFLOPs: ", str(gflop/time_per_iter) + " basic: " + basic_params_string + " channel bs: " + str(bc) + " tuning_string_ncp: " + str(tuning_string_ncp) + " tuning_string_cp: " + str(tuning_string_cp))

        print("PERFDUMP,FP,na,"  + str(N) + "," + str(N) + "," + str(C) + "," + str(C) + "," + str(H) + "," + str(W) + "," + "na" + "," + "na" + "," + "na" + "," + str(opt_padding[0]) + "," + str(opt_padding[2]) + "," + str(time_per_iter) + "," + str(gflop/time_per_iter) + ',' + str(has_relu) + ',' + str(has_eltwise) + ',' + str(training))

        print("memory (1 tensor) per core (bytes/Kb/Mb): ", H * W * C * (2 if opt_dtype==torch.bfloat16 else 4),
                                                            H * W * C * (2 if opt_dtype==torch.bfloat16 else 4) / 1024.0,
                                                            H * W * C * (2 if opt_dtype==torch.bfloat16 else 4) / 1024.0 / 1024.0)

        # Checking the timings
        print("timings: b1 full without pre: ", tuning_timings[0], tuning_timings[0] / timed_niters, gflop, gflop / (tuning_timings[0] / timed_niters) if tuning_timings[0] != 0.0 else 0.0)
        print("timings: b1 full: ", tuning_timings[1], tuning_timings[1] / timed_niters, gflop, gflop / (tuning_timings[1] / timed_niters) if tuning_timings[1] != 0.0 else 0.0)
        print("timings: b1 stats: ", tuning_timings[2], tuning_timings[2] / timed_niters)#, gflop, gflop / (tuning_timings[1] / timed_niters) if tuning_timings[1] != 0.0 else 0.0)
        print("timings: b1 reduce: ", tuning_timings[3], tuning_timings[3] / timed_niters)#, gflop, gflop / (tuning_timings[1] / timed_niters) if tuning_timings[1] != 0.0 else 0.0)
        print("timings: b1 scale: ", tuning_timings[4], tuning_timings[4] / timed_niters)#, gflop, gflop / (tuning_timings[1] / timed_niters) if tuning_timings[1] != 0.0 else 0.0)

    return
    #exit()

def main():
    opt_dtype = torch.float if not args.use_bf16_opt else torch.bfloat16
    ref_dtype = torch.float if not args.use_bf16_ref else torch.bfloat16

    batchnorm_cpp.batchnorm_pause_itt()

    bc = args.bc

    if args.basic_sizes is not None:
        if len(args.basic_sizes) != 9:
            print("Error: basic sizes must have exactly 7 elements if defined (N, H, W, C, has_relu, has_eltwise, track_running_stats, pad_in, pad_out)")
            exit()
        [N, H, W, C, has_relu_int, has_eltwise_int, track_running_stats_int, pad_in, pad_out] = args.basic_sizes
        has_relu            = False if has_relu_int == 0 else 1
        has_eltwise         = False if has_eltwise_int == 0 else 1
        track_running_stats = False if track_running_stats_int == 0 else 1
        opt_padding = [pad_in, pad_in, pad_out, pad_out] #[0, 0, 1, 1] #[1, 1, 0, 0] #[0, 0, 1, 1] #[4, 4, 6, 6] #[0, 0, 0, 0] #[4, 4, 6, 6]
        run_test_bn(N, H, W, C, bc, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, args.with_perf, args.test_module,
                        args.tuning_string_ncp, args.tuning_string_cp, args.niters, args.niters_warmup, args.perf_fwd, args.scale_only)

    else:
        #with open("resnet50_bn_test_data_extended_new_28thr.data") as f:
        #with open("resnet50_bn_test_data_extended_new_56thr.data") as f:
        with open(args.test_data_file) as f:
            contents = f.readlines()
            for line in contents:
                #print("line")
                #print(type(line))
                #print(line)
                [N, C, H, W, has_relu, has_eltwise, track_running_stats] = list(line.split(" "))
                #[inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode] = list(line.split(" "))
                string_list = list(line.strip().split(" "))
                has_relu    = False if has_relu.strip()    == 'False' else True
                has_eltwise = False if has_eltwise.strip() == 'False' else True
                track_running_stats = False if track_running_stats.strip() == 'False' else True
                #print(string_list)
                #print(string_list[:7])
                #integer_map = map(int, string_list[:7])
                #print(string_list[:4])
                integer_map = map(int, string_list[:4])
                #print(integer_map)
                [N, C, H, W] = list(integer_map)
                opt_padding = [0, 0, 0, 0] #[0, 0, 1, 1] #[1, 1, 0, 0] #[0, 0, 1, 1] #[4, 4, 6, 6] #[0, 0, 0, 0] #[4, 4, 6, 6]
                run_test_bn(N, H, W, C, bc, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, args.with_perf, args.test_module,
                            args.tuning_string_ncp, args.tuning_string_cp, args.niters, args.niters_warmup, args.perf_fwd, args.scale_only)
    exit()

    # Just a single size run
    N=24 #16
    H=2 #28
    W=2 #28
    C=64
    bc=64
    opt_padding = [4, 4, 6, 6]
    has_relu=False
    has_eltwise=False
    track_running_stats=False

    run_test_bn(N, H, W, C, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, args.with_perf, args.test_module)

if __name__ == "__main__":
    args = parser.parse_args()
    main()


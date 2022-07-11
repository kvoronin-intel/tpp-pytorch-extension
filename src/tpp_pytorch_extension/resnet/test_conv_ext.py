import argparse
import time
import numpy as np
import torch

import pcl_pytorch_extension
from pcl_pytorch_extension._C import _conv as conv_cpp
import conv as conv_py

import pcl_cgbp
import pcl_cgbp_cpp

import test_utils
from test_utils import compare_weight_grads, compare_padded_tensors

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--test-module', default='cnn_tpp', type=str,
                    help='module to test against the reference', dest='test_module')

parser.add_argument('--use-bf16-opt', action="store_true", default=False, dest='use_bf16_opt')
parser.add_argument('--use-bf16-ref', action="store_true", default=False, dest='use_bf16_ref')

parser.add_argument('--bc',  nargs='?', type=int)
parser.add_argument('--bk',  nargs='?', type=int)

parser.add_argument('--tuning-params', nargs="+", default=None, type=int, help='h_block, w_block, c_block, k_block; h_in_gemm; pack_input')
parser.add_argument('--tuning-string', default=None, type=str, help='conv_string')

parser.add_argument('--test-data-file', default='resnet50_conv_test_data_for_bottleneck_28thr.data', type=str,
                    help='file to read test input data from', dest='test_data_file')

parser.add_argument('--basic-sizes', nargs="+", default=None, type=int, help='N H W inc outc stride R for the conv')

parser.add_argument('--niters', type=int, default=10, help='number of timed iterations (warmup hardcoded)')

parser.add_argument("--with-bwd", action="store_true", default=False, help='if true, runs backward (for validation)', dest='with_bwd')
parser.add_argument("--perf-fwd", action="store_true", default=False, help='if true, runs forward perf', dest='perf_fwd')
parser.add_argument("--perf-bwd-d", action="store_true", default=False, help='if true, runs backward over data perf', dest='perf_bwd_d')
parser.add_argument("--perf-bwd-w", action="store_true", default=False, help='if true, runs backward over weights perf', dest='perf_bwd_w')


#import pdb

global_counter = 0

#torch.autograd.set_detect_anomaly(True)

def run_test_conv(N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype,
                  with_bwd, perf_fwd, perf_bwd_d, perf_bwd_w, test_module, tuning_params, tuning_string, niters):
    time_start = time.time()
    print("debug: run_test_conv called with N H W inc outc bc bk R stride padding dilation groups has_bias padding_mode opt_dtype ref_dtype with_bwd perf_fwd perf_bwd_d perf_bwd_w test_module niters",
            N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, with_bwd, perf_fwd, perf_bwd_d, perf_bwd_w, test_module, niters)

    global global_counter

    channel_block_sizes = [bc, bk]

    if tuning_params is not None and test_module != 'ext_tpp':
        print("Custom tuning params can only be used for ext_tpp test_module")
        exit()

    if (perf_fwd and perf_bwd_w) or (perf_fwd and perf_bwd_d) or (perf_bwd_d and perf_bwd_w):
        print("Error: only one of perf-fwd, perf-bwd-w and perf-bwd-d can be active")
        exit()

    if perf_fwd:
        tuning_params_count = 6
    elif perf_bwd_d:
        tuning_params_count = 5
    elif perf_bwd_wd:
        tuning_params_count = -1

    if tuning_params is not None and len(tuning_params) != tuning_params_count:
        print("Wrong length of the tuning params (must be " + str(tuning_params_count) + " if present) = " + str(tuning_params) + " " + str(len(tuning_params)))
        exit()

    if tuning_params is not None:
        if perf_fwd:
            [h_block, w_block, c_block, k_block, h_in_gemm, pack_input_for_1x1_strided ] = tuning_params
        if perf_bwd_d:
            [h_block, w_block, c_block, k_block, h_in_gemm ] = tuning_params
        if perf_fwd or perf_bwd_d:
            print("info: tuning params: h_block, w_block, c_block, k_block = ", h_block, w_block, c_block, k_block)
            print("info: tuning params: h_in_gemm = ", h_in_gemm)
            if perf_fwd:
                print("info: pack input for 1x1 strided = ", pack_input_for_1x1_strided)
    else:
        tuning_params = None
        print("info: tuning params are empty")

    if tuning_string is not None:
        if test_module != 'ext_tpp':
            print("Custom tuning string can only be used for ext_tpp test_module")
            exit()
        c_loop_string = tuning_string
        print("info: tuning string: c_string = ", c_loop_string)
    else:
        tuning_string = None
        print("info: tuning string are empty")

    #inc = 4

    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        print("One of the modules is using bf16 hence padding input channels to an even number")
        if inc % 2 != 0:
            inc = inc + 1

    if (bc != None or bk != None) and test_module != 'ext_tpp':
        print("Custom block sizes can only be used for ext_tpp test_module")
        exit()

    opt_has_physical_padding = False

    torch.manual_seed(0)
    if test_module == 'cnn_tpp':
        print("info: testing TPP module from CNN (pcl_cgbp)")
        opt_conv = pcl_cgbp.XsmmConv2dTPP(inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype)
        hardcoded_bc=64
    elif test_module == 'ext_tpp':
        print("info: testing TPP module from extensions (pcl_pytorch_extension)")
        print("caution: TPP module from extensions only works with physical padding")
        opt_conv = conv_py.DummyConv2dTPP(inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, bc=bc, bk=bk)
        opt_has_physical_padding = True
        hardcoded_bc=64
    elif test_module == 'handlebased':
        print("info: testing handle-based module")
        if opt_dtype != torch.float:
            print("error: handlebased testing is only implemented for float")
            exit()
        opt_conv = pcl_cgbp.XsmmConv2d(inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode)
        hardcoded_bc=64
    else:
        print("test_module not supported, test_module = ", test_module)
        exit()

    time_end = time.time()
    print("Setting up opt_conv module took (s) ", time_end - time_start)
    time_start = time.time()

    if opt_has_physical_padding != False:
        #input_hw_padding  = [opt_padding[0], opt_padding[0], opt_padding[1], opt_padding[1]]
        #output_hw_padding = [opt_padding[2], opt_padding[2], opt_padding[3], opt_padding[3]]
        input_hw_padding  = [padding, padding, padding, padding]
        output_hw_padding = [padding, padding, padding, padding]
        print("input_hw_padding = ",  input_hw_padding)
        print("output_hw_padding = ", output_hw_padding)
    else:
        input_hw_padding  = [0, 0, 0, 0]
        output_hw_padding = [0, 0, 0, 0]

    torch_conv = torch.nn.Conv2d(inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode, device=None, dtype=ref_dtype)

    time_end = time.time()
    print("Setting up reference torch_conv module took (s) ", time_end - time_start)
    time_start = time.time()

    torch.manual_seed(0)

    weight = torch.randn([outc, inc//groups, R, R], requires_grad=True)
    #weight = torch.ones_like(weight)
    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        weight_bf16 = weight.to(torch.bfloat16)
        weight      = weight_bf16.to(torch.float)
    #print("weight shape = ", weight.shape)

    if has_bias:
        bias = torch.randn(outc, requires_grad=True)
        # Would be a mistake for bn since weight and bias remain fp32 even for bf16 activations
        #if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        #    bias_bf16 = bias.to(torch.bfloat16)
        #    bias      = bias_bf16.to(torch.float)
    #print("Weight: ", weight)
    #print("Bias: ", bias)
    x = torch.randn(N, inc, H, W, requires_grad=True)
    #x = torch.ones_like(x, requires_grad=True)
    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        x_bf16 = x.to(torch.bfloat16)
        x      = x_bf16.to(torch.float)
    #print("x shape = ", x.shape)

    if opt_dtype == torch.bfloat16:
        opt_x_init = x_bf16
        opt_weight_init = weight_bf16
    else:
        opt_x_init = x
        opt_weight_init = weight
    if has_bias:
        opt_bias_init = bias

    if opt_has_physical_padding != False:
        opt_x_init = torch.nn.functional.pad(opt_x_init, input_hw_padding, mode='constant', value=0.0)

    if ref_dtype == torch.bfloat16:
        ref_x_init = x_bf16
        ref_weight_init = weight_bf16
    else:
        ref_x_init = x
        ref_weight_init = weight
    if has_bias:
        ref_bias_init = bias

    x1 = opt_x_init.clone().detach().requires_grad_()
    x1.retain_grad()
    x2 = ref_x_init.clone().detach().requires_grad_()

    opt_conv.weight.data   = opt_weight_init.clone() # blocked_layout.py is taking care of blocking
    #print("opt_conv_weight shape = ", opt_conv.weight.shape)
    opt_conv.weight.block()
    torch_conv.weight.data = ref_weight_init.clone()

    if has_bias:
        opt_conv.bias.data   = opt_bias_init.clone()
        torch_conv.bias.data = ref_bias_init.clone()

    #for i in range(10):
    #    print("i opt_x_init ref_x_init = ", i, opt_x_init.view(-1)[i].item(), ref_x_init.view(-1)[i].item())
    #for i in range(10):
    #    print("i opt_weight_init ref_weight_init = ", i, opt_weight_init.view(-1)[i].item(), ref_weight_init.view(-1)[i].item())
    #if has_bias:
    #    for i in range(10):
    #          print("i opt_bias_init ref_bias_init = ", i, opt_bias_init.view(-1)[i].item(), ref_bias_init.view(-1)[i].item())

     # should be available if setup script has been called

    [bc, bk, lp_block] = [opt_conv.Cblock, opt_conv.Kblock, opt_conv.lp_block]

    """
    if test_module == 'ext_tpp' and hasattr(conv_cpp,'conv_get_feature_map_blocks'):
        [bc, bk, lp_block] = conv_cpp.conv_get_feature_map_blocks(inc, outc, 0 if opt_dtype == torch.float else 1)
    if test_module == 'cnn_tpp' and hasattr(pcl_cgbp_cpp,'conv_get_feature_map_blocks'):
        [bc, bk, lp_block] = pcl_cgbp_cpp.conv_get_feature_map_blocks(inc, outc, 0 if opt_dtype == torch.float else 1)
    else:
        print("Warning: could not use pcl_cgbp_cpp.conv_get_feature_map_blocks/conv_cpp.conv_get_feature_map_blocks, hence used hardcoded block sizes in the test")
        if inc % hardcoded_bc == 0:
          bc = hardcoded_bc
        else:
          bc = inc

        if outc % hardcoded_bc == 0:
          bk = hardcoded_bc
        else:
          bk = 1
    """
    print("Info: bc, bk = ", bc, bk)

    #y1 = opt_conv(x1, x1_add)
    #y2 = relu(torch_conv(x2) + x2_add)

    #for i in range(10):
    #    print("i x1 x2 = ", i, x1.view(-1)[i].item(), x2.view(-1)[i].item())

    time_end = time.time()
    print("Setting up the data for the modules took (s) ", time_end - time_start)
    time_start = time.time()

    y1 = opt_conv(x1) #xp)

    time_end = time.time()
    print("First forward took (s) ", time_end - time_start)
    time_start = time.time()

    y2 = torch_conv(x2)

    time_end = time.time()
    print("First reference forward took (s) ", time_end - time_start)
    time_start = time.time()

    y1_unblocked  = y1.unblocked_tensor()
    nchw_shape = y1_unblocked.shape
    print("debug: nchw shape = ", nchw_shape)
    outH = nchw_shape[2] - output_hw_padding[0] - output_hw_padding[1]
    outW = nchw_shape[3] - output_hw_padding[2] - output_hw_padding[3]

    if opt_has_physical_padding != False:
        print("debug: Zeroing the rim of the output tensor on the Python side")
        y1_zeroed_rim = torch.zeros_like(y1_unblocked)
        print("range = ", 'full', ' ', 'full', output_hw_padding[0], outH + output_hw_padding[0], output_hw_padding[2], outW + output_hw_padding[2])
        y1_zeroed_rim[:,:,output_hw_padding[0]:outH + output_hw_padding[0],output_hw_padding[2]:outW + output_hw_padding[2]] = y1_unblocked[:,:,output_hw_padding[0]:outH + output_hw_padding[0],output_hw_padding[2]:outW + output_hw_padding[2]]
        #for i in range (40):
        #    print("ind y1_zeroed_rim[end-ind] = ", i, y1_zeroed_rim.view(-1)[-i].item())
        opt_y_fp32 = y1_zeroed_rim.to(torch.float)
        y1_full_nan_count = torch.isnan(y1_unblocked.view(-1)).sum()
        y1_zeroed_rim_nan_count = torch.isnan(y1_zeroed_rim.view(-1)).sum()
        print("y1_full_nan_count = (ok-ish to have nonzero due to the padding)", y1_full_nan_count)
        if y1_zeroed_rim_nan_count > 0:
            print("Error: even after zeroing rims there are nans in the output, #nans = ", y1_zeroed_rim_nan_count)
            exit()
    else:
        opt_y_fp32 = y1.unblocked_tensor().to(torch.float)
    #opt_y_fp32 = y1.unblocked_tensor().to(torch.float)

    #for i in range(10):
    #    print("i y1 y2 = ", i, y1.view(-1)[i].item(), y2.view(-1)[i].item())
    #for i in range(10):
    #    print("i y1 y2 = ", -i-1, y1.view(-1)[-i-1].item(), y2.view(-1)[-i-1].item())
    
    # Output (fwd)
    compare_padded_tensors(y1.unblocked_tensor(), y2, "Y", outW, output_hw_padding, zero_rim_for_opt = True)

    time_end = time.time()
    print("Validating tensors for fwd took (s) ", time_end - time_start)
    time_start = time.time()

    if with_bwd:
        z1 = y1.mean()
        z2 = y2.mean()

        if opt_has_physical_padding != False:
            y1_numel = y1.unblocked_tensor().numel() # if hasattr(y1,unblocked_tensor) else y1.numel() does not work as a check
            print("z1 for zeroed rim (with account for padding)", y1_zeroed_rim.mean() * y1_numel / y2.numel())
        else:
            y1_numel = y1.tensor().numel() # if hasattr(y1,unblocked_tensor) else y1.numel() does not work as a check
            print("z1                         ", z1)
        print("z2                         ", z2)

        """
        #temp
        ref_y_fp32 = y2.to(torch.float)
        if opt_has_physical_padding != False:
            ref_y_fp32 = torch.nn.functional.pad(ref_y_fp32, output_hw_padding, mode='constant', value=0.0)
        shift = output_hw_padding[0] * (W + output_hw_padding[2] + output_hw_padding[3]) + output_hw_padding[2]
        print("shift = ", shift)
        for i in range(10):
            ind = i + shift - 5 if opt_has_physical_padding != False else i
            print("ind opt_y_fp32 ref_y_fp32 = ", ind, opt_y_fp32.view(-1)[ind].item(), ref_y_fp32.view(-1)[ind].item())

        #np.savetxt('conv_forward_run_' + str(global_counter) + '.txt', opt_y_fp32.contiguous().view(-1).detach().to(torch.float).numpy())
        global_counter = global_counter + 1

        return
        #exit()
        """

        
        if opt_has_physical_padding != False:
            x1.requires_grad              = False
            y1._t.retain_grad()
            opt_conv.weight.requires_grad = False
            if has_bias:
                opt_conv.bias.requires_grad   = False
            z1.backward(retain_graph=True,gradient=torch.tensor(1.*y1_numel / y2.numel(), dtype=torch.float))
            #for i in range(10):
            #    print("i after first bwd opt_conv.weight.grad = ", i, opt_conv.weight.grad.view(-1)[i].item())
            # zero the rim
            print("debug: Zeroing the rim of the output tensor on the Python side")
            nchw_shape = y1.grad.shape
            y1_grad_zeroed_rim = torch.zeros_like(y1.grad)
            print("debug: nchw shape = ", nchw_shape)
            outH = nchw_shape[2] - output_hw_padding[0] - output_hw_padding[1]
            outW = nchw_shape[3] - output_hw_padding[2] - output_hw_padding[3]
            print("range = ", 'full', ' ', 'full', output_hw_padding[0], outH + output_hw_padding[0], output_hw_padding[2], outW + output_hw_padding[2])
            y1_grad_zeroed_rim[:,:,output_hw_padding[0]:outH + output_hw_padding[0],output_hw_padding[2]:outW + output_hw_padding[2]] = y1.grad[:,:,output_hw_padding[0]:outH + output_hw_padding[0],output_hw_padding[2]:outW + output_hw_padding[2]]
            # now doing the main backward()
            x1.requires_grad              = True
            opt_conv.weight.requires_grad = True
            if has_bias:
                opt_conv.bias.requires_grad   = True
            y1._t.backward(gradient=y1_grad_zeroed_rim)
        else:
            z1.backward(retain_graph=True)
        #z1.backward(retain_graph=True)

        z2.backward(retain_graph=True)

        time_end = time.time()
        print("Preparing data for backward and backwards (both opt and ref) took (s) ", time_end - time_start)
        time_start = time.time()

        # X gradient
        compare_padded_tensors(x1.grad, x2.grad, "X Grad", W, input_hw_padding, zero_rim_for_opt = True)

        # Bias gradient
        if has_bias:
          compare_padded_tensors(opt_conv.bias.grad, torch_conv.bias.grad, "Bias Grad")

        # Weight gradient
        compare_weight_grads( opt_conv.weight.grad, torch_conv.weight.grad, "conv")

        time_end = time.time()
        print("Validating tensors for backward took (s) ", time_end - time_start)
        time_start = time.time()

    """
    counter = 0
    counter_reldiff = 0
    for i in range(inc*outc*R*R):
        ind = i
        val_ref = ref_weight_grad.view(-1)[ind].item()
        val_opt = opt_weight_grad_unblocked.view(-1)[ind].item()
        diff    = (val_opt - val_ref)
        reldiff = diff / val_ref
        if (diff > 1e-4):
            counter = counter + 1
            print("ind diff reldiff val_ref val_opt = ", ind, diff, reldiff, val_ref, val_opt)
        #print("ind opt_weight_grad ref_weight_grad = ", ind, opt_weight_grad_unblocked.view(-1)[ind].item(), ref_weight_grad.view(-1)[ind].item())
        if (reldiff > 1e-1):
            counter_reldiff = counter_reldiff + 1
            print("reldiff is too high for ind, val_ref val_opt ", ind, reldiff, val_ref, val_opt);
    print("Stats: bad diffs     are X out of Y:", counter, inc*outc*R*R)
    print("Stats: bad reldiffs  are X out of Y:", counter_reldiff, inc*outc*R*R)
    """

    #return
    #exit()

    conv_cfg = opt_conv.config

    blocked_input = opt_conv.get_blocked_tensor(
        x1,
        opt_conv.blocked_input_signature,
        [None, opt_conv.Cblock, None, None],
    )

    if perf_fwd:

        inputs = [blocked_input, opt_conv.weight]

        warmup_niter = 5
        #logging.info("warmup_niter = ", warmup_niter)
        print("warmup_niter = ", warmup_niter)

        time_end = time.time()
        print("Reference forward took (s) ", time_end - time_start)

        #dummy_tuning_timings = [0.0] * 16
        dummy_tuning_timings = np.zeros(16, dtype=np.float32)
        time_start = time.time()

        for i in range(warmup_niter):
            if tuning_params is None or tuning_string is None or len(tuning_params) == 0 or len(tuning_string) == 0 or dummy_tuning_timings is None:
                conv_cpp.conv_fwd(conv_cfg, inputs)
            else:
                conv_cpp.conv_fwd_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)

        time_end = time.time()
        print("Warmup took (s) ", time_end - time_start)

        timed_niters = niters
        #logging.info("timed_niters = ", timed_niters)
        print("timed_niters = ", timed_niters)

        #tuning_timings = [0.0] * 16
        tuning_timings = np.zeros(16, dtype=np.float32)
        #print("tuning_timings before: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        time_start = time.time()
        for i in range(timed_niters):
            if tuning_params is None or tuning_string is None or len(tuning_params) == 0 or len(tuning_string) == 0 or tuning_timings is None:
                conv_cpp.conv_fwd(conv_cfg, inputs)
            else:
                conv_cpp.conv_fwd_ext(conv_cfg, inputs, tuning_params, tuning_string, tuning_timings)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / timed_niters

        #print("tuning_timings after: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        print("Timed loop took (s) ", time_end - time_start)
        print("Final perf time: ", time_per_iter)
        gflop = conv_cpp.conv_fwd_get_gflop(conv_cfg)
        basic_params_string = str(N) + " " + str(H) + " " + str(W) + " " + str(inc) + " " + str(outc) + " " + str(stride)
        print("Final perf GFLOPs: ", str(gflop/time_per_iter) + " basic: " + basic_params_string + " channel bs: " + str(channel_block_sizes) + " tuning params: "+ str(tuning_params) + " tuning_string: " + str(tuning_string))

        # Checking the timings
        print("timings: c1 gflop_c1 gflops_c1: ", tuning_timings[0], gflop, gflop / (tuning_timings[0] / timed_niters) if tuning_timings[0] != 0.0 else 0.0)

        sum_timings = tuning_timings[0]
        print("timing diff vs pure conv (part of conv_fwd_tmpl) (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

        sum_timings = tuning_timings[1]
        print("timing diff vs conv_fwd_tmpl (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

    if perf_bwd_d:

        inputs = [y1_grad_zeroed_rim, blocked_input, opt_conv.weight]

        warmup_niter = 5
        #logging.info("warmup_niter = ", warmup_niter)
        print("warmup_niter = ", warmup_niter)

        #dummy_tuning_timings = [0.0] * 16
        dummy_tuning_timings = np.zeros(16, dtype=np.float32)
        time_start = time.time()

        for i in range(warmup_niter):
            if tuning_params is None or tuning_string is None or len(tuning_params) == 0 or len(tuning_string) == 0 or dummy_tuning_timings is None:
                conv_cpp.conv_bwd_d(conv_cfg, inputs)
            else:
                conv_cpp.conv_bwd_d_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)

        time_end = time.time()
        print("Warmup took (s) ", time_end - time_start)

        timed_niters = niters
        #logging.info("timed_niters = ", timed_niters)
        print("timed_niters = ", timed_niters)

        #tuning_timings = [0.0] * 16
        tuning_timings = np.zeros(16, dtype=np.float32)
        #print("tuning_timings before: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        time_start = time.time()
        for i in range(timed_niters):
            if tuning_params is None or tuning_string is None or len(tuning_params) == 0 or len(tuning_string) == 0 or tuning_timings is None:
                conv_cpp.conv_bwd_d(conv_cfg, inputs)
            else:
                conv_cpp.conv_bwd_d_ext(conv_cfg, inputs, tuning_params, tuning_string, tuning_timings)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / timed_niters

        #print("tuning_timings after: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        print("Timed loop took (s) ", time_end - time_start)
        print("Final perf time: ", time_per_iter)
        gflop = conv_cpp.conv_bwd_d_get_gflop(conv_cfg)
        basic_params_string = str(N) + " " + str(H) + " " + str(W) + " " + str(inc) + " " + str(outc) + " " + str(stride)
        print("Final perf GFLOPs: ", str(gflop/time_per_iter) + " basic: " + basic_params_string + " channel bs: " + str(channel_block_sizes) + " tuning params: "+ str(tuning_params) + " tuning_string: " + str(tuning_string))

        # Checking the timings
        print("timings: c1 gflop_c1 gflops_c1: ", tuning_timings[0], gflop, gflop / (tuning_timings[0] / timed_niters) if tuning_timings[0] != 0.0 else 0.0)

        sum_timings = tuning_timings[0]
        print("timing diff vs pure conv (part of conv_fwd_tmpl) (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

        sum_timings = tuning_timings[2]
        print("timing diff vs conv_bwd_d_tmpl (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

    if perf_bwd_w:
        print("Error: perf_bwd_w = True is not supported")
        exit()

    return

def main():
    opt_dtype = torch.float if not args.use_bf16_opt else torch.bfloat16
    ref_dtype = torch.float if not args.use_bf16_ref else torch.bfloat16

    bc = args.bc
    bk = args.bk

    #with open("resnet50_conv_test_data.data") as f:
    #with open("resnet50_conv_test_data_extended.data") as f:
    #with open("resnet50_conv_test_data_extended_1thr.data") as f:
    #with open("resnet50_conv_test_data_extended_24thr_custom.data") as f:
    #with open("resnet50_conv_test_data_extended.data") as f:
    #with open("resnet50_conv_test_data_extended_new.data") as f:
    #with open("resnet50_conv_test_data_extended_new_28thr_reordered.data") as f:
    #with open("resnet50_conv_test_data_extended_new_28thr.data") as f:
    #with open("resnet50_conv_test_data_for_bottleneck_28thr.data") as f:
    if args.basic_sizes is not None:
        if len(args.basic_sizes) != 7:
            print("Error: basic sizes must have exactly 7 elements if defined (N, H, W, inc, outc, stride, R)")
            exit()
        [N, H, W, inc, outc, stride, R] = args.basic_sizes
        padding = R // 2
        groups=1
        dilation=1
        has_bias = False
        padding_mode='zeros'
        run_test_conv(N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype,
                      args.with_bwd, args.perf_fwd, args.perf_bwd_d, args.perf_bwd_w, args.test_module,
                      args.tuning_params, args.tuning_string, args.niters)
    else:
        with open(args.test_data_file) as f:
            contents = f.readlines()
            for line in contents:
                [N, H, W, inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode] = list(line.split(" "))
                #[inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode] = list(line.split(" "))
                string_list = list(line.strip().split(" "))
                has_bias=False if has_bias.strip() == 'False' else True
                padding_mode=padding_mode.strip()
                integer_map = map(int, string_list[:10])
                [N, H, W, inc, outc, R, stride, padding, dilation, groups] = list(integer_map)
                run_test_conv(N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype,
                              args.with_bwd, args.perf_fwd, args.perf_bwd_d, args.perf_bwd_w, args.test_module,
                              args.tuning_params, args.tuning_string, args.niters)
    exit()

    # Just a single size run
    inc=3
    outc=64
    bc=64
    bk=64
    R=7
    stride=2
    padding=3
    has_bias=False
    groups=1
    dilation=1
    padding=0
    padding_mode='zeros'

    N=24
    H=224
    W=224

    run_test_conv(N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype,
                  args.with_bwd, args.perf_fwd, args.perf_bwd_d, args.perf_bwd_w, args.test_module)

if __name__ == "__main__":
    args = parser.parse_args()
    main()


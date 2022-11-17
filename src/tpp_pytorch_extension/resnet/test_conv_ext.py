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

parser.add_argument('--test-data-file', default='resnet50_conv_test_data_for_bottleneck_56thr.data', type=str,
                    help='file to read test input data from', dest='test_data_file')

parser.add_argument('--basic-sizes', nargs="+", default=None, type=int, help='N H W inc outc stride R for the conv')

parser.add_argument("--logical-padding", action="store_true", default=False, help='if true, runs with logical padding', dest='logical_padding')

parser.add_argument('--niters', type=int, default=100, help='number of timed iterations')
parser.add_argument('--niters-warmup', type=int, default=10, help='number of warmup iterations')

parser.add_argument("--with-bwd", action="store_true", default=False, help='if true, runs backward (for validation)', dest='with_bwd')
parser.add_argument("--perf-fwd", action="store_true", default=False, help='if true, runs forward perf', dest='perf_fwd')
parser.add_argument("--perf-bwd-d", action="store_true", default=False, help='if true, runs backward over data perf', dest='perf_bwd_d')
parser.add_argument("--perf-bwd-w", action="store_true", default=False, help='if true, runs backward over weights perf', dest='perf_bwd_w')

parser.add_argument("--preallocated-output", action="store_true", default=False, help='if true, allocates output and calls in perf section conv wihtout preallocated output tensor', dest='preallocated_output')

parser.add_argument('--use-hardcoded-tunings', action="store_true", default=False, dest='use_hardcoded_tunings')

#import pdb

global_counter = 0

#torch.autograd.set_detect_anomaly(True)

def run_test_conv(N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype,
                  with_bwd, perf_fwd, perf_bwd_d, perf_bwd_w, test_module, tuning_params, tuning_string, niters, niters_warmup, preallocated_output,
                  logical_padding, use_hardcoded_tunings):
    time_start = time.time()
    print("debug: run_test_conv called with N H W inc outc bc bk R stride padding dilation groups has_bias padding_mode opt_dtype ref_dtype with_bwd perf_fwd perf_bwd_d perf_bwd_w test_module niters niters_warmup preallocated_output logical_padding use_hardcoded_tunings",
            N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode,
            opt_dtype, ref_dtype,
            with_bwd, perf_fwd, perf_bwd_d, perf_bwd_w, test_module, niters, niters_warmup,
            preallocated_output, logical_padding, use_hardcoded_tunings)

    global global_counter

    channel_block_sizes = [bc, bk]

    if tuning_params is not None and test_module != 'ext_tpp':
        print("Custom tuning params can only be used for ext_tpp test_module")
        exit()

    if (perf_fwd and perf_bwd_w) or (perf_fwd and perf_bwd_d) or (perf_bwd_d and perf_bwd_w):
        print("Error: only one of perf-fwd, perf-bwd-w and perf-bwd-d can be active")
        exit()

    disabled_bwd_d = not with_bwd
    if logical_padding and test_module == 'ext_tpp':
        print("For logical padding backward over data is not implemented in ext_tpp module")
        disabled_bwd_d = True

    if perf_fwd:
        tuning_params_count = 6
    elif perf_bwd_d:
        tuning_params_count = 5
    elif perf_bwd_w:
        tuning_params_count = 11

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
        if perf_bwd_w:
            [p_block,
               bf16_use_nchw_format,
               pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni,
               bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block,
               use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams ] = tuning_params
            print("info: tuning params: p_block = ", p_block)
            print("info: tuning params: bf16_use_nchw_format = ", bf16_use_nchw_format)
            print("info: tuning params: pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni = ", pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni)
            print("info: tuning params: bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block = ", bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block)
            print("info: tuning params: use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams = ", use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams)
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

    if (bc != None or bk != None) and (test_module != 'ext_tpp' and test_module != 'cnn_tpp'):
        print("Custom block sizes can only be used for ext_tpp and cnn_tpp test_modules")
        exit()

    opt_has_physical_padding = False

    torch.manual_seed(0)
    if test_module == 'cnn_tpp':
        print("info: testing TPP module from CNN (pcl_cgbp)")
        opt_conv = pcl_cgbp.XsmmConv2dTPP(inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, bc=bc, bk=bk)
    elif test_module == 'ext_tpp':
        print("info: testing TPP module from extensions (pcl_pytorch_extension)")
        print("caution: TPP module from extensions only works with physical padding")
        opt_conv = conv_py.DummyConv2dTPP(inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, bc=bc, bk=bk, logical_padding=logical_padding, use_hardcoded_tunings=use_hardcoded_tunings)
        opt_has_physical_padding = (not logical_padding) and (padding != 0)
    elif test_module == 'handlebased':
        print("info: testing handle-based module")
        if opt_dtype != torch.float:
            print("error: handlebased testing is only implemented for float")
            exit()
        opt_conv = pcl_cgbp.XsmmConv2d(inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode)
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
        print("running with logical padding")
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
    x = torch.randn(N, inc, H, W, requires_grad=not disabled_bwd_d)
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

    x1 = opt_x_init.clone().detach().requires_grad_(requires_grad=not disabled_bwd_d)
    if not disabled_bwd_d:
        x1.retain_grad()
    x2 = ref_x_init.clone().detach().requires_grad_(requires_grad=not disabled_bwd_d)

    if preallocated_output:
        allocated_y = torch.randn(N, outc, H//stride + 2 * output_hw_padding[0], W//stride + 2*output_hw_padding[2], requires_grad=False)
        print("allocated_y.shape = ", allocated_y.shape)
        #x = torch.ones_like(x, requires_grad=True)
        if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
            allocated_y_bf16 = allocated_y.to(torch.bfloat16)
            allocated_y      = allocated_y_bf16.to(torch.float)

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

    print("Info: bc, bk = ", bc, bk)

    #y1 = opt_conv(x1, x1_add)
    #y2 = relu(torch_conv(x2) + x2_add)

    #for i in range(10):
    #    print("i x1 x2 = ", i, x1.view(-1)[i].item(), x2.view(-1)[i].item())

    time_end = time.time()
    print("Setting up the data for the modules took (s) ", time_end - time_start)
    time_start = time.time()

    dummy_tuning_timings = np.zeros(16, dtype=np.float32)
    if perf_fwd:
        y1 = opt_conv(x1, tuning_params_fwd=tuning_params, tuning_string_fwd=tuning_string, tuning_timings_fwd=dummy_tuning_timings)
    elif perf_bwd_d:
        y1 = opt_conv(x1, tuning_params_d=tuning_params, tuning_string_d=tuning_string, tuning_timings_d=dummy_tuning_timings)
    elif perf_bwd_w:
        y1 = opt_conv(x1, tuning_params_w=tuning_params, tuning_string_w=tuning_string, tuning_timings_w=dummy_tuning_timings)
    else:
        y1 = opt_conv(x1)

    #y1 = opt_conv(x1) #xp)

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

    # Very loose tolerances to check only obvious errors
    rtol=1.5e-1
    atol=1e+0

    # Output (fwd)
    validation_check_fwd_failed = not compare_padded_tensors(y1.unblocked_tensor(), y2, "Y", outW, output_hw_padding, zero_rim_for_opt = True, rtol=rtol, atol=atol)

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
            y1_numel = y1.unblocked_tensor().numel()  if hasattr(y1,'unblocked_tensor') else y1.numel() #does not work as a check
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
            x1.requires_grad              = True and not disabled_bwd_d
            opt_conv.weight.requires_grad = True
            if has_bias:
                opt_conv.bias.requires_grad   = True
            y1._t.backward(gradient=y1_grad_zeroed_rim)
        else:
            y1._t.retain_grad()
            z1.backward(retain_graph=True)
            y1_grad_zeroed_rim = y1.grad
        #z1.backward(retain_graph=True)

        z2.backward(retain_graph=True)

        time_end = time.time()
        print("Preparing data for backward and backwards (both opt and ref) took (s) ", time_end - time_start)
        time_start = time.time()

        # X gradient
        if not disabled_bwd_d:
            validation_check_bwd_d_failed = not compare_padded_tensors(x1.grad, x2.grad, "X Grad", W, input_hw_padding, zero_rim_for_opt = True, rtol=rtol, atol=atol)
        else:
            validation_check_bwd_d_failed = False

        # Bias gradient
        if has_bias:
          validation_check_bwd_bias_failed = not compare_padded_tensors(opt_conv.bias.grad, torch_conv.bias.grad, "Bias Grad", rtol=rtol, atol=atol)
        else:
          validation_check_bwd_bias_failed = False

        # Weight gradient
        validation_check_bwd_w_failed = not compare_weight_grads( opt_conv.weight.grad, torch_conv.weight.grad, "W Grad", rtol=rtol, atol=atol)

        validation_checks_failed = validation_check_fwd_failed or validation_check_bwd_d_failed or validation_check_bwd_bias_failed or validation_check_bwd_w_failed
        print("validation_check_fwd_failed      = ", validation_check_fwd_failed)
        print("validation_check_bwd_d_failed    = ", validation_check_bwd_d_failed)
        print("validation_check_bwd_bias_failed = ", validation_check_bwd_bias_failed, " and disabled_bwd_d = ", disabled_bwd_d)
        print("validation_check_bwd_w_failed    = ", validation_check_bwd_w_failed)

        time_end = time.time()
        print("Validating tensors for backward took (s) ", time_end - time_start)
        time_start = time.time()

        if validation_checks_failed:
            print("Validation FAILED")
        else:
            print("Validation PASSED")
    else:
        if validation_check_fwd_failed:
            print("Validation FAILED")
        else:
            print("Validation PASSED")
    #exit()
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

    if test_module != 'ext_tpp':
        return

    conv_cfg = opt_conv.config

    blocked_input = opt_conv.get_blocked_tensor(
        x1,
        opt_conv.blocked_input_signature,
        [None, opt_conv.Cblock, None, None],
    )

    if perf_fwd:

        inputs = [blocked_input, opt_conv.weight]

        warmup_niter = niters_warmup
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
                if preallocated_output:
                    conv_cpp.conv_fwd_preallocated_output_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings, allocated_y_bf16)
                else:
                    #conv_cpp.conv_fwd_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)
                    conv_cpp.conv_fwd_as_fused_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)

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
                if preallocated_output:
                    conv_cpp.conv_fwd_preallocated_output_ext(conv_cfg, inputs, tuning_params, tuning_string, tuning_timings, allocated_y_bf16)
                else:
                    #print("calling conv_fwd_ext")
                    conv_cpp.conv_fwd_ext(conv_cfg, inputs, tuning_params, tuning_string, tuning_timings)
                    #conv_cpp.conv_fwd_as_fused_ext(conv_cfg, inputs, tuning_params, tuning_string, tuning_timings)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / timed_niters

        #print("tuning_timings after: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        print("Timed loop took (s) ", time_end - time_start)
        print("Final perf time: ", time_per_iter)
        gflop = conv_cpp.conv_fwd_get_gflop(conv_cfg)
        basic_params_string = str(N) + " " + str(H) + " " + str(W) + " " + str(inc) + " " + str(outc) + " " + str(stride) + " " + str(R)
        print("Final perf GFLOPs: ", str(gflop/time_per_iter) + " basic: " + basic_params_string + " channel bs: " + str(channel_block_sizes) + " tuning params: "+ str(tuning_params) + " tuning_string: " + str(tuning_string))

        print("PERFDUMP,FP,na,"  + str(N) + "," + str(N) + "," + str(inc) + "," + str(outc) + "," + str(H) + "," + str(W) + "," + str(R) + "," + str(R) + "," + str(stride) + "," + str(padding) + "," + str(padding) + "," + str(time_per_iter) + "," + str(gflop/time_per_iter))

        # Checking the timings
        print("timings: c1 gflop_c1 gflops_c1: ", tuning_timings[0], gflop, gflop / (tuning_timings[0] / timed_niters) if tuning_timings[0] != 0.0 else 0.0)

        sum_timings = tuning_timings[0]
        print("timing diff (per iter) for PT vs pure conv_fwd scope (part of conv_fwd_tmpl) (abs and %) = ", (time_end - time_start - sum_timings) / timed_niters, (time_end - time_start - sum_timings) / (time_end - time_start) * 100)
        print("Final conv_fwd perf GFLOPs: ", str(gflop/(sum_timings / timed_niters)) + " basic: " + basic_params_string + " channel bs: " + str(channel_block_sizes) + " tuning params: "+ str(tuning_params) + " tuning_string: " + str(tuning_string))

        print("PERFDUMP,FP,na2," + str(N) + "," + str(N) + "," + str(inc) + "," + str(outc) + "," + str(H) + "," + str(W) + "," + str(R) + "," + str(R) + "," + str(stride) + "," + str(padding) + "," + str(padding) + "," + str(sum_timings / timed_niters) + "," + str(gflop/(sum_timings / timed_niters)))

        sum_timings = tuning_timings[1]
        print("timing diff (per iter) for PT vs conv_fwd_tmpl (abs and %) = ", (time_end - time_start - sum_timings) / timed_niters, (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

        #printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, omp_get_max_threads(), N, C, K,
        #H, W, R, S, stride_h, pad_h, pad_w, ((double)((t_end - t_start)/n_iters)), (gflop)/(t_end - t_start), norms.l1_ref, norms.l1_tst,
        #norms.l2_abs, norms.l2_rel, norms.linf_abs, norms.linf_rel, norms.normf_rel);

        print("PERFDUMP,FP,na3," + str(N) + "," + str(N) + "," + str(inc) + "," + str(outc) + "," + str(H) + "," + str(W) + "," + str(R) + "," + str(R) + "," + str(stride) + "," + str(padding) + "," + str(padding) + "," + str(sum_timings / timed_niters) + "," + str(gflop/(sum_timings / timed_niters)))

    if perf_bwd_d:

        inputs = [y1_grad_zeroed_rim, blocked_input, opt_conv.weight]

        warmup_niter = niters_warmup
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
        basic_params_string = str(N) + " " + str(H) + " " + str(W) + " " + str(inc) + " " + str(outc) + " " + str(stride)  + " " + str(R)
        print("Final perf GFLOPs: ", str(gflop/time_per_iter) + " basic: " + basic_params_string + " channel bs: " + str(channel_block_sizes) + " tuning params: "+ str(tuning_params) + " tuning_string: " + str(tuning_string))

        # Checking the timings
        print("timings: c1 gflop_c1 gflops_c1: ", tuning_timings[0], gflop, gflop / (tuning_timings[0] / timed_niters) if tuning_timings[0] != 0.0 else 0.0)

        sum_timings = tuning_timings[0]
        print("timing diff vs pure conv (part of conv_fwd_tmpl) (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

        sum_timings = tuning_timings[2]
        print("timing diff vs conv_bwd_d_tmpl (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

    if perf_bwd_w:
        inputs = [y1_grad_zeroed_rim, blocked_input, opt_conv.weight]

        warmup_niter = niters_warmup
        #logging.info("warmup_niter = ", warmup_niter)
        print("warmup_niter = ", warmup_niter)

        #dummy_tuning_timings = [0.0] * 16
        dummy_tuning_timings = np.zeros(16, dtype=np.float32)
        time_start = time.time()

        for i in range(warmup_niter):
            if tuning_params is None or tuning_string is None or len(tuning_params) == 0 or len(tuning_string) == 0 or dummy_tuning_timings is None:
                conv_cpp.conv_bwd_w(conv_cfg, inputs)
            else:
                conv_cpp.conv_bwd_w_ext(conv_cfg, inputs, tuning_params, tuning_string, dummy_tuning_timings)

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
                conv_cpp.conv_bwd_w(conv_cfg, inputs)
            else:
                conv_cpp.conv_bwd_w_ext(conv_cfg, inputs, tuning_params, tuning_string, tuning_timings)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / timed_niters

        #print("tuning_timings after: ", type(tuning_timings), tuning_timings.dtype, tuning_timings)

        print("Timed loop took (s) ", time_end - time_start)
        print("Final perf time: ", time_per_iter)
        gflop = conv_cpp.conv_bwd_w_get_gflop(conv_cfg)
        basic_params_string = str(N) + " " + str(H) + " " + str(W) + " " + str(inc) + " " + str(outc) + " " + str(stride) + " " + str(R)
        print("Final perf GFLOPs: ", str(gflop/time_per_iter) + " basic: " + basic_params_string + " channel bs: " + str(channel_block_sizes) + " tuning params: "+ str(tuning_params) + " tuning_string: " + str(tuning_string))

        # Checking the timings
        print("timings: c1 gflop_c1 gflops_c1: ", tuning_timings[0], gflop, gflop / (tuning_timings[0] / timed_niters) if tuning_timings[0] != 0.0 else 0.0)

        sum_timings = tuning_timings[0]
        print("timing diff vs pure conv (part of conv_bwd_w_tmpl) (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

        sum_timings = tuning_timings[2]
        print("timing diff vs conv_bwd_w_tmpl (abs and %) = ", (time_end - time_start - sum_timings), (time_end - time_start - sum_timings) / (time_end - time_start) * 100)

        #print("Error: perf_bwd_w = True is not supported")
        #exit()

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
                      args.tuning_params, args.tuning_string, args.niters, args.niters_warmup, args.preallocated_output, args.logical_padding, args.use_hardcoded_tunings)
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
                              args.tuning_params, args.tuning_string, args.niters, args.niters_warmup, args.preallocated_output, args.logical_padding, args.use_hardcoded_tunings)
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


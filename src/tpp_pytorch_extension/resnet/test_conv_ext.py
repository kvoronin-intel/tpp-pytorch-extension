import argparse
import torch
import numpy as np

import pcl_pytorch_extension
from pcl_pytorch_extension._C import _conv as conv_cpp
import conv as conv_py

import pcl_cgbp
import pcl_cgbp_cpp

"""
import argparse
import torch
import numpy as np
import time

import pcl_cgbp
import pcl_cgbp_cpp

import blocked_layout
from blocked_layout import BlockedTensor, BlockedParameter
"""

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--test-module', default='cnn_tpp', type=str,
                    help='module to test against the reference', dest='test_module')

parser.add_argument("--with-perf", action="store_true", default=False, help='if true, measures performance additionally for the opt conv module', dest='with_perf')

parser.add_argument('--use-bf16-opt', action="store_true", default=False, dest='use_bf16_opt')
parser.add_argument('--use-bf16-ref', action="store_true", default=False, dest='use_bf16_ref')

#import pdb

global_counter = 0

#torch.autograd.set_detect_anomaly(True)

def run_test_conv(N, H, W, inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, with_perf, test_module):
    print("debug: run_test_conv called with N H W inc outc K stride padding dilation groups has_bias padding_mode opt_dtype ref_dtype with_perf test_module ",
            N, H, W, inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, with_perf, test_module)

    global global_counter
    #pcl_cgbp.init_libxsmm() # Not needed?

    #inc = 4

    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        print("One of the modules is using bf16 hence padding input channels to an even number")
        if inc % 2 != 0:
            inc = inc + 1

    opt_has_physical_padding = False

    torch.manual_seed(0)
    if test_module == 'cnn_tpp':
        print("info: testing TPP module from CNN (pcl_cgbp)")
        opt_conv = pcl_cgbp.XsmmConv2dTPP(inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype)
        hardcoded_bc=64
    elif test_module == 'ext_tpp':
        print("info: testing TPP module from extensions (pcl_pytorch_extension)")
        print("caution: TPP module from extensions only works with physical padding")
        opt_conv = conv_py.DummyConv2dTPP(inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype)
        opt_has_physical_padding = True
        hardcoded_bc=64
    elif test_module == 'handlebased':
        print("info: testing handle-based module")
        if opt_dtype != torch.float:
            print("error: handlebased testing is only implemented for float")
            exit()
        opt_conv = pcl_cgbp.XsmmConv2d(inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode)
        hardcoded_bc=64
    else:
        print("test_module not supported, test_module = ", test_module)
        exit()

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

    torch_conv = torch.nn.Conv2d(inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode, device=None, dtype=ref_dtype)

    torch.manual_seed(0)

    weight = torch.randn([outc, inc//groups, K, K], requires_grad=True)
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
    print("Info: bc, bk = ", bc, bk)

    #y1 = opt_conv(x1, x1_add)
    #y2 = relu(torch_conv(x2) + x2_add)

    #for i in range(10):
    #    print("i x1 x2 = ", i, x1.view(-1)[i].item(), x2.view(-1)[i].item())

    y1 = opt_conv(x1) #xp)
    y2 = torch_conv(x2)

    if opt_has_physical_padding != False:
        y1_unblocked  = y1.unblocked_tensor()
        print("debug: Zeroing the rim of the output tensor on the Python side")
        nchw_shape = y1_unblocked.shape
        y1_zeroed_rim = torch.zeros_like(y1_unblocked)
        print("debug: nchw shape = ", nchw_shape)
        outH = nchw_shape[2] - output_hw_padding[0] - output_hw_padding[1]
        outW = nchw_shape[3] - output_hw_padding[2] - output_hw_padding[3]
        print("range = ", 'full', ' ', 'full', output_hw_padding[0], outH + output_hw_padding[0], output_hw_padding[2], outW + output_hw_padding[2])
        y1_zeroed_rim[:,:,output_hw_padding[0]:outH + output_hw_padding[0],output_hw_padding[2]:outW + output_hw_padding[2]] = y1_unblocked[:,:,output_hw_padding[0]:outH + output_hw_padding[0],output_hw_padding[2]:outW + output_hw_padding[2]]
        #for i in range (40):
        #    print("ind y1_zeroed_rim[end-ind] = ", i, y1_zeroed_rim.view(-1)[-i].item())
        opt_y_fp32 = y1_zeroed_rim.to(torch.float)
    else:
        opt_y_fp32 = y1.unblocked_tensor().to(torch.float)
    #opt_y_fp32 = y1.unblocked_tensor().to(torch.float)

    #for i in range(10):
    #    print("i y1 y2 = ", i, y1.view(-1)[i].item(), y2.view(-1)[i].item())
    #for i in range(10):
    #    print("i y1 y2 = ", -i-1, y1.view(-1)[-i-1].item(), y2.view(-1)[-i-1].item())

    z1 = y1.mean()
    z2 = y2.mean()

    if opt_has_physical_padding != False:
        y1_numel = y1.unblocked_tensor().numel() # if hasattr(y1,unblocked_tensor) else y1.numel() does not work as a check
        print("z1 for zeroed rim (with account for padding)", y1_zeroed_rim.mean() * y1_numel / y2.numel())
    else:
        y1_numel = y1.unblocked_tensor().numel() # if hasattr(y1,unblocked_tensor) else y1.numel() does not work as a check
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

    #for i in range(10):
    #    print("i opt_conv.weight.grad = ", i, opt_conv.weight.grad.view(-1)[i].item())

    if opt_has_physical_padding != False:
        print("debug: Zeroing the rim of the x_grad tensor on the Python side")
        nchw_shape = x1.grad.shape
        opt_x_grad_zeroed_rim = torch.zeros_like(x1.grad)
        print("debug: nchw shape = ", nchw_shape)
        inH = nchw_shape[2] - input_hw_padding[0] - input_hw_padding[1]
        inW = nchw_shape[3] - input_hw_padding[2] - input_hw_padding[3]
        print("range = ", 'full', ' ', 'full', input_hw_padding[0], inH + input_hw_padding[0], input_hw_padding[2], inW + input_hw_padding[2])
        opt_x_grad_zeroed_rim[:,:,input_hw_padding[0]:inH + input_hw_padding[0],input_hw_padding[2]:inW + input_hw_padding[2]] = x1.grad[:,:,input_hw_padding[0]:inH + input_hw_padding[0],input_hw_padding[2]:inW + input_hw_padding[2]]
        opt_x_grad = opt_x_grad_zeroed_rim.to(torch.float)
    else:
        opt_x_grad = x1.grad.to(torch.float)
    #opt_x_grad = x1.grad.to(torch.float)

    opt_weight_grad = opt_conv.weight.grad.to(torch.float)
    #opt_y_fp32 = y1.unblocked_tensor().to(torch.float)

    """
    if opt_dtype == torch.bfloat16:
        opt_x_grad = x1.grad.to(torch.float)
        opt_weight_grad = opt_conv.weight.grad.to(torch.float)
        opt_y_fp32 = y1.unblocked_tensor().to(torch.float)
    else:
        opt_x_grad = x1.grad
        opt_weight_grad = opt_conv.weight.grad
        opt_y_fp32 = y1.unblocked_tensor()
    """

    if has_bias:
        opt_bias_grad = opt_conv.bias.grad

    ref_x_grad = x2.grad.to(torch.float)
    ref_weight_grad = torch_conv.weight.grad.to(torch.float)
    ref_y_fp32 = y2.to(torch.float)

    if opt_has_physical_padding != False:
        ref_y_fp32 = torch.nn.functional.pad(ref_y_fp32, output_hw_padding, mode='constant', value=0.0)
        ref_x_grad = torch.nn.functional.pad(ref_x_grad, input_hw_padding,  mode='constant', value=0.0)

    shift = output_hw_padding[0] * (W + output_hw_padding[2] + output_hw_padding[3]) + output_hw_padding[2]
    print("shift = ", shift)
    for i in range(10):
        ind = i + shift - 5 if opt_has_physical_padding != False else i
        print("ind opt_y_fp32 ref_y_fp32 = ", ind, opt_y_fp32.view(-1)[ind].item(), ref_y_fp32.view(-1)[ind].item())

    shift = (input_hw_padding[0] * (W + input_hw_padding[2] + input_hw_padding[3]) + input_hw_padding[2])
    print("shift = ", shift)
    for i in range(10):
        ind = i + shift - 5 if opt_has_physical_padding != False else i
        print("ind opt_x_grad ref_x_grad = ", ind, opt_x_grad.view(-1)[ind].item(), ref_x_grad.view(-1)[ind].item())

    """
    if ref_dtype == torch.bfloat16:
        ref_x_grad = x2.grad.to(torch.float)
        ref_weight_grad = torch_conv.weight.grad.to(torch.float)
        ref_y_fp32 = y2.to(torch.float)
    else:
        ref_x_grad = x2.grad
        ref_weight_grad = torch_conv.weight.grad
        ref_y_fp32 = y2
    """
    if has_bias:
        ref_bias_grad = torch_conv.bias.grad

    #for i in range(10):
    #    print("i opt_x_grad ref_x_grad = ", i, opt_x_grad.view(-1)[i].item(), ref_x_grad.view(-1)[i].item())

    # X gradient
    print("X Allclose: ", opt_x_grad.allclose(ref_x_grad, rtol=1e-4 if opt_dtype == torch.float else 1e-2, atol=1e-5))
    #print("(opt_x_grad - ref_x_grad).abs().sum()                                                      = ", (opt_x_grad - ref_x_grad).abs().sum())
    print("(opt_x_grad - ref_x_grad).norm(2)                                                          = ", (opt_x_grad - ref_x_grad).norm(2))
    xgrad_rel_norm_diff = (opt_x_grad - ref_x_grad).norm(2) / (opt_x_grad.norm(2))
    if (opt_dtype == torch.float and xgrad_rel_norm_diff > 1.0e-4) or (opt_dtype == torch.bfloat16 and xgrad_rel_norm_diff > 1.0e-2):
        print("warning, xgrad_rel_norm diff is too large, ", xgrad_rel_norm_diff)
    print("(opt_x_grad - ref_x_grad).norm(2) / ref_x_grad.norm                                         = ", (opt_x_grad - ref_x_grad).norm(2) / (ref_x_grad.norm(2)))
    print("(opt_x_grad - ref_x_grad).abs().norm(inf)                                                   = ", (opt_x_grad - ref_x_grad).norm(p=float('inf')))

    # X add gradient
    #print("XAdd Allclose: ", x1_add.grad.allclose(x2_add.grad, rtol=1e-5, atol=1e-5))

    # Bias gradient
    if has_bias:
      print("X Bias Allclose: ", ref_bias_grad.allclose(opt_bias_grad, rtol=1e-4 if opt_dtype == torch.float else 1e-2, atol=1e-6))

    # Weight gradient
    print("opt_weight_grad shape = ", opt_weight_grad.shape )
    #print("ref_weight_grad shape = ", ref_weight_grad.shape )

    #opt_weight_grad_blocked = blocked_layout.BlockedTensor(opt_weight_grad.shape, opt.)
    #print("type of opt_weight_grad = ", type(opt_weight_grad))
    #if type(opt_weight_grad) is BlockedParameter:
    #    opt_weight_grad_unblocked = opt_weight_grad.unblock()
    #else:
    #    opt_weight_grad_unblocked = opt_weight_grad

    if opt_weight_grad.dim() == 6:
        size = [opt_weight_grad.size(0)*opt_weight_grad.size(5), opt_weight_grad.size(1)*opt_weight_grad.size(4), opt_weight_grad.size(2), opt_weight_grad.size(3)]
        opt_weight_gradp = opt_weight_grad.permute([0,5,1,4,2,3]).contiguous().view(size)
        #print("opt_weight_gradp shape = ", opt_weight_gradp.shape)
        opt_weight_grad_unblocked = opt_weight_gradp
    elif opt_weight_grad.dim() == 7: #[K C R S bc/2 bk 2]
        size = [opt_weight_grad.size(0)*opt_weight_grad.size(5), opt_weight_grad.size(1)*opt_weight_grad.size(4)*opt_weight_grad.size(6), opt_weight_grad.size(2), opt_weight_grad.size(3)]
        opt_weight_gradp = opt_weight_grad.permute([0,5,1,4,6,2,3]).contiguous().view(size)
        #print("opt_weight_gradp shape = ", opt_weight_gradp.shape)
        opt_weight_grad_unblocked = opt_weight_gradp
    else:
        opt_weight_grad_unblocked = opt_weight_grad
    #print("opt_weight_grad_unblocked shape = ", opt_weight_grad_unblocked.shape)

    for i in range(10):
        ind = i
        print("ind opt_weight_grad ref_weight_grad = ", ind, opt_weight_grad_unblocked.view(-1)[ind].item(), ref_weight_grad.view(-1)[ind].item())

    print("X Wt Allclose: ", ref_weight_grad.allclose(opt_weight_grad_unblocked, rtol=1e-4 if opt_dtype == torch.float else 1e-2, atol=1e-6))

    """
    counter = 0
    counter_reldiff = 0
    for i in range(inc*outc*K*K):
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
    print("Stats: bad diffs     are X out of Y:", counter, inc*outc*K*K)
    print("Stats: bad reldiffs  are X out of Y:", counter_reldiff, inc*outc*K*K)
    """

    #print("opt_weight_grad_unblocked shape = ", opt_weight_grad_unblocked.shape)
    wgrad_rel_norm_diff = (opt_weight_grad_unblocked - ref_weight_grad).norm(2) / ref_weight_grad.norm(2)
    if (opt_dtype == torch.float and wgrad_rel_norm_diff > 1.0e-4) or (opt_dtype == torch.bfloat16 and wgrad_rel_norm_diff > 1.0e-2):
        print("warning, wgrad_rel_norm diff is too large, ", wgrad_rel_norm_diff)
    #for i in range(10):
    #    print("i opt_weight_grad_unblocked ref_weight_grad = ", i, opt_weight_grad_unblocked.view(-1)[i].item(), ref_weight_grad.view(-1)[i].item())

    print("(opt_weight_grad.permuted - ref_weight_grad).abs().norm(inf)               = ", (opt_weight_grad_unblocked - ref_weight_grad).norm(p=float('inf')))
    print("(opt_weight_grad.permuted - ref_weight_grad).norm(2) / torch.w.grad        = ", (opt_weight_grad_unblocked - ref_weight_grad).norm(2) / ref_weight_grad.norm(2))

    # Output (fwd)

    print("Y Allclose: ", opt_y_fp32.allclose(ref_y_fp32, rtol=1e-4 if opt_dtype == torch.float else 1e-2, atol=1e-6))
    y_rel_norm_diff = (opt_y_fp32 - ref_y_fp32).norm(2) / ref_y_fp32.norm(2)
    if (opt_dtype == torch.float and y_rel_norm_diff > 1.0e-4) or (opt_dtype == torch.bfloat16 and y_rel_norm_diff > 1.0e-2):
        print("warning, y_rel_norm diff is too large, ", y_rel_norm_diff)

    """
    if y1.dim() == 5: # for opt conv
      print("y1 size", y1.size())
      size = [y1.size(0), y1.size(1)*y1.size(4), y1.size(2), y1.size(3)]
      print("size", size)
      y1p = y1.permute([0,1,4,2,3]).contiguous().view(size)
      print("y1p size",y1p.size())
    print("(y1.permuted - y2).abs().norm(inf)             = ", (y1p - y2).abs().norm(p=float('inf')))
    """
    print("(y1 - y2).abs().norm(inf)              = ", (opt_y_fp32 - ref_y_fp32).abs().norm(p=float('inf')))
    print("(y1 - y2).abs().norm(2)   / y2.norm(2) = ", (opt_y_fp32 - ref_y_fp32).norm(2) / ref_y_fp32.norm(2))

    #print(opt_x_grad - ref_x_grad)
    #print(opt_weight_grad_unblocked)
    #print(ref_weight_grad)

    # Does not work at the moment for bwd
    if with_perf:

        # perf part for fwd
        # warmup
        for i in range(10):
            y1 = opt_conv(x1) #(xp)

        # main
        niter = 100
        time_start = time.time()
        for i in range(niter):
            y1 = opt_conv(x1) #(xp)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / niter

        time_per_fwd = time_per_iter
        print("time per fwd = ", time_per_iter)

        # perf part for bwd

        x1.requires_grad=True
        opt_conv.weight.requires_grad=False
        if has_bias:
            opt_conv.bias.requires_grad=False
        # warmup
        for i in range(10):
            #y1 = opt_conv(xp)
            #z1 = y1.mean() #z1.detach()
            #print(z1)
            #z1.backward(retain_graph=True)
            torch.autograd.backward(tensors=y1, grad_tensors=y1_grad, retain_graph=True)

        #exit()

        # main
        #y1_grad = y1.grad
        niter = 100
        time_start = time.time()
        for i in range(niter):
            #y1 = opt_conv(xp)
            #z1 = y1.mean() #z1.detach()
            z1.backward(retain_graph=True)
            #torch.autograd.backward(tensors=y1, inputs=x1.grad, grad_tensors = y1_grad, retain_graph=True)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / niter

        time_per_bwd = time_per_iter #- time_per_fwd
        print("time per bwd = ", time_per_bwd)

        # perf part for upd
        x1.requires_grad=False
        opt_conv.weight.requires_grad=True
        if has_bias:
            opt_conv.bias.requires_grad=False
        # warmup
        for i in range(10):
            #y1 = opt_conv(xp)
            #z1 = y1.mean() #z1.detach()
            z1.backward(retain_graph=True)
            #y1.backward(retain_graph=True)

        # main
        #weight1_grad = opt_weight_grad
        niter = 100
        time_start = time.time()
        for i in range(niter):
            #y1 = opt_conv(xp)
            #z1 = y1.mean() #z1.detach()
            z1.backward(retain_graph=True)
            #y1.backward(retain_graph=True)
            #torch.autograd.backward(tensors=y1,inputs=opt_conv.weight, grad_tensors = weight1_grad, retain_graph=True)
        time_end = time.time()
        time_per_iter = (time_end - time_start) / niter

        time_per_upd = time_per_iter #- time_per_fwd
        print("time per upd = ", time_per_upd)

    #exit()

def main():
    opt_dtype = torch.float if not args.use_bf16_opt else torch.bfloat16
    ref_dtype = torch.float if not args.use_bf16_ref else torch.bfloat16
    #with open("resnet50_conv_test_data.data") as f:
    #with open("resnet50_conv_test_data_extended.data") as f:
    #with open("resnet50_conv_test_data_extended_1thr.data") as f:
    #with open("resnet50_conv_test_data_extended_24thr_custom.data") as f:
    #with open("resnet50_conv_test_data_extended.data") as f:
    #with open("resnet50_conv_test_data_extended_new.data") as f:
    #with open("resnet50_conv_test_data_extended_new_28thr_reordered.data") as f:
    #with open("resnet50_conv_test_data_extended_new_28thr.data") as f:
    with open("resnet50_conv_test_data_for_bottleneck_28thr.data") as f:
        contents = f.readlines()
        for line in contents:
            #print("line")
            #print(type(line))
            #print(line)
            [N, H, W, inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode] = list(line.split(" "))
            #[inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode] = list(line.split(" "))
            string_list = list(line.strip().split(" "))
            has_bias=False if has_bias.strip() == 'False' else True
            padding_mode=padding_mode.strip()
            #print(string_list)
            #print(string_list[:7])
            #integer_map = map(int, string_list[:7])
            #print(string_list[:10])
            integer_map = map(int, string_list[:10])
            #print(integer_map)
            [N, H, W, inc, outc, K, stride, padding, dilation, groups] = list(integer_map)
            #[inc, outc, K, stride, padding, dilation, groups] = list(integer_map)
            #print(type(inc))
            #print(type(groups))
            #print(type(has_bias))
            run_test_conv(N, H, W, inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, args.with_perf, args.test_module)
    exit()

    # Just a single size run
    inc=3
    outc=64
    K=7
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

    run_test_conv(N, H, W, inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, args.with_perf, args.test_module)

if __name__ == "__main__":
    args = parser.parse_args()
    main()


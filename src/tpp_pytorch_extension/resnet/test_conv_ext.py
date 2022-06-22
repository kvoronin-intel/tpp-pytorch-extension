import argparse
import torch
import numpy as np

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

parser.add_argument("--with-perf", action="store_true", default=False, help='if true, measures performance additionally for the opt conv module', dest='with_perf')

parser.add_argument('--use-bf16-opt', action="store_true", default=False, dest='use_bf16_opt')
parser.add_argument('--use-bf16-ref', action="store_true", default=False, dest='use_bf16_ref')

parser.add_argument('--bc',  nargs='?', type=int)
parser.add_argument('--bk',  nargs='?', type=int)

#import pdb

global_counter = 0

#torch.autograd.set_detect_anomaly(True)

def run_test_conv(N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, with_perf, test_module):
    print("debug: run_test_conv called with N H W inc outc bc bk R stride padding dilation groups has_bias padding_mode opt_dtype ref_dtype with_perf test_module ",
            N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, with_perf, test_module)

    global global_counter

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

    y1 = opt_conv(x1) #xp)
    y2 = torch_conv(x2)

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

    # X gradient
    compare_padded_tensors(x1.grad, x2.grad, "X Grad", W, input_hw_padding, zero_rim_for_opt = True)

    # Bias gradient
    if has_bias:
      compare_padded_tensors(opt_conv.bias.grad, torch_conv.bias.grad, "Bias Grad")

    # Weight gradient
    compare_weight_grads( opt_conv.weight.grad, torch_conv.weight.grad, "conv")

    # Output (fwd)
    compare_padded_tensors(y1.unblocked_tensor(), y2, "Y", outW, output_hw_padding, zero_rim_for_opt = True)

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

    return
    #exit()

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
    with open("resnet50_conv_test_data_for_bottleneck_28thr.data") as f:
        contents = f.readlines()
        for line in contents:
            #print("line")
            #print(type(line))
            #print(line)
            [N, H, W, inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode] = list(line.split(" "))
            #[inc, outc, R, stride, padding, dilation, groups, has_bias, padding_mode] = list(line.split(" "))
            string_list = list(line.strip().split(" "))
            has_bias=False if has_bias.strip() == 'False' else True
            padding_mode=padding_mode.strip()
            #print(string_list)
            #print(string_list[:7])
            #integer_map = map(int, string_list[:7])
            #print(string_list[:10])
            integer_map = map(int, string_list[:10])
            #print(integer_map)
            [N, H, W, inc, outc, R, stride, padding, dilation, groups] = list(integer_map)
            #[inc, outc, R, stride, padding, dilation, groups] = list(integer_map)
            #print(type(inc))
            #print(type(groups))
            #print(type(has_bias))
            run_test_conv(N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, args.with_perf, args.test_module)
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

    run_test_conv(N, H, W, inc, outc, bc, bk, R, stride, padding, dilation, groups, has_bias, padding_mode, opt_dtype, ref_dtype, args.with_perf, args.test_module)

if __name__ == "__main__":
    args = parser.parse_args()
    main()


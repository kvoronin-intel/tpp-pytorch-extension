import argparse
import torch
#import pcl_cgbp
#import pcl_cgbp_cpp
import numpy as np

import pcl_pytorch_extension
from pcl_pytorch_extension._C import _batchnorm as batchnorm_cpp
import batchnorm as batchnorm_py

import pcl_cgbp
import pcl_cgbp_cpp

import test_utils
from test_utils import compare_weight_grads, compare_padded_tensors

"""
import sys, inspect

def print_classes():
    for name, obj in inspect.getmembers(sys.modules['pcl_pytorch_extension']):
        print(obj)
        #if inspect.isclass(obj):
        #    print(obj)

print_classes()

exit()
"""

#from batchnorm_py import DummyBatchNormTPP

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--test-module', default='cnn_tpp', type=str,
                    help='module to test against the reference', dest='test_module')

parser.add_argument("--with-perf", action="store_true", default=False, help='if true, measures performance additionally for the opt module', dest='with_perf')

parser.add_argument('--use-bf16-opt', action="store_true", default=False, dest='use_bf16_opt')
parser.add_argument('--use-bf16-ref', action="store_true", default=False, dest='use_bf16_ref')

torch.autograd.set_detect_anomaly(True)

def run_test_bn(N, H, W, C, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, with_perf, test_module):
    print("debug: run_test_bn called with N H W C opt_padding has_relu has_eltwise track_running_stats opt_dtype ref_dtype with_perf test_module ",
            N, H, W, C, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, with_perf, test_module)

    eps=1.e-7

    #opt_padding = [0, 0, 0, 0]

    if opt_padding != None and len(opt_padding) != 4:
        print("Error: padding should have four elements [pad_h_in, pad_w_in, pad_h_out, pad_w_out]")
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
        opt_bn = pcl_cgbp.XsmmBatchNormTPP(C, eps=eps, track_running_stats=track_running_stats, relu=has_relu, eltwise=has_eltwise, dtype=opt_dtype)
        hardcoded_bc=64
    elif test_module == 'ext_tpp':
        print("info: testing TPP module from extensions (pcl_pytorch_extension)")
        opt_bn = batchnorm_py.DummyBatchNormTPP(C, opt_padding, eps=eps, track_running_stats=track_running_stats, relu=has_relu, eltwise=has_eltwise, dtype=opt_dtype)
        hardcoded_bc=64
    else:
        print("test_module not supported, test_module = ", test_module)
        exit()
    print("info: hardcoded_bc = ", hardcoded_bc)
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

    """
    #x1 = torch.randn(N, C, H, W, requires_grad=True)
    #n  = torch.distributions.Normal(torch.tensor([4.0]), torch.tensor([0.5]))
    #x1 = n.sample((N,C,H,W), reguires_grad=True)
    x1 = torch.randn(N, C, H, W, requires_grad=True) * 0.5 + 4.0
    #print("x1 size = ", x1.size())
    #exit()
    x2 = x1.clone().detach().requires_grad_()
    """

    #x = torch.randn(N, C, H, W, requires_grad=True)
    #x = torch.randn(N, C, H, W, requires_grad=True) * 0.5 + 4.0
    #x = torch.ones_like(x, requires_grad=True)
    rates = torch.rand(N, C, H, W) * 5
    x = torch.poisson(rates)
    x.requires_grad_()
    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        x_bf16 = x.to(torch.bfloat16)
        x      = x_bf16.to(torch.float)
    #print("x shape = ", x.shape)

    """
    if has_eltwise == True:
        x1_add = torch.randn(N, C, H, W, requires_grad=True)
        x2_add = x1_add.clone().detach().requires_grad_()
    """
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
    if test_module == 'ext_tpp' and hasattr(batchnorm_cpp,'batchnorm_get_c_block'):
        bc = batchnorm_cpp.batchnorm_get_c_block(C)
    if test_module == 'cnn_tpp' and hasattr(pcl_cgbp_cpp,'bnorm_get_c_block'):
        bc = pcl_cgbp_cpp.bnorm_get_c_block(C)
    else:
        print("Warning: could not use batchnorm_cpp.batchnorm_get_c_block/pcl_cgbp_cpp.bnorm_get_c_block, hence used hardcoded block sizes in the test")
        if C % hardcoded_bc == 0:
          bc = hardcoded_bc
        else:
          bc = C
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

    print("x1.shape = ", x1.shape)
    print("x2.shape = ", x2.shape)

    #y1 = opt_conv(x1, x1_add)
    #y2 = relu(torch_conv(x2) + x2_add)
    if has_eltwise == True:
        #y1 = opt_bn(xp, xp_add)
        y1 = opt_bn(x1, x1_add)
    else:
        #y1 = opt_bn(xp)
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
    z1.backward(retain_graph=True,gradient=torch.tensor(1.*y1_numel / y2.numel(), dtype=torch.float))
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
    compare_padded_tensors(x1.grad, x2.grad, "X Grad", W, input_hw_padding)

    # X_ADD gradient
    if has_eltwise == True:
        compare_padded_tensors(x1_add.grad, x2_add.grad, "X ADD Grad", W, input_hw_padding)

    compare_weight_grads( opt_bn.weight.grad, torch_bn.weight.grad, "Weight Grad")

    """
    # Weight gradient
    print("W Allclose: ", opt_weight_grad.allclose(ref_weight_grad, rtol=1e-5, atol=1e-5))
    #print("(opt_bn.weight.data - torch_bn.weight.data).abs().norm(inf)                          = ", (opt_bn.weight.data - torch_bn.weight.data).norm(p=float('inf')))
    #print("(opt_bn.weight.data - original weight).abs().norm(inf)                               = ", (opt_bn.weight.data - weight).norm(p=float('inf')))
    #print("opt_weight_grad size")
    #print(opt_weight_grad.size())
    #print("ref_weight_grad size")
    #print(ref_weight_grad.size())
    wgrad_rel_norm_diff = (opt_weight_grad - ref_weight_grad).norm(2) / ref_weight_grad.norm(2)
    if wgrad_rel_norm_diff > 3.0e-6:
        print("warning, wgrad_rel_norm-diff is too large, ", wgrad_rel_norm_diff)
    print("(opt_weight_grad - ref_weight_grad).abs().norm(inf)               = ", (opt_weight_grad - ref_weight_grad).norm(p=float('inf')))
    print("(opt_weight_grad - ref_weight_grad).abs().norm(2) / torch.w.grad  = ", (opt_weight_grad - ref_weight_grad).norm(2) / ref_weight_grad.norm(2))
    #print("(opt_x_grad - ref_x_grad).abs().sum() / 64*3*7*7 = ", (opt_x_grad - ref_x_grad).reshape(-1).abs().sum()/((opt_x_grad - ref_x_grad).reshape(-1).size()))
    """

    # Bias
    compare_padded_tensors(opt_bn.bias.grad, torch_bn.bias.grad, "Bias Grad")

    # Out (Y)
    compare_padded_tensors(y1.unblocked_tensor(), y2, "Y", W, output_hw_padding)

    if track_running_stats == True:
        print("(opt_bn.running_mean - torch_bn.running_mean).abs().norm(inf)                    = ", (opt_bn.running_mean - torch_bn.running_mean).norm(p=float('inf')))
        print("(opt_bn.running_mean - torch_bn.running_mean).norm(2) / torch_bn.run_mean        = ", (opt_bn.running_mean - torch_bn.running_mean).norm(2) / torch_bn.running_mean.norm(2))
        print("(opt_bn.running_var  - torch_bn.running_var ).abs().norm(inf)                    = ", (opt_bn.running_var  - torch_bn.running_var) .norm(p=float('inf')))
        print("(opt_bn.running_var  - torch_bn.running_var ).norm(2) / torch_bn.run_var         = ", (opt_bn.running_var  - torch_bn.running_var ).norm(2) / torch_bn.running_var.norm(2))
        #print("opt_bn.running_mean   = ", opt_bn.running_mean);
        #print("opt_bn.running_var    = ", opt_bn.running_var);
        #print("torch_bn.running_mean = ", torch_bn.running_mean);
        #print("torch_bn.running_var  = ", torch_bn.running_var);

    if with_perf:
        print("Performance part is not implemented for this test!")

    return
    exit()

def main():
    opt_dtype = torch.float if not args.use_bf16_opt else torch.bfloat16
    ref_dtype = torch.float if not args.use_bf16_ref else torch.bfloat16
    
    with open("resnet50_bn_test_data_extended_new_28thr.data") as f:
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
            opt_padding = [4, 4, 6, 6] #[1, 1, 0, 0] #[0, 0, 1, 1] #[4, 4, 6, 6] #[0, 0, 0, 0] #[4, 4, 6, 6]
            run_test_bn(N, H, W, C, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, args.with_perf, args.test_module)
    exit()
    

    # Just a single size run
    N=24 #16
    H=2 #28
    W=2 #28
    C=64
    opt_padding = [4, 4, 6, 6]
    has_relu=False
    has_eltwise=False
    track_running_stats=False

    run_test_bn(N, H, W, C, opt_padding, has_relu, has_eltwise, track_running_stats, opt_dtype, ref_dtype, args.with_perf, args.test_module)

if __name__ == "__main__":
    args = parser.parse_args()
    main()


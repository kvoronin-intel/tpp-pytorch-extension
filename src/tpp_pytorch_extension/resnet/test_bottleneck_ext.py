import argparse
import torch
import numpy as np
import time

import pcl_pytorch_extension

from pcl_pytorch_extension._C import _bottleneck as bottleneck_cpp
import bottleneck as bottleneck_py
from pcl_pytorch_extension._C import _conv as conv_cpp
import conv as conv_py
from pcl_pytorch_extension._C import _batchnorm as batchnorm_cpp
import batchnorm as batchnorm_py

import pcl_cgbp
import pcl_cgbp_cpp

from pcl_cgbp import Bottleneck_base, BottleneckTPP, XsmmBatchNormTPP, XsmmGroupNormTPP, XsmmConv2dTPP

import blocked_layout
from blocked_layout import BlockedTensor, BlockedParameter

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--test-module', default='tpp_bottleneck', type=str,
                    help='module to test against the reference', dest='test_module')
parser.add_argument('--ref-module', default='pt_native', type=str,
                    help='module to be used as reference', dest='ref_module')

parser.add_argument("--with-perf", action="store_true", default=False, help='if true, measures performance additionally for the opt module', dest='with_perf')

parser.add_argument('--use-bf16-opt', action="store_true", default=False, dest='use_bf16_opt')
parser.add_argument('--use-bf16-ref', action="store_true", default=False, dest='use_bf16_ref')

parser.add_argument('--use-physical-3x3-padding', action="store_true", default=False, dest='use_physical_3x3_padding')

parser.add_argument('--use-groupnorm', action="store_true", default=False, dest='use_groupnorm')

#import pdb

# When physical padding is on, rims can be nans
#torch.autograd.set_detect_anomaly(True)

def gn_init(m, zero_init=False):
    #assert isinstance(m, nn.GroupNorm) or isinstance(m, pcl_cgbp.nn_GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()

def run_test_bottleneck(N, H, W, inc, outc, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm, opt_dtype, ref_dtype, with_perf, test_module, ref_module):
    print("debug: run_test_bottleneck called with N H W inc outc stride eps expansion has_downsample use_physical_3x3_padding use_groupnorm opt_dtype ref_dtype with_perf test_module ref_module",
            N, H, W, inc, outc, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm, opt_dtype, ref_dtype, with_perf, test_module, ref_module)

    pcl_cgbp.init_libxsmm()

    if opt_dtype == torch.bfloat16 or ref_dtype == torch.bfloat16:
        print("One of the modules is using bf16 hence padding input channels to an even number")
        if inc % 2 != 0:
            inc = inc + 1

    if (stride != 1 or inc != outc * expansion) and has_downsample == False:
        print("For these stride, inc, outc and expansion has_downsample must be True")
        exit()

    if (stride == 1 and inc == outc * expansion) and has_downsample == True:
        print("Warning: For these stride, inc, outc and expansion has_downsample should be False (are you sure you want to have it as True) in Resnet")

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
            opt_downsample1 = conv_py.DummyConv2dTPP(inc, outc * expansion, kernel_size=1, stride=stride, bias=False, dtype=opt_dtype)
            torch.manual_seed(0)
            if use_groupnorm:
                print("For test_module = ext_bottleneck groupnorm has not been implemented")
                #opt_downsample2 = XsmmGroupNormTPP(32, outc * expansion, eps, dtype=opt_dtype)
                #gn_init(opt_downsample2)
            else:
                #opt_bn = batchnorm_py.DummyBatchNormTPP(C, opt_padding, eps=eps, track_running_stats=track_running_stats, relu=has_relu, eltwise=has_eltwise, dtype=opt_dtype)
                opt_downsample2 = batchnorm_py.DummyBatchNormTPP(outc * expansion, padding=[0, 0, 0, 0],eps=eps, dtype=opt_dtype)
        #print("has_downsample = True has not been implemented")
        #exit()
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

    print("Saving initialized PT-based bottleneck")
    torch.save(torch_bottleneck.state_dict(), 'checkpoint_ref_bottleneck.pth.tar')

    if test_module == 'ext_bottleneck':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        opt_bottleneck = bottleneck_py.BottleneckTPP(inc, outc, eps, stride, use_physical_3x3_padding, opt_downsample1, opt_downsample2, use_groupnorm=use_groupnorm, dtype=opt_dtype)
    elif test_module == 'tpp_bottleneck':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        opt_bottleneck = pcl_cgbp.BottleneckTPP(inc, outc, eps, stride, use_physical_3x3_padding, opt_downsample1, opt_downsample2, use_groupnorm=use_groupnorm, dtype=opt_dtype)
    elif test_module == 'pt_tpp':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        opt_bottleneck = pcl_cgbp.Bottleneck_base(inc, outc, eps, stride, opt_downsample1, opt_downsample2, use_ref_conv=False, use_ref_norm=False, use_groupnorm=use_groupnorm, dtype=opt_dtype)
    else:
        print("test_module not supported, test_module = ", test_module)
        exit()

    print("Loading initialized bottleneck from a checkpoint checkpoint_ref_bottleneck.pth.tar")
    checkpoint = torch.load('checkpoint_ref_bottleneck.pth.tar')
    opt_bottleneck.load_state_dict(checkpoint)

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

    #for i in range(10):
    #    print("i opt_x_init ref_x_init = ", i, opt_x_init.view(-1)[i].item(), ref_x_init.view(-1)[i].item())
    #for i in range(10):
    #    print("i opt_weight_init ref_weight_init = ", i, opt_weight_init.view(-1)[i].item(), ref_weight_init.view(-1)[i].item())
    #if has_bias:
    #    for i in range(10):
    #          print("i opt_bias_init ref_bias_init = ", i, opt_bias_init.view(-1)[i].item(), ref_bias_init.view(-1)[i].item())

    print("running ref bottleneck forward")
    y2 = torch_bottleneck(x2)
    print("running opt bottleneck forward")
    y1 = opt_bottleneck(x1)

    print("y1 shape = ", y1.shape)
    print("y2 shape = ", y2.shape)

    for i in range(10):
        print("i y1 y2 = ", i, y1.view(-1)[i].item(), y2.view(-1)[i].item())
    for i in range(10):
        print("i y1 y2 = ", -i-1, y1.view(-1)[-i-1].item(), y2.view(-1)[-i-1].item())

    z1 = y1.mean()
    z2 = y2.mean()
    print("z1", z1)
    print("z2", z2)

    print("type(y1) = ", type(y1))
    print("type(y1.unblocked_tensor()) = ", type(y1.unblocked_tensor()))

    #if hasattr(y1,'to'): #type(y1) is BlockedTensor:
    #    opt_y_fp32 = y1.unblocked_tensor().to(torch.float)
    #else:
    #    opt_y_fp32 = y1.to(torch.float)
    #opt_y_fp32 = y1.unblocked_tensor().to(torch.float) if type(y1) is BlockedTensor else y1.to(torch.float)
    if test_module == 'tpp_bottleneck' or test_module == 'ext_bottleneck':
        opt_y_fp32 = y1.unblocked_tensor().to(torch.float)
    else:
        opt_y_fp32 = y1.to(torch.float)

    ref_y_fp32 = y2.to(torch.float)

    print("Y Allclose: ", opt_y_fp32.allclose(ref_y_fp32, rtol=1e-5, atol=1e-6))

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

    #return
    #exit()

    z1.backward(retain_graph=True)
    z2.backward(retain_graph=True)


    #exit()

    #for i in range(10):
    #    print("i opt_conv.weight.grad = ", i, opt_conv.weight.grad.view(-1)[i].item())

    opt_x_grad = x1.grad.to(torch.float)
    ref_x_grad = x2.grad.to(torch.float)

    # X gradient
    print("X Allclose: ", opt_x_grad.allclose(ref_x_grad, rtol=1e-5, atol=1e-5))
    #print("(opt_x_grad - ref_x_grad).abs().sum()                                                      = ", (opt_x_grad - ref_x_grad).abs().sum())
    print("(opt_x_grad - ref_x_grad).norm(2)                                                          = ", (opt_x_grad - ref_x_grad).norm(2))
    xgrad_rel_norm_diff = (opt_x_grad - ref_x_grad).norm(2) / (opt_x_grad.norm(2))
    if xgrad_rel_norm_diff > 3.0e-6:
        print("warning, xgrad_rel_norm diff is too large, ", xgrad_rel_norm_diff)
    print("(opt_x_grad - ref_x_grad).norm(2) / opt_x_grad.norm                                         = ", (opt_x_grad - ref_x_grad).norm(2) / (opt_x_grad.norm(2)))
    print("(opt_x_grad - ref_x_grad).abs().norm(inf)                                                   = ", (opt_x_grad - ref_x_grad).norm(p=float('inf')))

    for i in range(10):
        print("i opt_x_grad ref_x_grad = ", i, opt_x_grad.view(-1)[i].item(), ref_x_grad.view(-1)[i].item())

    #exit()

    opt_bn3_weight_grad = opt_bottleneck.bn3.weight.grad.to(torch.float)
    ref_bn3_weight_grad = torch_bottleneck.bn3.weight.grad.to(torch.float)

    if opt_bn3_weight_grad.dim() == 6:
        size = [opt_bn3_weight_grad.size(0)*opt_bn3_weight_grad.size(5), opt_bn3_weight_grad.size(1)*opt_bn3_weight_grad.size(4), opt_bn3_weight_grad.size(2), opt_bn3_weight_grad.size(3)]
        opt_bn3_weight_gradp = opt_bn3_weight_grad.permute([0,5,1,4,2,3]).contiguous().view(size)
        #print("opt_bn3_weight_gradp shape = ", opt_bn3_weight_gradp.shape)
        opt_bn3_weight_grad_unblocked = opt_bn3_weight_gradp
    elif opt_bn3_weight_grad.dim() == 7:
        size = [opt_bn3_weight_grad.size(0)*opt_bn3_weight_grad.size(5), opt_bn3_weight_grad.size(1)*opt_bn3_weight_grad.size(4)*opt_bn3_weight_grad.size(6), opt_bn3_weight_grad.size(2), opt_bn3_weight_grad.size(3)]
        opt_bn3_weight_gradp = opt_bn3_weight_grad.permute([0,5,1,4,6,2,3]).contiguous().view(size)
        #print("opt_bn3_weight_gradp shape = ", opt_bn3_weight_gradp.shape)
        opt_bn3_weight_grad_unblocked = opt_bn3_weight_gradp
    else:
        opt_bn3_weight_grad_unblocked = opt_bn3_weight_grad
    #print("opt_bn3_weight_grad_unblocked shape = ", opt_bn3_weight_grad_unblocked.shape)

    print("BN3 Wt Allclose: ", ref_bn3_weight_grad.allclose(opt_bn3_weight_grad_unblocked, rtol=1e-5, atol=1e-6))

    #print("opt_bn3_weight_grad_unblocked shape = ", opt_bn3_weight_grad_unblocked.shape)
    wgrad_rel_norm_diff = (opt_bn3_weight_grad_unblocked - ref_bn3_weight_grad).norm(2) / ref_bn3_weight_grad.norm(2)
    if wgrad_rel_norm_diff > 1.0e-5:
        print("warning, wgrad_rel_norm diff is too large, ", wgrad_rel_norm_diff)
    #for i in range(10):
    #    print("i opt_bn3_weight_grad_unblocked ref_bn3_weight_grad = ", i, opt_bn3_weight_grad_unblocked.view(-1)[i].item(), ref_bn3_weight_grad.view(-1)[i].item())

    print("(opt_bn3_weight_grad.permuted - ref_bn3_weight_grad).abs().norm(inf)               = ", (opt_bn3_weight_grad_unblocked - ref_bn3_weight_grad).norm(p=float('inf')))
    print("(opt_bn3_weight_grad.permuted - ref_bn3_weight_grad).norm(2) / torch.w.grad        = ", (opt_bn3_weight_grad_unblocked - ref_bn3_weight_grad).norm(2) / ref_bn3_weight_grad.norm(2))

    for i in range(10):
        print("i opt_bn3_weight_grad ref_bn3_weight_grad = ", i, opt_bn3_weight_grad_unblocked.view(-1)[i].item(), ref_bn3_weight_grad.view(-1)[i].item())

    return
    exit()

    if has_bias:
        opt_bias_grad = opt_conv.bias.grad
        ref_bias_grad = torch_conv.bias.grad


    # X add gradient
    #print("XAdd Allclose: ", x1_add.grad.allclose(x2_add.grad, rtol=1e-5, atol=1e-5))

    # Bias gradient
    if has_bias:
      print("X Bias Allclose: ", ref_bias_grad.allclose(opt_bias_grad, rtol=1e-5, atol=1e-6))

    # Weight gradient
    #print("opt_weight_grad shape = ", opt_weight_grad.shape )
    #print("ref_weight_grad shape = ", ref_weight_grad.shape )

    #opt_weight_grad_blocked = blocked_layout.BlockedTensor(opt_weight_grad.shape, opt.)
    #print("type of opt_weight_grad = ", type(opt_weight_grad))
    #if type(opt_weight_grad) is BlockedParameter:
    #    opt_weight_grad_unblocked = opt_weight_grad.unblock()
    #else:
    #    opt_weight_grad_unblocked = opt_weight_grad



    # Output (fwd)


    #print(opt_x_grad - ref_x_grad)
    #print(opt_weight_grad_unblocked)
    #print(ref_weight_grad)

    # Does not work at the moment for bwd
    if with_perf:
        print("Error: performance part is not implemented for this test!")
        exit()

    exit()

def main():
    opt_dtype = torch.float if not args.use_bf16_opt else torch.bfloat16
    ref_dtype = torch.float if not args.use_bf16_ref else torch.bfloat16
    
    #with open("resnet50_bottleneck_test_data_28thr.data") as f:
    #with open("resnet50_bottleneck_test_data_28thr_dbg.data") as f:
    #with open("resnet50_bottleneck_test_data_28thr_saved.data") as f:
    with open("resnet50_bottleneck_test_data_28thr.data") as f:
        contents = f.readlines()
        for line in contents:
            if line[0] == '#' or len(line) < 2:
                continue
            #print("line = ", line)
            #print(type(line))
            #print(line)
            #print("line split = ", line.split(" "))
            #print("list line split = ", list(line.split(" ")))
            preprocessed_line = " ".join(line.strip().split()) # to remove extra spaces in the input line
            #print("preprocessed_line = ", preprocessed_line)
            #print("list preprocessed_line = ", list(preprocessed_line))
            [N, H, W, inc, outc, stride, expansion, has_downsample, eps] = list(preprocessed_line.split(" ")) #list(line.split(" "))
            #[inc, outc, K, stride, padding, dilation, groups, has_bias, padding_mode] = list(line.split(" "))
            has_downsample = False if has_downsample.strip() == 'False' else True
            string_list = list(preprocessed_line.split(" ")) #list(line.strip().split(" "))
            #print(string_list)
            #print(string_list[:7])
            integer_map = map(int, string_list[:7])
            #print(integer_map)
            [N, H, W, inc, outc, stride, expansion] = list(integer_map)
            #print(type(inc))
            #print(type(has_downsample))
            eps = float(eps)
            print("eps, type(eps) = ", eps, type(eps))
            run_test_bottleneck(N, H, W, inc, outc, stride, eps, expansion, has_downsample, args.use_physical_3x3_padding, args.use_groupnorm,
                                opt_dtype, ref_dtype, args.with_perf, args.test_module, args.ref_module)
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

    run_test_bottleneck(N, H, W, inc, outc, stride, eps, expansion, has_downsample, args.use_physical_3x3_padding, args.use_groupnorm, opt_dtype, ref_dtype, args.with_perf, args.test_module, args.ref_module)

if __name__ == "__main__":
    args = parser.parse_args()
    main()


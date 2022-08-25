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

import test_utils
from test_utils import compare_weight_grads, compare_padded_tensors

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--test-module', default='tpp_bottleneck', type=str,
                    help='module to test against the reference', dest='test_module')
parser.add_argument('--ref-module', default='pt_native', type=str,
                    help='module to be used as reference', dest='ref_module')

parser.add_argument("--with-perf", action="store_true", default=False, help='if true, measures performance additionally for the opt module', dest='with_perf')

parser.add_argument('--use-bf16-opt', action="store_true", default=True, dest='use_bf16_opt')
parser.add_argument('--use-bf16-ref', action="store_true", default=False, dest='use_bf16_ref')

parser.add_argument('--use-physical-3x3-padding', action="store_true", default=False, dest='use_physical_3x3_padding')

parser.add_argument('--use-groupnorm', action="store_true", default=False, dest='use_groupnorm')

parser.add_argument('--use-hardcoded-tunings', action="store_true", default=False, dest='use_hardcoded_tunings')

parser.add_argument('--channel-block-size', type=int, default=None, dest='channel_block_size')

#import pdb

# When physical padding is on, rims can be nans
#torch.autograd.set_detect_anomaly(True)

def gn_init(m, zero_init=False):
    #assert isinstance(m, nn.GroupNorm) or isinstance(m, pcl_cgbp.nn_GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()

def run_test_bottleneck(N, H, W, inc, outc, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm, opt_dtype, ref_dtype, with_perf,
                        test_module, ref_module,
                        use_hardcoded_tunings, channel_block_size):
    print("debug: run_test_bottleneck called with N H W inc outc stride eps expansion has_downsample use_physical_3x3_padding use_groupnorm opt_dtype ref_dtype with_perf test_module ref_module use_hardcoded_tunings channel_block_size = ",
            N, H, W, inc, outc, stride, eps, expansion, has_downsample, use_physical_3x3_padding, use_groupnorm, opt_dtype, ref_dtype, with_perf, test_module, ref_module, use_hardcoded_tunings, channel_block_size)

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
            opt_downsample1 = pcl_cgbp.XsmmConv2dTPP(inc, outc * expansion, kernel_size=1, stride=stride, bias=False, dtype=opt_dtype, bc = channel_block_size, bk = channel_block_size)
            torch.manual_seed(0)
            if use_groupnorm:
                opt_downsample2 = pcl_cgbp.XsmmGroupNormTPP(32, outc * expansion, eps, dtype=opt_dtype)
                gn_init(opt_downsample2)
            else:
                opt_downsample2 = pcl_cgbp.XsmmBatchNormTPP(outc * expansion, eps, dtype=opt_dtype, bc = channel_block_size)
        elif test_module == 'ext_bottleneck':
            if not use_physical_3x3_padding:
                print("Error, for test_module = ext_bottleneck only physical padding for 3x3 convolutions is supported")
                exit()
            torch.manual_seed(0)
            opt_downsample1 = conv_py.DummyConv2dTPP(inc, outc * expansion, kernel_size=1, stride=stride, bias=False, dtype=opt_dtype,
                                                     bc=channel_block_size, bk=channel_block_size)
            torch.manual_seed(0)
            if use_groupnorm:
                print("For test_module = ext_bottleneck groupnorm has not been implemented")
                #opt_downsample2 = XsmmGroupNormTPP(32, outc * expansion, eps, dtype=opt_dtype)
                #gn_init(opt_downsample2)
            else:
                #opt_bn = batchnorm_py.DummyBatchNormTPP(C, opt_padding, eps=eps, track_running_stats=track_running_stats, relu=has_relu, eltwise=has_eltwise, dtype=opt_dtype)
                opt_downsample2 = batchnorm_py.DummyBatchNormTPP(outc * expansion, padding=[0, 0, 0, 0],eps=eps, dtype=opt_dtype, bc=channel_block_size)
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
        opt_bottleneck = bottleneck_py.BottleneckTPP(inc, outc, eps, stride, use_physical_3x3_padding, opt_downsample1, opt_downsample2, use_groupnorm=use_groupnorm, dtype=opt_dtype,
                                                      use_hardcoded_tunings=use_hardcoded_tunings,
                                                      bc_conv1=channel_block_size, bc_conv2=channel_block_size, bc_conv3=channel_block_size, bk_conv3=channel_block_size)
    elif test_module == 'tpp_bottleneck':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        opt_bottleneck = pcl_cgbp.BottleneckTPP(inc, outc, eps, stride, use_physical_3x3_padding, opt_downsample1, opt_downsample2, use_groupnorm=use_groupnorm, dtype=opt_dtype,
                                                bc_conv1=channel_block_size, bc_conv2=channel_block_size, bc_conv3=channel_block_size, bk_conv3=channel_block_size)
    elif test_module == 'pt_tpp':
        #with XsmmBatchNormTPP as torch.nn.BatchNorm2d, XsmmConv2dTPP as torch.nn.Conv2d:
        opt_bottleneck = pcl_cgbp.Bottleneck_base(inc, outc, eps, stride, opt_downsample1, opt_downsample2, use_ref_conv=False, use_ref_norm=False, use_groupnorm=use_groupnorm, dtype=opt_dtype,
                                                  channel_block_size=channel_block_size)
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

    print("running ref bottleneck forward")
    y2 = torch_bottleneck(x2)
    print("running opt bottleneck forward")
    y1 = opt_bottleneck(x1)

    print("y1 shape = ", y1.shape)
    print("y2 shape = ", y2.shape)

    z1 = y1.mean()
    z2 = y2.mean()
    print("z1", z1)
    print("z2", z2)

    print("x2 shape = ", x2.shape)
    print("y2 shape = ", y2.shape)

    # Y (Out)
    if opt_dtype == torch.bfloat16:
        rtol=1.5e-1
    else:
        rtol=5.0e-4
    atol=1e+0

    validation_check_failed1 = not compare_padded_tensors(y1.unblocked_tensor(), y2, "Y (Out)", rtol=rtol, atol=atol)

    z1.backward(retain_graph=True)
    z2.backward(retain_graph=True)
    #exit()

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
    if validation_checks_failed:
        print("Validation FAILED")
    else:
        print("Validation PASSED")

    return
    #exit()

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
                                opt_dtype, ref_dtype, args.with_perf, args.test_module, args.ref_module, args.use_hardcoded_tunings, args.channel_block_size)
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

    run_test_bottleneck(N, H, W, inc, outc, stride, eps, expansion, has_downsample, args.use_physical_3x3_padding, args.use_groupnorm, opt_dtype, ref_dtype, args.with_perf,
                        args.test_module, args.ref_module,
                        args.use_hardcoded_tunings, args.channel_block_size)

if __name__ == "__main__":
    args = parser.parse_args()
    main()


import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Function
import pcl_cgbp_cpp
import blocked_layout
from blocked_layout import BlockedModule
from blocked_layout import BlockedParameter
from blocked_layout import BlockedTensor # for debugging?
import numpy as np

#for debugging
import os

#torch.autograd.set_detect_anomaly(True)

# Shoudld only be used in the old code
#global_dtype = torch.float32

#global_tensor_bwd_counter = 0

from torch.nn import Conv2d      as nn_Conv2d
from torch.nn import GroupNorm   as nn_GroupNorm
from torch.nn import BatchNorm2d as nn_BatchNorm2d
from torch.nn import MaxPool2d   as nn_MaxPool2d
from torch.nn import AvgPool2d   as nn_AvgPool2d
from torch.nn import Linear      as nn_Linear

import torch.distributed as dist

#nn_Conv2d = torch.nn.Conv2d
#nn_GroupNorm = torch.nn.GroupNorm
#nn_BatchNorm2d = torch.nn.BatchNorm2d
#nn_MaxPool2d = torch.nn.MaxPool2d
#nn_AvgPool2d = torch.nn.AvgPool2d
#nn_Fc = torch.nn.Linear

#global_training_iteration = 0

def wait_for_debugger(rank):

    if os.getenv("MY_MPI_DEBUG") is not None and rank == 0:
        pcl_cgbp_cpp.wait_for_debugger_local(rank)

    dist.barrier()

    return

class XsmmConvHandle:
    def __init__(self, N, C, H, W, K, R, S, padding, stride, dtype):
        self.handle = pcl_cgbp_cpp.conv_create_handle(N, C, H, W, K, R, S, padding, stride, 1 if dtype == torch.float32 else 2)
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.K = K
        self.R = R
        self.S = S
        self.padding = padding
        self.stride = stride

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.conv_destroy_handle(self.handle)
            self.handle = None

class XsmmConvHandleTPP:
    def __init__(self, N, C, H, W, K, R, S, padding, stride, dtype, bc, bk):
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.K = K
        self.R = R
        self.S = S
        self.padding = padding
        self.stride = stride
        self.bc = bc
        self.bk = bk

        # Always logical padding here (notice that for bottleneck-based bf16 implementation this class is only for the first convolution)
        if dtype == torch.bfloat16:
            self.handle = pcl_cgbp_cpp.conv_setup_new(N, C, H, W, K, R, S, padding, padding, 0, 0, 0, 0, stride, 0 if dtype == torch.float else 1, self.bc if self.bc is not None else -1, self.bk if self.bk is not None else -1)
        else:
            self.handle = pcl_cgbp_cpp.conv_setup_new(N, C, H, W, K, R, S, padding, padding, 0, 0, 0, 0, stride, 0 if dtype == torch.float else 1, self.bc if self.bc is not None else -1, self.bk if self.bk is not None else -1)

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.conv_setup_destroy_new(self.handle)
            self.handle = None

class XsmmConv(Function):
    @staticmethod
    def forward(ctx, input, weight, handle, output_size):
        input = input.contiguous()
        weight = weight.contiguous()

        output = pcl_cgbp_cpp.conv_forward(handle.handle, input, weight, output_size)

        ctx.xsmm_handle = handle
        ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        handle = ctx.xsmm_handle
        del ctx.xsmm_handle

        input, weight = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        grad_input, grad_weight = pcl_cgbp_cpp.conv_backward(handle.handle, grad_output, input, weight)

        return (grad_input, grad_weight, None, None, None)

class XsmmConvTPP(Function):
    @staticmethod
    def forward(ctx, input, weight, handle, output_size):
        input = input.contiguous()
        weight = weight.contiguous()

        output = pcl_cgbp_cpp.conv_forward_new(handle.handle, input, weight, output_size)

        """
        print("debug: conv input shape = ",  input.shape)
        print("debug: conv output shape = ", output.shape)

        for i in range(10):
            ind = i + 0 #shift_output
            print("debug: i fwd input      = ", ind, input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_output
            print("debug: i fwd output     = ", ind, output.view(-1)[ind].item())
        """

        ctx.xsmm_handle = handle
        ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output):

        handle = ctx.xsmm_handle
        del ctx.xsmm_handle

        input, weight = ctx.saved_tensors

        grad_output = grad_output.contiguous()


        #for i in range(10):
        #    print("debug: i bwd input       = ", i, input.view(-1)[i].item())
        #for i in range(10):
        #    print("debug: i bwd weight      = ", i, weight.view(-1)[i].item())
        #for i in range(10):
        #    print("debug: i bwd grad_output = ", i, grad_output.view(-1)[i].item())

        """
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0

        if rank == 0:
            print("in conv (pcl_cgbp) before in bwd")
            print("debug: grad_output input weight types = ", grad_output.type(), input.type(), weight.type())
            print("nan check in pt for grad_output, nancount = ", torch.isnan(grad_output.view(-1)).sum(), hex(grad_output.data_ptr()), grad_output.shape)
        """

        if input.requires_grad:
          grad_input, grad_weight = pcl_cgbp_cpp.conv_backward_new(handle.handle, grad_output, input, weight)
        else:
          grad_input    = None
          [grad_weight] = pcl_cgbp_cpp.conv_backward_new(handle.handle, grad_output, input, weight)

        """
        if rank == 0:
            print("in conv (pcl_cgbp) after in bwd")
            grad_weight_nan_count = torch.isnan(grad_weight.view(-1)).sum()
            print("nan check in pt for grad_weight, nancount = ", grad_weight_nan_count, hex(grad_weight.data_ptr()), grad_weight.shape)
            if grad_input is not None:
                grad_input_nan_count = torch.isnan(grad_input.view(-1)).sum()
                print("nan check in pt for grad_input, nancount = ", grad_input_nan_count, hex(grad_input.data_ptr()), grad_input.shape)
            print("in conv (pcl_cgbp) after in bwd, before returning")
        """
        """
        for i in range(10):
            ind = i + 0 #shift_output
            print("debug: i bwd input      = ", ind, input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_weight
            print("debug: i bwd weight     = ", ind, weight.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_grad_output
            print("debug: i bwd gradout    = ", ind, grad_output.view(-1)[ind].item())

        if input.requires_grad:
            for i in range(10):
                ind = i + 0 #shift_grad_weight
                print("debug: ind bwd grad_input       = ", ind, grad_input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_grad_weight
            print("debug: ind bwd grad_weight      = ", ind, grad_weight.view(-1)[ind].item())

        #for i in range(10):
        #    print("debug: i grad_weight = ", i, grad_weight.view(-1)[i].item())
        """

        return (grad_input, grad_weight, None, None, None)

class XsmmConv2dTPP(BlockedModule, nn_Conv2d):
    r"""PCL Conv2d module for using libxsmm Conv TPP"""

    # Parameters logical_padding and use_hardcoded_tunings have been added to ensure compatibility with the convolution module in PCL PT extension and are not really used
    # Caution: parameters bc and bk can get ignored inside the conv handle constructor routines
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', dtype=torch.float,
                  bc=None, bk=None, logical_padding=True, use_hardcoded_tunings=None):
        self.C            = in_channels
        self.C_pad        = in_channels
        self.bc           = bc
        self.bk           = bk

        self.dtype        = dtype
        self.use_bf16     = True if self.dtype == torch.bfloat16 else False

        self.low_prec_vnni_blocking = pcl_cgbp_cpp.get_vnni_blocking(dtype)

        if not (self.dtype == torch.float32 or (self.dtype == torch.bfloat16 and self.low_prec_vnni_blocking == 2)):
            print("Error: XsmmConv2dTPP (at least the Python wrapper part) currently only supports fp32 or bf16 with VNNI2 format")
            exit()

        if self.use_bf16 and self.C_pad%2 != 0:
          self.C_pad = self.C_pad + 1

        if logical_padding is not True:
            print("Error: logical_padding must be True for XsmmConv2dTPP but got ", logical_padding)
            exit()

        if use_hardcoded_tunings is not None and use_hardcoded_tunings is not False:
            print("Warning: use_hardcoded_tunings is ignored by XsmmConv2dTPP (it is already meaningful for PCL PT conv but was set to ", use_hardcoded_tunings)

        #super(XsmmConv2dTPP, self).__init__()
        #nn_Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=None, dtype=dtype)
        nn_Conv2d.__init__(self, self.C_pad, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=None, dtype=dtype)

        self.N            = 0
        self.K            = out_channels
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = _pair(kernel_size)
        self.R            = kernel_size
        self.S            = kernel_size
        self.stride       = _pair(stride)
        self.padding      = _pair(padding)
        self.strides      = stride
        self.pads         = padding
        self.dilation     = _pair(dilation)

        self.weight = BlockedParameter(self.weight.data)
        if bias:
          self.bias = Parameter(torch.empty(self.K))

        #print("debug: C_pad dtype use_bf16 = ", self.C_pad, self.dtype, self.use_bf16)

        if self.bc is None or self.bk is None:
            [self.Cblock, self.Kblock, self.lp_block] = pcl_cgbp_cpp.conv_get_feature_map_blocks(self.C_pad, self.K, 0 if self.dtype == torch.float else 1)
        else:
            self.lp_block = 1 if self.dtype == torch.float else 2
            self.Cblock   = self.bc
            self.Kblock   = self.bk
        print("debug: Cblock, Kblock, lp_block = ", self.Cblock, self.Kblock, self.lp_block)

        if self.use_bf16:
            self.weight.set_blocking_param(
                (
                    [self.Kblock, [self.Cblock // self.lp_block, self.lp_block] , None, None],
                    [0, 2, 5, 6, 3, 1, 4],
                )
            )
            # K 0 1 | C 2 3 4 | R 5 | S 6 -> KCRSck2c, lp_block = 2
        else:
            self.weight.set_blocking_param(
                (
                    [self.Kblock, self.Cblock, None, None],
                    [0, 2, 4, 5, 3, 1],
                )
            )

        self.blocked_input_signature = blocked_layout.get_blocking_signature(
            "NCHW", "NCHWC"
        )
        self.blocked_output_signature = self.blocked_input_signature

    def maybe_block_params(self):
        self.weight.block()
        #self.bias.block()

    def forward(self, input):
        #print('Conv Input {} Padding:{} Stride:{}'.format(input.shape, self.padding, self.stride))
        self.maybe_block_params()

        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        #input = input.to(global_dtype).contiguous()
        #weight = self.weight.to(global_dtype) #switched to self.weight below

        #if self.C != self.C_pad:
        if input.shape[1] != self.C_pad:
          pad_shape = list(input.shape)
          #pad_shape[-1] = 1
          #pad_shape[1] = 1
          pad_shape[1] = self.C_pad - input.shape[1]
          #print("debug: pad_shape = ", pad_shape)
          self.zero_pad = input.new_zeros(pad_shape)
          #np.savetxt('my_conv1_forward_special_input_zero_pad_tst_rank_0.txt', self.zero_pad.contiguous().view(-1).detach().to(torch.float).numpy())

        #if self.C != self.C_pad:
        if input.shape[1] != self.C_pad:
          #input = torch.cat((input, self.zero_pad), dim=-1)
          input = torch.cat((input, self.zero_pad), dim=1)

        #np.savetxt('my_conv1_forward_special_input_tst_rank_0.txt', input.contiguous().view(-1).detach().to(torch.float).numpy())

        if N != self.N:
          if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
              print("Conv Create handle: ", N, self.C_pad, self.H, self.W, self.K, self.R, self.S, self.pads, self.strides, self.dtype)
          #self.xsmm_handle = XsmmConvHandle(N, self.C_pad, self.H, self.W, self.K, self.R, self.S, self.pads, self.strides, input.dtype)
          self.xsmm_handle = XsmmConvHandleTPP(N, self.C_pad, self.H, self.W, self.K, self.R, self.S, self.pads, self.strides, self.dtype, self.bc, self.bk)
          self.N = N

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.Cblock, None, None],
        )

        #output_size = pcl_cgbp_cpp.get_conv_tensor_layout(self.xsmm_handle.handle, "output");

        inH = input.size(2)
        inW = input.size(3)
        outH = (inH + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)//self.strides + 1
        outW = (inW + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)//self.strides + 1
        output_size  = [self.N, self.K//self.Kblock, outH, outW, self.Kblock]

        output = XsmmConvTPP.apply(blocked_input, self.weight, self.xsmm_handle, output_size)

        blocked_output = blocked_layout.BlockedTensor(output, self.blocked_output_signature)

        #for i in range(10):
        #    print("i blocked_output = ", i, blocked_output.unblocked_tensor().view(-1)[i].item())


        return blocked_output


class XsmmConv2d(nn.Module):
    r"""PCL Conv2d module for using libxsmm Conv"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', dtype=torch.float):
        super(XsmmConv2d, self).__init__()
        print(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.N = 0
        self.C = in_channels
        self.C_pad = in_channels
        self.K = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        N = 1
        H = 1
        W = 1
        self.kernel_size = _pair(kernel_size)
        self.R = kernel_size
        self.S = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.strides = stride
        self.pads = padding
        self.dilation = _pair(dilation)
        self.dtype    = dtype

        if self.dtype == torch.bfloat16 and self.C_pad%2 != 0:
          self.C_pad = self.C_pad + 1

        self.xsmm_handle = XsmmConvHandle(N, self.C_pad, H, W, self.K, self.R, self.S, self.pads, self.strides, self.dtype)
        wt_dims = pcl_cgbp_cpp.get_conv_tensor_layout(self.xsmm_handle.handle, 'weight')
        self.weight = Parameter(torch.empty(wt_dims))
        if torch.distributed.is_initialized():
          if torch.distributed.get_rank() == 0:
            print(f'global dtype = {global_dtype}')
            print(f'wt_dims = {wt_dims}')
            print(f'weight shape = {self.weight.shape}')

        if bias:
          self.bias = Parameter(torch.empty(self.K))
        self.reset_parameters()

    def reset_parameters(self):
      self.weight.data.fill_(0.)

    def forward(self, input):
        #print('Conv Input {} Padding:{} Stride:{}'.format(input.shape, self.padding, self.stride))
        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        input = input.to(self.dtype).contiguous()
        weight = self.weight.to(self.dtype)

        if N != self.N:
          if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
              print("Conv Create handle: ", N, self.C_pad, self.H, self.W, self.K, self.R, self.S, self.pads, self.strides, input.dtype)
          self.xsmm_handle = XsmmConvHandle(N, self.C_pad, self.H, self.W, self.K, self.R, self.S, self.pads, self.strides, input.dtype)
          self.N = N

          if self.C != self.C_pad:
            pad_shape = list(input.shape)
            pad_shape[-1] = 1
            self.zero_pad = input.new_zeros(pad_shape)

        if self.C != self.C_pad:
          input = torch.cat((input, self.zero_pad), dim=-1)

        output_size = pcl_cgbp_cpp.get_conv_tensor_layout(self.xsmm_handle.handle, "output");
        output =  XsmmConv.apply(input, weight, self.xsmm_handle, output_size)

        return output

class XsmmGNHandle:
    def __init__(self, N, C, H, W, G, relu=False, eltwise=False, dtype=torch.float):
        self.handle = pcl_cgbp_cpp.fusedgroupnorm_create_handle(N, C, H, W, G, 1 if dtype == torch.float32 else 2, 1 if relu == True else 0, 1 if eltwise == True else 0)
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.G = G
        self.relu = relu
        self.eltwise = eltwise

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.fusedgroupnorm_destroy_handle(self.handle)
            self.handle = None

class XsmmGNHandleTPP:
    def __init__(self, N, C, H, W, G, eps, relu=False, eltwise=False, dtype=torch.float):
        self.N       = N
        self.C       = C
        self.H       = H
        self.W       = W
        self.G       = G
        self.eps     = eps
        self.relu    = relu
        self.eltwise = eltwise

        self.fuse_type = -1
        if relu == True:
            if eltwise == True:
                self.fuse_type = 5 # elwise+relu+mask
            else:
                self.fuse_type = 4 # relu+mask
        else: # relu = False
            if eltwise == True:
                self.fuse_type = 2 # eltwise+no mask
            else:
                self.fuse_type = 0 # no fusion

        if self.fuse_type == -1:
            print("unsupported fuse_type is requested")
            exit()

        # Always no padding here
        self.handle = pcl_cgbp_cpp.gnorm_setup_new(N, C, H, W, G, 0, 0, 0, 0, self.eps, self.fuse_type, 0 if dtype == torch.float else 1)

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.gnorm_setup_destroy_new(self.handle)
            self.handle = None

class XsmmGN(Function):
    @staticmethod
    def forward(ctx, input, input_add, weight, bias, handle, stats_size, output_size):

        input = input.contiguous()
        if input_add.numel() > 0:
          input_add = input_add.contiguous()
        else:
          input_add = torch.Tensor()

        weight = weight.contiguous()
        bias = bias.contiguous()

        output, mean, var, invstd, relu_mask = pcl_cgbp_cpp.gnorm_forward(handle.handle, input, input_add, weight, bias, stats_size, output_size)

        ctx.xsmm_handle = handle
        ctx.save_for_backward(input, input_add, weight, output, mean, var, invstd, relu_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.xsmm_handle
        del ctx.xsmm_handle
        input, input_add, weight, output, mean, var, invstd, relu_mask = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_input, grad_input_add, grad_weight, grad_bias = pcl_cgbp_cpp.gnorm_backward(handle.handle, grad_output, input, input_add, weight, output, mean, var, invstd, relu_mask)

        del output, mean, var, invstd, relu_mask
        return (grad_input, grad_input_add, grad_weight, grad_bias, None, None, None)

class XsmmGNTPP(Function):
    @staticmethod
    def forward(ctx, input, input_add, weight, bias, mean, var, invstd, handle, output_size):

        #input = input.contiguous()
        #if input_add.numel() > 0:
        #  input_add = input_add.contiguous()
        #else:
        #  input_add = torch.Tensor()

        weight = weight.contiguous()
        bias = bias.contiguous()

        output, relu_mask = pcl_cgbp_cpp.gnorm_forward_new(handle.handle, input, input_add, weight, bias, mean, var, invstd, output_size)
        #output, mean, var, relu_mask = pcl_cgbp_cpp.gnorm_forward_new(handle.handle, input, input_add, weight, bias, stats_size, output_size)

        ctx.xsmm_handle = handle
        ctx.save_for_backward(input, input_add, weight, output, mean, var, invstd, relu_mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.xsmm_handle
        del ctx.xsmm_handle
        input, input_add, weight, output, mean, var, invstd, relu_mask = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_input, grad_input_add, grad_weight, grad_bias = pcl_cgbp_cpp.gnorm_backward_new(handle.handle, grad_output, input, input_add, weight, output, mean, var, invstd, relu_mask)

        #del output, mean, var, relu_mask
        del output, mean, var, invstd, relu_mask
        return (grad_input, grad_input_add, grad_weight, grad_bias, None, None, None, None, None)

class XsmmGroupNorm(nn.Module):
    r"""PCL GroupNorm module for using libxsmm GN"""

    def __init__(self, num_groups, num_channels, eps, affine=True, relu=False, eltwise=False):
        super(XsmmGroupNorm, self).__init__()
        C = num_channels
        G = num_groups
        self.C = C
        self.G = G
        self.N = 0
        self.xsmm_handle = None
        self.relu = relu
        self.eltwise = eltwise
        self.weight = Parameter(torch.Tensor(C))
        self.bias = Parameter(torch.Tensor(C))
        self.reset_parameters()

    def reset_parameters(self):
      self.weight.data.fill_(1.)
      self.bias.data.zero_()

    def forward(self, input, input_add=None):
        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        if input_add == None:
          input_add = torch.Tensor()

        input = input.contiguous()
        if self.eltwise:
          input_add = input_add.contiguous()

        if N != self.N:
            print("GN Create handle: ", N, self.C, self.H, self.W, self.G, self.relu, self.eltwise, input.dtype)
            self.xsmm_handle = XsmmGNHandle(N, self.C, self.H, self.W, self.G, self.relu, self.eltwise, input.dtype)
            self.N = N

        stats_size = pcl_cgbp_cpp.get_gn_tensor_layout(self.xsmm_handle.handle, "mean");
        output_size = pcl_cgbp_cpp.get_gn_tensor_layout(self.xsmm_handle.handle, "output");
        output =  XsmmGN.apply(input, input_add, self.weight, self.bias, self.mean, self.var, self.xsmm_handle, stats_size, output_size)

        return output

class XsmmGroupNormTPP(BlockedModule, nn_GroupNorm):
    r"""PCL GroupNorm module for using libxsmm GN with TPP"""

    def __init__(self, num_groups, num_channels, eps, affine=True, relu=False, eltwise=False, dtype=torch.float):
        nn_GroupNorm.__init__(self, num_groups, num_channels, eps, affine, device=None, dtype=dtype)
        C = num_channels
        G = num_groups
        self.C = C
        self.G = G
        self.N = 0
        self.xsmm_handle = None
        self.relu = relu
        self.eltwise = eltwise
        self.weight = Parameter(torch.Tensor(C))
        self.bias = Parameter(torch.Tensor(C))
        self.reset_parameters()
        self.eps = eps
        self.dtype = dtype

        self.blocked_input_signature = blocked_layout.get_blocking_signature(
            "NCHW", "NCHWC"
        )
        self.blocked_output_signature = self.blocked_input_signature

    def reset_parameters(self):
      self.weight.data.fill_(1.)
      self.bias.data.zero_()

    def forward(self, input, input_add=None):
        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        if input_add == None:
          input_add = torch.Tensor()

        #input = input.contiguous()
        #if self.eltwise:
        #  input_add = input_add.contiguous()

        if N != self.N:
            #print("GN Create handle: ", N, self.C, self.H, self.W, self.G, self.eps, self.relu, self.eltwise, self.dtype)
            #self.xsmm_handle = XsmmGNHandle(N, self.C, self.H, self.W, self.G, self.relu, self.eltwise, input.dtype)
            self.xsmm_handle = XsmmGNHandleTPP(N, self.C, self.H, self.W, self.G, self.eps, self.relu, self.eltwise, self.dtype)
            self.N = N
            self.Cblock = pcl_cgbp_cpp.gnorm_get_c_block(self.C)
            self.mean   = torch.empty(N, self.G)
            self.var    = torch.empty(N, self.G)
            self.invstd = torch.empty(N, self.G)

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.Cblock, None, None],
        )
        if self.eltwise:
            blocked_input_add = self.get_blocked_tensor(
                input_add,
                self.blocked_input_signature,
                [None, self.Cblock, None, None],
            )
        else:
            blocked_input_add = input_add

        #stats_size  = [self.N, self.G]
        output_size = [self.N, self.C // self.Cblock, self.H, self.W, self.Cblock]

        #output =  XsmmGN.apply(input, input_add, self.weight, self.bias, self.xsmm_handle, stats_size, output_size)
        output =  XsmmGNTPP.apply(blocked_input, blocked_input_add, self.weight, self.bias, self.mean, self.var, self.invstd, self.xsmm_handle, output_size)

        blocked_output = blocked_layout.BlockedTensor(output, self.blocked_output_signature)

        return blocked_output


class XsmmBNHandle:
    def __init__(self, N, C, H, W, relu=False, eltwise=False, train=True, dtype=torch.float32):
        self.handle = pcl_cgbp_cpp.fusedbatchnorm_create_handle(N, C, H, W, 1 if dtype == torch.float32 else 2, relu, eltwise, train)
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.relu = relu
        self.eltwise = eltwise
        self.train = train

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.fusedbatchnorm_destroy_handle(self.handle)
            self.handle = None

class XsmmBNHandleTPP:
    def __init__(self, N, C, H, W, eps, relu=False, eltwise=False, train=True, dtype=torch.float32, bc=None ):

        self.N       = N
        self.C       = C
        self.H       = H
        self.W       = W
        self.eps     = eps
        self.relu    = relu
        self.eltwise = eltwise
        self.train   = train
        self.bc      = bc

        self.fuse_type = -1
        if relu == True:
            if eltwise == True:
                self.fuse_type = 5 # elwise+relu+mask
            else:
                self.fuse_type = 4 # relu+mask
        else: # relu = False
            if eltwise == True:
                self.fuse_type = 2 # eltwise+no mask
            else:
                self.fuse_type = 0 # no fusion

        if self.fuse_type == -1:
            print("unsupported fuse_type is requested")
            exit()

        # Always no padding here
        self.handle = pcl_cgbp_cpp.bnorm_setup_new(N, C, H, W, 0, 0, 0, 0, self.eps, self.fuse_type, 0 if dtype == torch.float else 1, self.bc if self.bc is not None else -1)

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.bnorm_setup_destroy_new(self.handle)
            self.handle = None

class XsmmBN(Function):
    @staticmethod
    def forward(ctx, input, input_add, weight, bias, mean, var, invstd, handle, output_size, training):

        weight = weight.contiguous()
        bias = bias.contiguous()

        output, relu_mask = pcl_cgbp_cpp.bnorm_forward(handle.handle, input, input_add, weight, bias, mean, var, invstd, output_size) 

        if training:
          ctx.xsmm_handle = handle
          ctx.save_for_backward(input, input_add, weight, output, mean, var, invstd, relu_mask)

        return output


    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.xsmm_handle
        del ctx.xsmm_handle
        input, input_add, weight, output, mean, var, invstd, relu_mask = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_input, grad_input_add, grad_weight, grad_bias = pcl_cgbp_cpp.bnorm_backward(handle.handle, grad_output, input, input_add, weight, output, mean, var, invstd, relu_mask)

        return (grad_input, grad_input_add, grad_weight, grad_bias, None, None, None, None, None, None, None)


class XsmmBNTPP(Function):
    @staticmethod
    def forward(ctx, input, input_add, weight, bias, mean, var, invstd, handle, output_size, training):

        weight = weight.contiguous()
        bias = bias.contiguous()

        norm_type = 0 if training else 1
        output, relu_mask = pcl_cgbp_cpp.bnorm_forward_new(handle.handle, input, input_add, weight, bias, mean, var, invstd, output_size, norm_type)

        if training:
          ctx.xsmm_handle = handle
          ctx.save_for_backward(input, input_add, weight, output, mean, var, invstd, relu_mask)
        """
        #print("debug: fwd relu eltwise eps = ", relu, eltwise, eps)
        for i in range(10):
            print("debug: i fwd input      = ", i, input.view(-1)[i].item())
        for i in range(10):
            print("debug: i fwd mean       = ", i, mean.view(-1)[i].item())
        for i in range(10):
            print("debug: i fwd var        = ", i, var.view(-1)[i].item())
        for i in range(10):
            print("debug: i fwd output     = ", i, output.view(-1)[i].item())
        """
        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.xsmm_handle
        del ctx.xsmm_handle
        input, input_add, weight, output, mean, var, invstd, relu_mask = ctx.saved_tensors

        grad_output = grad_output.contiguous()
        grad_input, grad_input_add, grad_weight, grad_bias = pcl_cgbp_cpp.bnorm_backward_new(handle.handle, grad_output, input, input_add, weight, output, mean, var, invstd, relu_mask)
        """
        for i in range(10):
            print("debug: i bwd grad_output      = ", i, grad_output.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd input            = ", i, input.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd weight           = ", i, weight.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd grad_input       = ", i, grad_input.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd grad_weight      = ", i, grad_weight.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd grad_bias        = ", i, grad_bias.view(-1)[i].item())
        """
        """
        print("grad_output dtype = ", grad_input.dtype)
        print("nan check in pt for grad_output, nan count = ", torch.isnan(grad_output.view(-1)).sum())
        print("input dtype = ", input.dtype)
        print("nan check in pt for input, nan count = ", torch.isnan(input.view(-1)).sum())

        print("grad_input dtype = ", grad_input.dtype)
        print("nan check in pt for grad_input, nan count = ", torch.isnan(grad_input.view(-1)).sum())
        print("nan check in pt for grad_input_add, nan count = ", torch.isnan(grad_input_add.view(-1)).sum())
        print("nan check in pt for grad_weight, nan count = ", torch.isnan(grad_weight.view(-1)).sum())
        print("nan check in pt for grad_bias, nan count = ", torch.isnan(grad_bias.view(-1)).sum())
        """

        return (grad_input, grad_input_add, grad_weight, grad_bias, None, None, None, None, None, None, None)


class XsmmBatchNorm(nn.Module):
    r"""PCL batchNorm module for using libxsmm BN"""

    def __init__(self, num_channels, eps, momentum=0.1, affine=True, track_running_stats=True, relu=False, eltwise=False):
        super(XsmmBatchNorm, self).__init__()
        C = num_channels
        self.C = C
        self.N = 0
        self.affine = affine
        self.momentum = momentum
        self.xsmm_handle = None
        self.train_handle = None
        self.eval_handle = None
        self.relu = relu
        self.eltwise = eltwise
        self.track_running_stats = track_running_stats
        self.mean = torch.empty(C)
        self.var = torch.empty(C)
        self.invstd = torch.empty(C)

        if self.affine:
          self.weight = Parameter(torch.Tensor(C))
          self.bias = Parameter(torch.Tensor(C))
        else:
          self.register_parameter('weight', None)
          self.register_parameter('bias', None)

        if self.track_running_stats:
          self.register_buffer('running_mean', torch.zeros(C))
          self.register_buffer('running_var', torch.ones(C))
          self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
          self.register_parameter('running_mean', None)
          self.register_parameter('running_var', None)
          self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
          self.running_mean.zero_()
          self.running_var.fill_(1)
          self.num_batches_tracked.zero_()

    def reset_parameters(self):
      self.reset_running_stats()
      if self.affine:
        self.weight.data.fill_(1.)
        self.bias.data.zero_()

    def forward(self, input, input_add=None):

        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        if input_add == None:
          input_add = torch.Tensor()

        input = input.contiguous()
        if self.eltwise:
          input_add = input_add.contiguous()

        if N != self.N:
          self.xsmm_handle = XsmmBNHandle(N, self.C, self.H, self.W, self.relu, self.eltwise, self.training, input.dtype)
          self.N = N

        if self.training and self.track_running_stats:
          if self.num_batches_tracked is not None:
            self.num_batches_tracked = self.num_batches_tracked + 1

        output_size = pcl_cgbp_cpp.get_bn_tensor_layout(self.xsmm_handle.handle, "output");
        output =  XsmmBN.apply(input, input_add, self.weight, self.bias, self.mean, self.var, self.invstd, self.xsmm_handle, output_size, self.training)

        return output

class XsmmBatchNormTPP(BlockedModule, nn_BatchNorm2d):
    r"""PCL batchNorm TPP module for using libxsmm BN"""

    def __init__(self, num_channels, eps, momentum=0.1, affine=True, track_running_stats=True, relu=False, eltwise=False, dtype=torch.float32, bc = None):
        #super(XsmmBatchNormTPP, self).__init__()
        nn_BatchNorm2d.__init__(self, num_channels, eps, momentum, affine, track_running_stats, device=None, dtype=dtype)

        #print("dtype in XsmmBatchNormTPP constructor weight.dtype = ", dtype, self.weight.dtype)

        C = num_channels
        self.C = C
        self.N = 0
        self.affine = affine
        self.momentum = momentum
        self.xsmm_handle = None
        self.train_handle = None
        self.eval_handle = None
        self.relu = relu
        self.eltwise = eltwise
        self.track_running_stats = track_running_stats
        self.mean = torch.empty(C)
        self.var = torch.empty(C)
        self.invstd = torch.empty(C)
        self.eps = eps
        self.dtype = dtype
        self.bc    = bc

        if self.bc is None:
            self.Cblock = pcl_cgbp_cpp.bnorm_get_c_block(self.C)
        else:
            self.Cblock = bc

        self.blocked_input_signature = blocked_layout.get_blocking_signature(
            "NCHW", "NCHWC"
        )
        self.blocked_output_signature = self.blocked_input_signature

        if self.affine:
          self.weight = Parameter(torch.Tensor(C))
          self.bias   = Parameter(torch.Tensor(C))
          #print("for self.affine = True dtype in XsmmBatchNormTPP constructor weight.dtype = ", dtype, self.weight.dtype)

          nn.init.constant_(self.weight, 1)
          nn.init.constant_(self.bias, 0)

        else:
          self.register_parameter('weight', None)
          self.register_parameter('bias', None)

        if self.track_running_stats:
          self.register_buffer('running_mean', torch.zeros(C))
          self.register_buffer('running_var', torch.ones(C))
          self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        #else:
        #  self.register_parameter('running_mean', None)
        #  self.register_parameter('running_var', None)
        #  self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
          self.running_mean.zero_()
          self.running_var.fill_(1)
          self.num_batches_tracked.zero_()

    def reset_parameters(self):
      self.reset_running_stats()
      if self.affine:
        self.weight.data.fill_(1.)
        self.bias.data.zero_()

    def forward(self, input, input_add=None):
        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        if input_add == None:
          input_add = torch.Tensor()

        #input = input.contiguous()
        #if self.eltwise:
        #  input_add = input_add.contiguous()

        if N != self.N:
          self.xsmm_handle = XsmmBNHandleTPP(N, self.C, self.H, self.W, self.eps, self.relu, self.eltwise, self.training, self.dtype, self.bc)
          #self.xsmm_handle = XsmmBNHandle(N, self.C, self.H, self.W, input.dtype, self.relu, self.eltwise, train=True)
          self.N = N

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.Cblock, None, None],
        )
        if self.eltwise:
            blocked_input_add = self.get_blocked_tensor(
                input_add,
                self.blocked_input_signature,
                [None, self.Cblock, None, None],
            )
        else:
            blocked_input_add = input_add

        output_size = [self.N, self.C // self.Cblock, self.H, self.W, self.Cblock]
        #print("debug: calling bn with N C H W relu eltwise ", self.N, self.C, self.H, self.W, self.relu, self.eltwise)
        if not self.training and self.track_running_stats: # using during evaluation the running_mean and running_var computed during training beforehand
            output = XsmmBNTPP.apply(blocked_input, blocked_input_add, self.weight, self.bias, self.running_mean, self.running_var, self.invstd, self.xsmm_handle, output_size, self.training)
        else:
            output = XsmmBNTPP.apply(blocked_input, blocked_input_add, self.weight, self.bias, self.mean, self.var, self.invstd, self.xsmm_handle, output_size, self.training)
            #output = XsmmBNTPP.apply(input, input_add, self.weight, self.bias, self.mean, self.var, self.invstd, self.xsmm_handle, output_size, self.training)
        #output = XsmmBN.apply(input, input_add, self.weight, self.bias, self.mean, self.var, self.invstd, self.xsmm_handle, output_size, self.training)

        """
        print("momentum = ", self.momentum)

        for i in range(10):
            print("debug: i fwd running mean pre update     = ", i, self.running_mean.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd running var  pre update     = ", i, self.running_var.view(-1)[i].item())

        for i in range(10):
            print("debug: i fwd mean at update              = ", i, self.mean.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd var  at update              = ", i, self.var.view(-1)[i].item())
        """
        if self.training and self.track_running_stats:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.var
            """
            for i in range(10):
                print("debug: i fwd running mean post update    = ", i, self.running_mean.view(-1)[i].item())
            for i in range(10):
                print("debug: i bwd running var  post update    = ", i, self.running_var.view(-1)[i].item())
            """
            # Unused?
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1

        blocked_output = blocked_layout.BlockedTensor(output, self.blocked_output_signature)

        return blocked_output


class XsmmAvgPoolHandle:
    def __init__(self, N, C, H, W, R, S, padding, stride, dtype):
        self.handle = pcl_cgbp_cpp.avg_pooling_create_handle(N, C, H, W, R, S, padding, stride, 1 if dtype == torch.float32 else 2)
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.R = R
        self.S = S
        self.padding = padding
        self.stride = stride

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.avg_pooling_destroy_handle(self.handle)
            self.handle = None

class XsmmAvgPool(Function):
    @staticmethod
    def forward(ctx, input, handle, output_size, grad_in_size):
        input = input.contiguous()
        output = pcl_cgbp_cpp.avg_pooling_forward(handle.handle, input, output_size)

        ctx.xsmm_handle = handle
        ctx.grad_in_size = grad_in_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.xsmm_handle
        del ctx.xsmm_handle
        grad_in_size = ctx.grad_in_size

        grad_output = grad_output.contiguous()

        grad_input = pcl_cgbp_cpp.avg_pooling_backward(handle.handle, grad_output, grad_in_size)
        return (grad_input, None, None, None, None)

class XsmmAvgPool2d(nn.Module):
    r"""PCL AvgPool2d module for using libxsmm Pooling"""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
        super(XsmmAvgPool2d, self).__init__()
        self.N = 0
        self.C = 1
        self.H = 1
        self.W = 1
        self.xsmm_handle = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input):
        N = input.size(0)
        if input.dim() == 4:
          self.C = input.size(1)
        elif input.dim() == 5:
          self.C = input.size(1) * input.size(4)
        self.H = input.size(2)
        self.W = input.size(3)

        input = input.contiguous()

        if N != self.N:
            print("AvgPool Create handle: ", N, self.C, self.H, self.W, self.kernel_size, self.kernel_size, self.padding, self.stride, input.dtype)
            self.xsmm_handle = XsmmAvgPoolHandle(N, self.C, self.H, self.W, self.kernel_size, self.kernel_size, self.padding, self.stride, input.dtype)
            self.N = N

        output_size = pcl_cgbp_cpp.get_pooling_tensor_layout(self.xsmm_handle.handle, "output");
        grad_in_size = pcl_cgbp_cpp.get_pooling_tensor_layout(self.xsmm_handle.handle, "grad_input");
        output =  XsmmAvgPool.apply(input, self.xsmm_handle, output_size, grad_in_size)
        return output

class XsmmMaxPoolHandle:
    def __init__(self, N, C, H, W, R, S, padding, stride, dtype):
        self.handle = pcl_cgbp_cpp.max_pooling_create_handle(N, C, H, W, R, S, padding, stride, 1 if dtype == torch.float32 else 2)
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.R = R
        self.S = S
        self.padding = padding
        self.stride = stride

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.max_pooling_destroy_handle(self.handle)
            self.handle = None

class XsmmMaxPool(Function):
    @staticmethod
    def forward(ctx, input, handle, output_size, grad_in_size):
        input = input.contiguous()

        output, mask = pcl_cgbp_cpp.max_pooling_forward(handle.handle, input, output_size)

        ctx.xsmm_handle = handle
        ctx.grad_in_size = grad_in_size
        ctx.save_for_backward(mask)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.xsmm_handle
        del ctx.xsmm_handle

        mask, = ctx.saved_tensors
        grad_in_size = ctx.grad_in_size

        grad_output = grad_output.contiguous()
        grad_input = pcl_cgbp_cpp.max_pooling_backward(handle.handle, grad_output, mask, grad_in_size)

        return (grad_input, None, None, None, None)

class XsmmMaxPool2d(nn.Module):
    r"""PCL Conv2d module for using libxsmm Conv"""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(XsmmMaxPool2d, self).__init__()
        self.N = 0
        self.C = 1
        self.H = 1
        self.W = 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.xsmm_handle = None

    def forward(self, input):
        N = input.size(0)
        if input.dim() == 4:
          self.C = input.size(1)
        elif input.dim() == 5:
          self.C = input.size(1) * input.size(4)
        self.H = input.size(2)
        self.W = input.size(3)

        input = input.contiguous()

        if N != self.N:
            print("MaxPool Create handle: ", N, self.C, self.H, self.W, self.kernel_size, self.kernel_size, self.padding, self.stride, input.dtype)
            self.xsmm_handle = XsmmMaxPoolHandle(N, self.C, self.H, self.W, self.kernel_size, self.kernel_size, self.padding, self.stride, input.dtype)
            self.N = N

        output_size = pcl_cgbp_cpp.get_pooling_tensor_layout(self.xsmm_handle.handle, "output");
        grad_in_size = pcl_cgbp_cpp.get_pooling_tensor_layout(self.xsmm_handle.handle, "grad_input");
        output =  XsmmMaxPool.apply(input, self.xsmm_handle, output_size, grad_in_size)
        return output


class XsmmFcHandleTPP:
    def __init__(self, N, C, K, bn, bc, bk, eltwise, dtype):
        self.N  = N
        self.C  = C
        self.K  = K
        self.bn = bn
        self.bc = bc
        self.bk = bk

        self.fuse_type = -1
        if eltwise == True:
            self.fuse_type = 1 # eltwise
        else:
            self.fuse_type = 0 # no fusion

        if self.fuse_type == -1:
            print("unsupported fuse_type is requested")
            exit()

        self.handle = pcl_cgbp_cpp.fc_setup_new(N, C, K, self.fuse_type, 0 if dtype == torch.float32 else 1, self.bn if self.bn is not None else -1 , self.bc if self.bc is not None else -1, self.bk if self.bk is not None else -1)

    def __del__(self):
        if self.handle:
            pcl_cgbp_cpp.fc_setup_destroy_new(self.handle)
            self.handle = None

class XsmmFcTPP(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, handle, output_size):
        input  = input.contiguous()
        weight = weight.contiguous()
        bias   = bias.contiguous()

        #print("dbg linear: norm-2 of input, weighted = ",  input.norm(2), input.norm(2)/input.numel())
        #print("dbg linear: norm-2 of weight, weighted = ",  weight.norm(2), weight.norm(2)/weight.numel())
        #print("dbg linear: norm-2 of bias, weighted = ",  bias.norm(2), bias.norm(2)/bias.numel())

        output, relumask = pcl_cgbp_cpp.fc_forward_new(handle.handle, input, weight, bias, output_size)

        ctx.xsmm_handle = handle
        ctx.save_for_backward(input, weight, bias, relumask)

        #print("dbg linear: norm-2 of output, weighted = ",  output.norm(2), output.norm(2)/output.numel())

        return output

    @staticmethod
    def backward(ctx, grad_output):

        handle = ctx.xsmm_handle
        del ctx.xsmm_handle

        input, weight, bias, relumask = ctx.saved_tensors

        grad_output = grad_output.contiguous()

        bias_size = bias.size()
        """
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if rank == 0:
            grad_output_nan_count = torch.isnan(grad_output.view(-1)).sum()
            print("nan check in new fc bwd for grad_output, nancount = ", grad_output_nan_count)
            input_nan_count = torch.isnan(input.view(-1)).sum()
            print("nan check in new fc bwd for input, nancount = ", input_nan_count)
            weight_nan_count = torch.isnan(weight.view(-1)).sum()
            print("nan check in new fc bwd for weight, nancount = ", weight_nan_count)

            if grad_output_nan_count > 0 or input_nan_count > 0 or weight_nan_count > 0:
                print("Exiting because nan count is not zero in new fc bwd input tensors (gradout, input, weight) ")
                exit(-1)
        """
        grad_input, grad_weight, grad_bias = pcl_cgbp_cpp.fc_backward_new(handle.handle, grad_output, input, weight, relumask, bias_size)
        """
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if rank == 0:
            grad_input_nan_count = torch.isnan(grad_input.view(-1)).sum()
            print("nan check in new fc bwd for grad_input, nancount = ", grad_input_nan_count)
            grad_weight_nan_count = torch.isnan(grad_weight.view(-1)).sum()
            print("nan check in new fc bwd for grad_weight, nancount = ", grad_weight_nan_count)
            grad_bias_nan_count = torch.isnan(grad_bias.view(-1)).sum()
            print("nan check in new fc bwd for grad_bias, nancount = ", grad_bias_nan_count)

            if grad_input_nan_count > 0 or grad_weight_nan_count > 0 or grad_bias_nan_count > 0:
                if grad_input_nan_count > 0:
                    for i in range(grad_input.numel()):
                        print("i grad_input = ", i, grad_input.view(-1)[i].item())

                print("Exiting because nan count is not zero in new fc bwd gradients ")
                exit(-1)
        """
        return (grad_input, grad_weight, grad_bias, None, None)


class XsmmLinearTPP(BlockedModule, nn_Linear):
    r"""PCL FC (Fully Connected) module for using libxsmm Fc TPP"""

    def __init__(self, in_channels, out_channels, bn=None, bc=None, bk=None, eltwise=True, dtype=torch.float):
        #super(XsmmLinearTPP, self).__init__()
        nn_Linear.__init__(self, in_channels, out_channels, bias=eltwise, device=None, dtype=dtype)
        self.N            = 0
        self.C            = in_channels
        self.K            = out_channels
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.eltwise      = eltwise
        self.dtype        = dtype
        self.use_bf16     = True if self.dtype == torch.bfloat16 else False
        self.bn           = bn
        self.bc           = bc
        self.bk           = bk

        self.low_prec_vnni_blocking = pcl_cgbp_cpp.get_vnni_blocking(dtype)

        #print("dbg: in XsmmLinearTPP constructor self.low_prec_vnni_blocking = ", self.low_prec_vnni_blocking)

        if self.use_bf16 and self.in_channels%self.low_prec_vnni_blocking != 0:
            print("Error: in XsmmLinearTPP constructor in_channels is not divisible by the vnni blocking = ", self.low_prec_vnni_blocking)
            exit()

        if self.bc is None or self.bk is None:
            [self.Cblock, self.Kblock] = pcl_cgbp_cpp.fc_get_feature_map_blocks(self.C, self.K)
        else:
            self.Cblock = bc
            self.Kblock = bk

        self.lp_block = 1 if self.dtype == torch.float else self.low_prec_vnni_blocking

        self.weight = BlockedParameter(self.weight.data)
        if self.use_bf16:
            self.weight.set_blocking_param(
                (
                    [self.Kblock, [self.Cblock // self.lp_block, self.lp_block]],
                    [0, 2, 3, 1, 4],
                )
            )
            # K 0 1 | C 2 3 4 -> KCckxc, lp_block = xc
        else:
            self.weight.set_blocking_param(
                (
                    [self.Kblock, self.Cblock],
                    [0, 2, 3, 1],
                )
            )


        self.blocked_input_signature = blocked_layout.get_blocking_signature(
            "NC", "NCNC"
        )
        self.blocked_output_signature = blocked_layout.get_blocking_signature(
            "NC", "NCNC"
        )

        if self.eltwise:
          self.bias = Parameter(torch.empty(self.K))

    def maybe_block_params(self):
        self.weight.block()
        #self.bias.block()

    def forward(self, input):
        #print('Fc Input {}'.format(input.shape))

        self.maybe_block_params()

        N = input.size(0) # after blocked_layout started to take care of blocking transparently
        #N = input.size(0) * input.size(2)

        #input  = input.to(global_dtype).contiguous()
        #print("Fixme!!! Does TPP FC support bf16 properly? Likely not yet, dtype = ", self.dtype)
        weight = self.weight.to(self.dtype)
        if self.eltwise:
            bias   = self.bias.to(self.dtype)
        else:
            bias = torch.Tensor().to(self.dtype)

        if N != self.N:
          if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
              print("Fc Create handle: ", N, self.C, self.K, self.dtype)
          self.N = N
          if self.bn is None:
              self.Nblock = pcl_cgbp_cpp.fc_get_n_block(self.N)
          else:
              self.Nblock = self.bn
              if (self.N - (self.N // self.bn) * self.bn != 0):
                  print("Error: input bn is not compatible with N in XsmmLinearTPP constructor")
                  exit()
          #self.Nblock = pcl_cgbp_cpp.fc_get_n_block(self.N)
          #print("dbg: Nblock, Cblock, Kblock = ", self.Nblock, self.Cblock, self.Kblock)
          #print("dbg: eltwise = ", self.eltwise)
          self.xsmm_handle = XsmmFcHandleTPP(N, self.C, self.K, self.Nblock, self.Cblock, self.Kblock, self.eltwise, self.dtype)

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [self.Nblock, self.Cblock],
        )

        #print("N, C, K, Nblock, Cblock, Kblock = ", self.N, self.C, self.K, self.Nblock, self.Cblock, self.Kblock)
        output_size  = [self.N//self.Nblock, self.K//self.Kblock, self.Nblock, self.Kblock]
        output =  XsmmFcTPP.apply(blocked_input, weight, bias, self.xsmm_handle, output_size)

        blocked_output = blocked_layout.BlockedTensor(output, self.blocked_output_signature)

        return blocked_output

class XsmmPoolTPPHandle:
    def __init__(self, N, C, H, W, R, S, padding, stride, pooltype, dtype, bc):
        self.pooltype = 0 if pooltype == "max" else 1
        #if self.pooltype == "max":
        #    self.handle = pcl_cgbp_cpp.max_pooling_create_handle(N, C, H, W, R, S, padding, stride, 1 if dtype == torch.float32 else 2)
        #else: #if self.pooltype == "avg":
        #    self.handle = pcl_cgbp_cpp.avg_pooling_create_handle(N, C, H, W, R, S, padding, stride, 1 if dtype == torch.float32 else 2)
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        self.R = R
        self.S = S
        self.padding = padding
        self.stride = stride
        self.bc = bc

        self.handle = pcl_cgbp_cpp.pooling_setup_new(N, C, H, W, R, S, padding, stride, self.pooltype, 0 if dtype == torch.float32 else 1, self.bc if self.bc is not None else -1)


    def __del__(self):
        if self.handle:
            #if self.pooltype == "max":
            #    pcl_cgbp_cpp.max_pooling_destroy_handle(self.handle)
            #else: #if self.pooltype == "avg":
            #    pcl_cgbp_cpp.avg_pooling_destroy_handle(self.handle)
            #if self.handle.fwd_cfg or self.handle.bwd_cfg:
            pcl_cgbp_cpp.pooling_setup_destroy_new(self.handle)
            self.handle = None

class XsmmPoolTPP(Function):
    @staticmethod
    def forward(ctx, input, handle, output_size, grad_in_size):
        input = input.contiguous()

        if handle.pooltype == 0: # max
            output, mask = pcl_cgbp_cpp.max_pooling_forward_new(handle.handle, input, output_size)
        else: #if handle.pooltype == 1 ~ "avg"
            output, mask = pcl_cgbp_cpp.avg_pooling_forward_new(handle.handle, input, output_size)
        ctx.save_for_backward(mask)

        ctx.xsmm_handle  = handle
        ctx.grad_in_size = grad_in_size
        return output

    @staticmethod
    def backward(ctx, grad_output):
        handle = ctx.xsmm_handle
        del ctx.xsmm_handle
        grad_in_size = ctx.grad_in_size

        grad_output = grad_output.contiguous()

        mask, = ctx.saved_tensors
        if handle.pooltype == 0: #"max":
            grad_input = pcl_cgbp_cpp.max_pooling_backward_new(handle.handle, grad_output, mask, grad_in_size)
        else: #if self.pooltype == 1 ~ "avg":
            grad_input = pcl_cgbp_cpp.avg_pooling_backward_new(handle.handle, grad_output, mask, grad_in_size)

        return (grad_input, None, None, None, None)

class XsmmAvgPoolTPP2d(BlockedModule, nn_AvgPool2d):
    r"""PCL AvgPoolTPP2d module for using libxsmm TPP Pooling"""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None, dtype=torch.float, bc=None):
        #super(XsmmAvgPoolTPP2d, self).__init__()
        nn_AvgPool2d.__init__(self, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        self.N = 0
        self.C = None
        self.H = None
        self.W = None
        self.xsmm_handle = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.dtype = dtype
        self.bc    = bc

        self.blocked_input_signature = blocked_layout.get_blocking_signature(
            "NCHW", "NCHWC"
        )
        self.blocked_output_signature = self.blocked_input_signature

    def forward(self, input):
        N = input.size(0)
        if input.dim() == 4:
          self.C = input.size(1)
        elif input.dim() == 5:
          self.C = input.size(1) * input.size(4)
        self.H = input.size(2)
        self.W = input.size(3)

        #input = input.contiguous()

        if N != self.N:
            print("AvgPool Create handle: ", N, self.C, self.H, self.W, self.kernel_size, self.kernel_size, self.padding, self.stride, self.dtype, self.bc)
            self.xsmm_handle = XsmmPoolTPPHandle(N, self.C, self.H, self.W, self.kernel_size, self.kernel_size, self.padding, self.stride, "avg", self.dtype, self.bc)
            self.N = N
            if self.bc is not None:
                self.Cblock = self.bc
            else:
                self.Cblock = pcl_cgbp_cpp.pooling_get_c_block(self.C)

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.Cblock, None, None],
        )

        inH = input.size(2)
        inW = input.size(3)
        outH = (inH + 2 * self.padding - self.kernel_size)//self.stride + 1
        outW = (inW + 2 * self.padding - self.kernel_size)//self.stride + 1

        output_size = [self.N, self.C//self.Cblock, outH, outW, self.Cblock]

        grad_in_size = [self.N, self.C//self.Cblock, inH, inW, self.Cblock]

        output = XsmmPoolTPP.apply(blocked_input, self.xsmm_handle, output_size, grad_in_size)

        blocked_output = blocked_layout.BlockedTensor(output, self.blocked_output_signature)

        return blocked_output

class XsmmMaxPoolTPP2d(BlockedModule, nn_MaxPool2d):
    r"""PCL MaxPoolTPP2d module for using libxsmm TPP Pooling"""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False, dtype=torch.float, bc=None):
        #super(XsmmMaxPoolTPP2d, self).__init__()
        nn_MaxPool2d.__init__(self, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.N = 0
        self.C = None
        self.H = None
        self.W = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.xsmm_handle = None
        self.dtype = dtype
        self.bc    = bc

        self.blocked_input_signature = blocked_layout.get_blocking_signature(
            "NCHW", "NCHWC"
        )
        self.blocked_output_signature = self.blocked_input_signature

    def forward(self, input):

        N = input.size(0)
        if input.dim() == 4:
          self.C = input.size(1)
        elif input.dim() == 5:
          self.C = input.size(1) * input.size(4)
        self.H = input.size(2)
        self.W = input.size(3)

        #input = input.contiguous()

        if N != self.N:
            print("MaxPool Create handle: ", N, self.C, self.H, self.W, self.kernel_size, self.kernel_size, self.padding, self.stride, self.dtype, self.bc)
            self.xsmm_handle = XsmmPoolTPPHandle(N, self.C, self.H, self.W, self.kernel_size, self.kernel_size, self.padding, self.stride, "max", self.dtype, self.bc)
            self.N = N
            if self.bc is not None:
                self.Cblock = self.bc
            else:
                self.Cblock = pcl_cgbp_cpp.pooling_get_c_block(self.C)

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.Cblock, None, None],
        )

        inH = input.size(2)
        inW = input.size(3)
        outH = (inH + 2 * self.padding - self.kernel_size)//self.stride + 1
        outW = (inW + 2 * self.padding - self.kernel_size)//self.stride + 1

        output_size = [self.N, self.C//self.Cblock, outH, outW, self.Cblock]

        grad_in_size = [self.N, self.C//self.Cblock, inH, inW, self.Cblock]

        output = XsmmPoolTPP.apply(blocked_input, self.xsmm_handle, output_size, grad_in_size)

        blocked_output = blocked_layout.BlockedTensor(output, self.blocked_output_signature)

        return blocked_output


# Generic base class for batchnorm/groupnorm bottleneck (with a control flag for the norm in the constructor)
class Bottleneck_base(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, eps, stride=1, downsample1=None, downsample2=None, use_ref_conv=False, use_ref_norm=False, use_groupnorm=False, dtype=torch.float,
                 bc_conv1=None, bc_conv2=None, bc_conv3=None, bk_conv3=None):
        print("dbg: Bottleneck_base constructor called with inplanes, planes, eps, stride, downsample1, downsample2, use_ref_conv, use_ref_norm use_groupnorm dtype = ",
                  inplanes, planes, eps, stride, downsample1, downsample2, use_ref_conv, use_ref_norm, use_groupnorm, dtype)
        super(Bottleneck_base, self).__init__()

        self.use_ref_conv  = use_ref_conv
        self.use_ref_norm  = use_ref_norm
        self.use_groupnorm = use_groupnorm
        self.dtype         = dtype

        self.bc_conv1 = bc_conv1
        self.bc_conv2 = bc_conv2
        self.bc_conv3 = bc_conv3
        self.bk_conv3 = bk_conv3

        # eltwise is accounted for in the forward()
        # but relu is created here for the PyTorch reference impl
        if self.use_ref_norm == True:
            self.relu = nn.ReLU(inplace=False)

        if self.use_ref_conv != True:
            #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dtype=self.dtype)
            self.conv1 = XsmmConv2dTPP(inplanes, planes, kernel_size=1, bias=False, dtype=self.dtype, bc = self.bc_conv1, bk = self.bc_conv2)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dtype=self.dtype)

        if self.use_groupnorm:
            if self.use_ref_norm != True:
                self.bn1 = XsmmGroupNormTPP(32, planes, eps, relu=True, dtype=self.dtype)
            else:
                self.bn1 = nn.GroupNorm(32, planes, eps)
        else:
            if self.use_ref_norm != True:
                #self.bn1 = nn.BatchNorm2d(planes, eps, relu=True, dtype=self.dtype)
                self.bn1 = XsmmBatchNormTPP(planes, eps, relu=True, dtype=self.dtype, bc=self.bc_conv2)
                #self.bn1  = nn.BatchNorm2d(planes, eps)
            else:
                self.bn1 = nn.BatchNorm2d(planes, eps)
                #self.bn1  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        if self.use_ref_conv != True:
            self.conv2 = XsmmConv2dTPP(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, dtype=self.dtype, bc = self.bc_conv2, bk = self.bc_conv3)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False, dtype=self.dtype)

        if self.use_groupnorm:
            if self.use_ref_norm != True:
                self.bn2 = XsmmGroupNormTPP(32, planes, eps, relu=True, dtype=self.dtype)
            else:
                self.bn2 = nn.GroupNorm(32, planes, eps, dtype=self.dtype)
        else:
            if self.use_ref_norm != True:
                #self.bn2 = nn.BatchNorm2d(planes, eps, relu=True, dtype=self.dtype)
                self.bn2 = XsmmBatchNormTPP(planes, eps, relu=True, dtype=self.dtype, bc=self.bc_conv3)
                #self.bn2 = nn.BatchNorm2d(planes, eps)
            else:
                self.bn2 = nn.BatchNorm2d(planes, eps, dtype=self.dtype)
                #self.bn2  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        if self.use_ref_conv != True:
            #self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dtype=self.dtype)
            self.conv3 = XsmmConv2dTPP(planes, planes * 4, kernel_size=1, bias=False, dtype=self.dtype, bc = self.bc_conv3, bk = self.bk_conv3)
        else:
            self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dtype=self.dtype)

        if self.use_groupnorm:
            if self.use_ref_norm != True:
                self.bn3 = XsmmGroupNormTPP(32, planes * 4, eps, relu=True, eltwise=True, dtype=self.dtype)
            else:
                self.bn3 = nn.GroupNorm(32, planes * 4, eps, dtype=self.dtype)
        else:
            if self.use_ref_norm != True:
                #self.bn3 = nn.BatchNorm2d(planes * 4, eps, relu=True, eltwise=True, dtype=self.dtype)
                self.bn3 = XsmmBatchNormTPP(planes * 4, eps, relu=True, eltwise=True, dtype=self.dtype, bc=self.bk_conv3)
                #self.bn3  = nn.BatchNorm2d(planes * 4, eps)
            else:
                self.bn3  = nn.BatchNorm2d(planes * 4, eps, dtype=self.dtype)
                #self.bn3  = nn.BatchNorm2d(planes * 4, eps, track_running_stats=False)

        self.stride = stride

        #self.downsample1 = downsample1 # this is the conv part of downsampling;
        #self.downsample2 = downsample2 # this is the bn   part of downsampling;

        # Otherwise loading a reference model's state_dict does not work
        if downsample1 != None and downsample2 != None:
            self.add_module("downsample1", downsample1)
            self.add_module("downsample2", downsample2)
            if hasattr(downsample1, "bc") and hasattr(downsample1, "bk") and (downsample1.bc != self.bc_conv1 or downsample1.bk != self.bk_conv3):
                print("Error: downsample1 block sizes mismatch the bottleneck block sizes! Sizes are (downsample1) (btlnk)", downsample1.bc, downsample1.bk, self.bc_conv1, self.bk_conv3)
                exit()
            if hasattr(downsample2, "bc") and downsample2.bc != self.bk_conv3:
                print("Error: downsample2 block size mismatch the bottleneck block sizes!")
                exit()
        else:
            self.downsample1 = None
            self.downsample2 = None

        if self.use_groupnorm:
            def gn_init(m, zero_init=False):
                #assert isinstance(m, nn.GroupNorm) or isinstance(m, pcl_cgbp.nn_GroupNorm)
                m.weight.data.fill_(0. if zero_init else 1.)
                m.bias.data.zero_()
            gn_init(self.bn1)
            gn_init(self.bn2)
            gn_init(self.bn3, zero_init=True)

        #print("submodules of the base")
        #for m in self.modules():
        #    print(m)

    def forward(self, x):

        dump = False

        """

        #global global_tensor_x_counter
        #global global_block_forward_counter

        global_tensor_x_counter = 0

        self.rank = int(os.environ.get("PMI_RANK", -1))
        if self.rank < 0:
            self.rank = 0
        if self.training:
            self.dump_file_suffix    = '_train_sfx' + '_rank_' + str(self.rank)
            #self.dump_file_suffix    = '_train_tst' + '_rank_' + str(self.rank)
        else:
            self.dump_file_suffix    = '_eval_sfx' + '_rank_' + str(self.rank)
            #self.dump_file_suffix    = '_eval_tst' + '_rank_' + str(self.rank)
        """

        residual = x

        
        if dump:
            if type(self.conv1.weight) is BlockedParameter:
                self.conv1.weight.unblock()
            np.savetxt('my_layer_conv1_forward_weight' + str(global_tensor_x_counter)  + self.dump_file_suffix + '.txt', self.conv1.weight.contiguous().view(-1).detach().to(torch.float).numpy())
        
        
        if dump:
            tmp_tensor = x.unblocked_tensor() if type(x) is BlockedTensor else x
            #tmp_tensor = x.blocked_tensor() if type(x) is BlockedTensor else x
            np.savetxt('my_layer_conv1_forward_input_x_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1
        
        #print("dbg: calling conv1")
        out = self.conv1(x)
        
        if dump:
            tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
            #tmp_tensor = out.blocked_tensor() if type(out) is BlockedTensor else out
            np.savetxt('my_layer_conv1_forward_output_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1
        

        if self.use_ref_norm == True:
            out = self.bn1(out)
            out = self.relu(out)
        else:
            out = self.bn1(out)


        if dump:
            tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
            #tmp_tensor = out.blocked_tensor() if type(out) is BlockedTensor else out
            np.savetxt('my_layer_conv2_forward_input_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

        #print("dbg: calling conv2")
        out = self.conv2(out)

        #tmp_res = out.unblocked_tensor()
        #print("tmp_res.shape = ", tmp_res.shape)
        #return tmp_res

        
        if dump:
            tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
            #tmp_tensor = out.blocked_tensor() if type(out) is BlockedTensor else out
            np.savetxt('my_layer_conv2_forward_output_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1
        

        if self.use_ref_norm == True:
            out = self.bn2(out)
            out = self.relu(out)
        else:
            out = self.bn2(out)
        #out = self.bn2(out)

        if dump:
            tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
            #tmp_tensor = out.blocked_tensor() if type(out) is BlockedTensor else out
            np.savetxt('my_layer_conv3_forward_input_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

        out = self.conv3(out)

        #tmp_res = out.unblocked_tensor()
        #print("tmp_res.shape = ", tmp_res.shape)
        #return tmp_res

        if dump:
            tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
            #tmp_tensor = out.blocked_tensor() if type(out) is BlockedTensor else out
            np.savetxt('my_layer_conv3_forward_output_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

        if self.downsample1 is not None and self.downsample2 is not None:
            residual1 = self.downsample1(x)

            #tmp_res = residual1.unblocked_tensor()
            #print("tmp_res.shape = ", tmp_res.shape)
            #return tmp_res

            residual  = self.downsample2(residual1)

            #tmp_res = residual.unblocked_tensor()
            #print("tmp_res.shape = ", tmp_res.shape)
            #return tmp_res

        if self.use_ref_norm == True:
            out = self.bn3(out)
            out += residual
            out = self.relu(out)
        else:
            out = self.bn3(out, residual)

        if dump:
            tmp_tensor = residual.unblocked_tensor() if type(residual) is BlockedTensor else residual
            #tmp_tensor = residual.blocked_tensor() if type(residual) is BlockedTensor else residual
            np.savetxt('my_layer_residual_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

        if dump:
            tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
            #tmp_tensor = out.blocked_tensor() if type(out) is BlockedTensor else out
            #np.savetxt('my_layer_block_forward_output_out_' + str(global_block_forward_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            #global_block_forward_counter = global_block_forward_counter + 1
            np.savetxt('my_layer_block_forward_output_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1
            #exit()

        #return out
        if type(out) is BlockedTensor:
            return out.unblocked_tensor()
        else:
            return out


class BottleneckBNHandleTPP:
    def __init__(self, N, inplanes, H, W, planes, stride, bn_eps, bn_momentum, bn_track_running_stats, expansion, physical_3x3_padding, dtype, bc_conv1, bc_conv2, bc_conv3, bk_conv3):
        self.N         = N
        self.inplanes  = inplanes
        self.H         = H
        self.W         = W
        self.planes    = planes
        self.stride    = stride
        self.expansion = expansion
        self.physical_3x3_padding = physical_3x3_padding
        self.bc_conv1  = bc_conv1
        self.bc_conv2  = bc_conv2
        self.bc_conv3  = bc_conv3
        self.bk_conv3  = bk_conv3

        self.bn_eps                 = bn_eps
        self.bn_momentum            = bn_momentum
        self.bn_track_running_stats = bn_track_running_stats

        self.dtype    = dtype

        # Could be (if bugged) inconsistent with the defnition used in BottleneckTPP module
        if self.stride != 1 or self.inplanes != self.planes * self.expansion:
            self.has_residual_conv = True
        else:
            self.has_residual_conv = False

        #print("dbg: in bottleneck bn handle tpp create")
        self.handle = pcl_cgbp_cpp.bottleneck_bn_setup_new(N, inplanes, H, W, planes, stride, bn_eps, bn_momentum, bn_track_running_stats, expansion,
                                                           1 if physical_3x3_padding else 0, 0 if dtype == torch.float else 1,
                                                           self.bc_conv1 if self.bc_conv1 is not None else -1,
                                                           self.bc_conv2 if self.bc_conv2 is not None else -1,
                                                           self.bc_conv3 if self.bc_conv3 is not None else -1,
                                                           self.bk_conv3 if self.bk_conv3 is not None else -1)
        #print("dbg: bottleneck_bn_setup_new called")

    def __del__(self):
        if hasattr(self, "handle"):
            if self.handle:
                pcl_cgbp_cpp.bottleneck_bn_setup_destroy_new(self.handle)
            self.handle = None
        else:
            print("Error: destructor should not be called before handle is defined")
            exit()

class BottleneckApplyBNTPP(Function):
    @staticmethod
    def forward(ctx, handle, training, *inputs):

        #print("dbg: in bottleneck bn apply tpp forward")

        bn_norm_type = 0 if training else 1

        #input_tensors = []
        #for entity in inputs:
        #    print("type of entity = ", type(entity))
        #    input_tensors.append(entity.data) # not necessary

        #tmp_list = list(inputs)
        #tmp_list[0] = torch.ones_like(tmp_list[0])
        #tmp_list[1] = torch.ones_like(tmp_list[1])
        #inputs = tuple(tmp_list)

        output, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out = pcl_cgbp_cpp.bottleneck_bn_forward_new(handle.handle, bn_norm_type, inputs) #_tensors)
        #print("dbg: bottleneck_forward_new called")

        if handle.has_residual_conv == False:
            dummy_tensor = torch.empty(1)
            bn4_relu_out = dummy_tensor
            conv4_out    = dummy_tensor

        (input,
         c1w, c2w, c3w, c4w,
         b1w, b2w, b3w, b4w,
         b1b, b2b, b3b, b4b,
         b1m, b2m, b3m, b4m,
         b1n, b2n, b3n, b4n) = inputs

        """
        global_tensor_x_counter = 0

        dump = False

        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if training:
            dump_file_suffix    = '_train_tst' + '_rank_' + str(rank)
        else:
            dump_file_suffix    = '_eval_tst' + '_rank_' + str(rank)

        if dump:
            if type(c1w) is BlockedParameter:
                c1w.unblock()
            #else:
            tmp_tensor = c1w
            #tmp_tensor = c1w.unblock() if type(c1w) is BlockedParameter else c1w
            np.savetxt('my_layer_conv1_forward_weight_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1
            
            tmp_tensor = input.unblocked_tensor() if type(input) is BlockedTensor else input
            np.savetxt('my_layer_conv1_forward_input_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1


            print("types are ", type(conv1_out), type(bn1_out), type(conv2_out), type(bn2_out), type(conv3_out), type(bn4_out), type(output))

            tmp_tensor = conv1_out.unblocked_tensor() if type(conv1_out) is BlockedTensor else conv1_out
            np.savetxt('my_layer_conv1_forward_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = bn1_out.unblocked_tensor() if type(bn1_out) is BlockedTensor else bn1_out
            np.savetxt('my_layer_conv2_forward_input_out_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = conv2_out.unblocked_tensor() if type(conv2_out) is BlockedTensor else conv2_out
            np.savetxt('my_layer_conv2_forward_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = bn2_out.unblocked_tensor() if type(bn2_out) is BlockedTensor else bn2_out
            np.savetxt('my_layer_conv3_forward_input_out_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = conv3_out.unblocked_tensor() if type(conv3_out) is BlockedTensor else conv3_out
            np.savetxt('my_layer_conv3_forward_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = bn4_out.unblocked_tensor() if type(bn4_out) is BlockedTensor else bn4_out
            np.savetxt('my_layer_residual_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = output.unblocked_tensor() if type(output) is BlockedTensor else output
            #np.savetxt('my_layer_block_forward_output_out_' + str(global_block_forward_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            #global_block_forward_counter = global_block_forward_counter + 1
            np.savetxt('my_layer_block_forward_output_out_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            #tmp_tensor = conv3_out.unblocked_tensor() if type(conv3_out) is BlockedTensor else conv3_out
            #np.savetxt('my_layer_conv3_forward_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            #global_tensor_x_counter = global_tensor_x_counter + 1
            

        """

        ctx.xsmm_handle = handle

        ctx.save_for_backward(input, c1w, c2w, c3w, c4w, b1w, b2w, b3w, b4w, b1b, b2b, b3b, b4b, b1m, b2m, b3m, b4m, b1n, b2n, b3n, b4n,
                              conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out,
                              bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out)

        return output

    @staticmethod
    def backward(ctx, *grad_outs):
        #print("dbg: in bottleneck apply backward")
        inputs = []
        inputs += [g.contiguous() for g in grad_outs]

        inputs += ctx.saved_tensors

        handle = ctx.xsmm_handle
        # FIXME: Not sure if necessary
        #del ctx.xsmm_handle

        #for entity in inputs:
        #    print("type of entity = ", type(entity))


        (grad_c1w, grad_c2w, grad_c3w, grad_c4w,
         grad_b1w, grad_b2w, grad_b3w, grad_b4w,
         grad_b1b, grad_b2b, grad_b3b, grad_b4b,
         grad_c1i, grad_c4i) = pcl_cgbp_cpp.bottleneck_bn_backward_new(handle.handle, inputs) #_tensors)

        #print("dbg: bottleneck_backward_new called")

        """
        for i in range(10):
            print("i grad_c1w grad_c2w grad_c3w ", i, grad_c1w.view(-1)[i].item(), grad_c2w.view(-1)[i].item(), grad_c3w.view(-1)[i].item())

        for i in range(10):
            print("i grad_b1w grad_b2w grad_b3w ", i, grad_b1w.view(-1)[i].item(), grad_b2w.view(-1)[i].item(), grad_b3w.view(-1)[i].item())

        for i in range(10):
            print("i grad_b1b grad_b2b grad_b3b ", i, grad_b1b.view(-1)[i].item(), grad_b2b.view(-1)[i].item(), grad_b3b.view(-1)[i].item())

        for i in range(10):
            print("i grad_c1i grad_c4i ", i, grad_c1i.view(-1)[i].item(), grad_c4i.view(-1)[i].item())
        """

        """

        global_tensor_x_counter = 0

        dump = False

        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        dump_file_suffix    = '_train_tst' + '_rank_' + str(rank)

        if dump:
            tmp_tensor = grad_c3w.unblocked_tensor() if type(grad_c3w) is BlockedTensor else grad_c3w
            np.savetxt('my_layer_conv3_bwd_w_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = grad_c3i.unblocked_tensor() if type(grad_c3i) is BlockedTensor else grad_c3i
            np.savetxt('my_layer_conv3_bwd_i_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = grad_b3i.unblocked_tensor() if type(grad_b3i) is BlockedTensor else grad_b3i
            np.savetxt('my_layer_bn3_bwd_i_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = grad_b3i_add.unblocked_tensor() if type(grad_b3i_add) is BlockedTensor else grad_b3i_add
            np.savetxt('my_layer_bn3_bwd_iadd_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = grad_b3w.unblocked_tensor() if type(grad_b3w) is BlockedTensor else grad_b3w
            np.savetxt('my_layer_bn3_bwd_w_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            tmp_tensor = grad_b3b.unblocked_tensor() if type(grad_b3b) is BlockedTensor else grad_b3b
            np.savetxt('my_layer_bn3_bwd_b_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1
        """

        """
        for i in range(10):
            print("i grad_c1w grad_c2w grad_c3w ", i, grad_c1w.view(-1)[i].item(), grad_c2w.view(-1)[i].item(), grad_c3w.view(-1)[i].item())

        for i in range(10):
            print("i grad_b1w grad_b2w grad_b3w ", i, grad_b1w.view(-1)[i].item(), grad_b2w.view(-1)[i].item(), grad_b3w.view(-1)[i].item())

        for i in range(10):
            print("i grad_b1b grad_b2b grad_b3b ", i, grad_b1b.view(-1)[i].item(), grad_b2b.view(-1)[i].item(), grad_b3b.view(-1)[i].item())

        for i in range(10):
            print("i grad_c1i grad_c4i ", i, grad_c1i.view(-1)[i].item(), grad_c4i.view(-1)[i].item())
        """

        grad_input = grad_c1i + grad_c4i

        """
        print("nan check in pt for grad_c1i, nan count = ", torch.isnan(grad_c1i.view(-1)).sum())
        print("nan check in pt for grad_c4i, nan count = ", torch.isnan(grad_c4i.view(-1)).sum())
        print("nan check in pt for grad_input, nan count = ", torch.isnan(grad_input.view(-1)).sum())
        print("nan check in pt for grad_c1w,   nan count = ", torch.isnan(grad_c1w.view(-1)).sum())
        print("nan check in pt for grad_c2w,   nan count = ", torch.isnan(grad_c2w.view(-1)).sum())
        print("nan check in pt for grad_c3w,   nan count = ", torch.isnan(grad_c3w.view(-1)).sum())
        if handle.has_residual_conv:
            print("nan check in pt for grad_c4w,   nan count = ", torch.isnan(grad_c4w.view(-1)).sum())
        """

        return (None, None, # for handle and training arguments in forward
                grad_input,
                grad_c1w, grad_c2w, grad_c3w, grad_c4w,
                grad_b1w, grad_b2w, grad_b3w, grad_b4w,
                grad_b1b, grad_b2b, grad_b3b, grad_b4b,
                None,     None,     None,     None, # for means
                None,     None,     None,     None) # for vars

class BottleneckGNHandleTPP:
    def __init__(self, N, inplanes, H, W, planes, G, stride, gn_eps, expansion, physical_3x3_padding, dtype):
        self.N         = N
        self.inplanes  = inplanes
        self.H         = H
        self.W         = W
        self.planes    = planes
        self.G         = G
        self.stride    = stride
        self.expansion = expansion
        self.physical_3x3_padding = physical_3x3_padding

        self.gn_eps                 = gn_eps

        self.dtype    = dtype

        # Could be (if bugged) inconsistent with the definition used in BottleneckTPP module
        if self.stride != 1 or self.inplanes != self.planes * self.expansion:
            self.has_residual_conv = True
        else:
            self.has_residual_conv = False

        #print("dbg: in bottleneck gn handle tpp create")
        self.handle = pcl_cgbp_cpp.bottleneck_gn_setup_new(N, inplanes, H, W, planes, G, stride, gn_eps, expansion,
                                                        1 if physical_3x3_padding else 0, 0 if dtype == torch.float else 1)
        #print("dbg: bottleneck_gn_setup_new called")

    def __del__(self):
        if hasattr(self, "handle"):
            if self.handle:
                pcl_cgbp_cpp.bottleneck_gn_setup_destroy_new(self.handle)
            self.handle = None
        else:
            print("Error: destructor should not be called before handle is defined")
            exit()

class BottleneckApplyGNTPP(Function):
    @staticmethod
    def forward(ctx, handle, training, *inputs):

        #print("dbg: in bottleneck apply tpp forward")

        #input_tensors = []
        #for entity in inputs:
        #    print("type of entity = ", type(entity))
        #    input_tensors.append(entity.data) # not necessary

        output, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out = pcl_cgbp_cpp.bottleneck_gn_forward_new(handle.handle, inputs) #_tensors)
        #print("dbg: bottleneck_forward_new called")

        if handle.has_residual_conv == False:
            dummy_tensor = torch.empty(1)
            bn4_relu_out = dummy_tensor
            conv4_out    = dummy_tensor

        (input,
         c1w, c2w, c3w, c4w,
         b1w, b2w, b3w, b4w,
         b1b, b2b, b3b, b4b,
         b1m, b2m, b3m, b4m,
         b1n, b2n, b3n, b4n) = inputs

        ctx.xsmm_handle = handle

        ctx.save_for_backward(input, c1w, c2w, c3w, c4w, b1w, b2w, b3w, b4w, b1b, b2b, b3b, b4b, b1m, b2m, b3m, b4m, b1n, b2n, b3n, b4n,
                              conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out,
                              bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out)

        return output

    @staticmethod
    def backward(ctx, *grad_outs):
        #print("dbg: in bottleneck gn apply backward")
        inputs = []
        inputs += [g.contiguous() for g in grad_outs]

        inputs += ctx.saved_tensors

        handle = ctx.xsmm_handle

        # FIXME: Not sure if necessary
        #del ctx.xsmm_handle

        (grad_c1w, grad_c2w, grad_c3w, grad_c4w,
         grad_b1w, grad_b2w, grad_b3w, grad_b4w,
         grad_b1b, grad_b2b, grad_b3b, grad_b4b,
         grad_c1i, grad_c4i) = pcl_cgbp_cpp.bottleneck_gn_backward_new(handle.handle, inputs) #_tensors)

        #print("dbg: bottleneck_backward_new called")

        """
        for i in range(10):
            print("i grad_c1w grad_c2w grad_c3w ", i, grad_c1w.view(-1)[i].item(), grad_c2w.view(-1)[i].item(), grad_c3w.view(-1)[i].item())

        for i in range(10):
            print("i grad_b1w grad_b2w grad_b3w ", i, grad_b1w.view(-1)[i].item(), grad_b2w.view(-1)[i].item(), grad_b3w.view(-1)[i].item())

        for i in range(10):
            print("i grad_b1b grad_b2b grad_b3b ", i, grad_b1b.view(-1)[i].item(), grad_b2b.view(-1)[i].item(), grad_b3b.view(-1)[i].item())

        for i in range(10):
            print("i grad_c1i grad_c4i ", i, grad_c1i.view(-1)[i].item(), grad_c4i.view(-1)[i].item())
        """

        grad_input = grad_c1i + grad_c4i

        """
        print("nan check in pt for grad_c1i, nan count = ", torch.isnan(grad_c1i.view(-1)).sum())
        print("nan check in pt for grad_c4i, nan count = ", torch.isnan(grad_c4i.view(-1)).sum())
        print("nan check in pt for grad_input, nan count = ", torch.isnan(grad_input.view(-1)).sum())
        print("nan check in pt for grad_c1w,   nan count = ", torch.isnan(grad_c1w.view(-1)).sum())
        print("nan check in pt for grad_c2w,   nan count = ", torch.isnan(grad_c2w.view(-1)).sum())
        print("nan check in pt for grad_c3w,   nan count = ", torch.isnan(grad_c3w.view(-1)).sum())
        if handle.has_residual_conv:
            print("nan check in pt for grad_c4w,   nan count = ", torch.isnan(grad_c4w.view(-1)).sum())
        """

        return (None, None, # for handle and training arguments in forward
                grad_input,
                grad_c1w, grad_c2w, grad_c3w, grad_c4w,
                grad_b1w, grad_b2w, grad_b3w, grad_b4w,
                grad_b1b, grad_b2b, grad_b3b, grad_b4b,
                None,     None,     None,     None, # for means
                None,     None,     None,     None) # for vars


# Generic monolithic bottleneck class for batchnorm/groupnorm bottleneck (with a control flag for the norm in the constructor and if-switches)
class BottleneckTPP(BlockedModule, Bottleneck_base):

    def __init__(self, inplanes, planes, eps, stride=1, use_physical_3x3_padding=False, downsample1=None, downsample2=None, use_groupnorm=False, dtype=torch.float,
                  bc_conv1=None, bc_conv2=None, bc_conv3=None, bk_conv3=None):
        super(BottleneckTPP, self).__init__(inplanes, planes, eps, stride, downsample1, downsample2, use_ref_conv=False, use_ref_norm=False, use_groupnorm=use_groupnorm, dtype=dtype,
                                            bc_conv1=bc_conv1, bc_conv2=bc_conv2, bc_conv3=bc_conv3, bk_conv3=bk_conv3)

        print("debug: BottleneckTPP constructor called with inplanes, planes, eps, stride, downsample1, downsample2 use_groupnorm dtype = ",
                  inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm, dtype)

        self.xsmm_handle = None
        self.norm_eps = self.bn1.eps
        if not use_groupnorm:
            self.bn_momentum = self.bn1.momentum
            self.bn_track_running_stats = self.bn1.track_running_stats
        self.dtype    = dtype
        self.use_bf16 = True if self.dtype == torch.bfloat16 else False
        self.use_physical_3x3_padding = use_physical_3x3_padding
        self.use_groupnorm = use_groupnorm

        #print("dbg: dtype use_bf16 = " , self.dtype, self.use_bf16)

        self.inplanes  = inplanes
        self.planes    = planes
        self.expansion = 4
        if self.use_groupnorm:
            self.G = 32 # hardcoded for now

        self.blocked_input_signature = blocked_layout.get_blocking_signature(
            "NCHW", "NCHWC"
        )
        self.blocked_output_signature = self.blocked_input_signature

        if downsample1 != None and downsample2 != None:
            self.has_residual_conv = True
        else:
            if self.stride != 1 or self.inplanes != self.planes * self.expansion:
                print("Error: downsample1/downsample2 are None but the stride/planes configuration says that they should be present!")
                exit()
            self.has_residual_conv = False
            self.dummy_tensor      = torch.empty(1)
        #print("debug: has_residual_conv = ", self.has_residual_conv)

        if not self.use_groupnorm and self.bn_track_running_stats:
            #print("Setting running stats to default values")
            self.bn1.running_mean.zero_()
            self.bn1.running_var.fill_(1)
            self.bn2.running_mean.zero_()
            self.bn2.running_var.fill_(1)
            self.bn3.running_mean.zero_()
            self.bn3.running_var.fill_(1)
            if self.has_residual_conv == True:
                self.downsample2.running_mean.zero_()
                self.downsample2.running_var.fill_(1)

    def maybe_block(self):
        for m in self.modules():
            if hasattr(m, "maybe_block_params"):
                m.maybe_block_params()

    def forward(self, input):

        #print("bn1.running_mean at the start of forward = ", self.bn1.running_mean)

        self.maybe_block()

        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        if self.xsmm_handle == None:
            if self.use_groupnorm:
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                      print("BottleneckTPP Create GN-powered handle: ", N, self.inplanes, self.H, self.W, self.planes, self.stride, self.norm_eps, self.dtype)
                self.xsmm_handle = BottleneckGNHandleTPP(N, self.inplanes, self.H, self.W, self.planes, self.G, self.stride,
                                                         self.norm_eps,
                                                         self.expansion, self.use_physical_3x3_padding, self.dtype)
                self.bn1.mean   = torch.empty(N, self.G)
                self.bn1.var    = torch.empty(N, self.G)
                self.bn1.invstd = torch.empty(N, self.G)
                self.bn2.mean   = torch.empty(N, self.G)
                self.bn2.var    = torch.empty(N, self.G)
                self.bn2.invstd = torch.empty(N, self.G)
                self.bn3.mean   = torch.empty(N, self.G)
                self.bn3.var    = torch.empty(N, self.G)
                self.bn3.invstd = torch.empty(N, self.G)
                if self.has_residual_conv == True:
                    self.downsample2.mean   = torch.empty(N, self.G)
                    self.downsample2.var    = torch.empty(N, self.G)
                    self.downsample2.invstd = torch.empty(N, self.G)
            else:
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                      print("BottleneckTPP Create BN-powered handle: ", N, self.inplanes, self.H, self.W, self.planes, self.stride, self.norm_eps, self.bn_momentum, self.bn_track_running_stats, self.dtype,
                                                                        self.bc_conv1, self.bc_conv2, self.bc_conv3, self.bk_conv3 )
                self.xsmm_handle = BottleneckBNHandleTPP(N, self.inplanes, self.H, self.W, self.planes, self.stride,
                                                         self.norm_eps, self.bn_momentum, 1 if self.bn_track_running_stats else 0,
                                                         self.expansion, self.use_physical_3x3_padding, self.dtype,
                                                         self.bc_conv1, self.bc_conv2, self.bc_conv3, self.bk_conv3)
            self.N = N
            #print("dbg: created the handle in BottleneckBNTPP")

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.conv1.Cblock, None, None],
        )

        #if not self.training and self.track_running_stats: # using during evaluation the running_mean and running_var computed during training beforehand
        #    output = XsmmBNTPP.apply(blocked_input, blocked_input_add, self.weight, self.bias, self.running_mean, self.running_var, self.invstd, self.xsmm_handle, output_size, self.training)
        #else:
        #    output = XsmmBNTPP.apply(blocked_input, blocked_input_add, self.weight, self.bias, self.mean, self.var, self.invstd, self.xsmm_handle, output_size, self.training)

        #attrs = vars(self.bn1)
        #print(', '.join("{}: {}".format(item[0], item[1]) for item in attrs.items()))

        if not self.use_groupnorm and not self.training and self.bn_track_running_stats: # using during evaluation the running_mean and running_var computed during training beforehand
            if self.has_residual_conv == True:
                inputs = [blocked_input,
                          self.conv1.weight, self.conv2.weight, self.conv3.weight, self.downsample1.weight,
                          self.bn1.weight, self.bn2.weight, self.bn3.weight, self.downsample2.weight,
                          self.bn1.bias, self.bn2.bias, self.bn3.bias, self.downsample2.bias,
                          self.bn1.running_mean, self.bn2.running_mean, self.bn3.running_mean, self.downsample2.running_mean,
                          self.bn1.running_var, self.bn2.running_var, self.bn3.running_var, self.downsample2.running_var]
            else:
                inputs = [blocked_input,
                          self.conv1.weight, self.conv2.weight, self.conv3.weight, self.dummy_tensor,
                          self.bn1.weight, self.bn2.weight, self.bn3.weight, self.dummy_tensor,
                          self.bn1.bias, self.bn2.bias, self.bn3.bias, self.dummy_tensor,
                          self.bn1.running_mean, self.bn2.running_mean, self.bn3.running_mean, self.dummy_tensor,
                          self.bn1.running_var, self.bn2.running_var, self.bn3.running_var, self.dummy_tensor]
        else:
            if self.has_residual_conv == True:
                inputs = [blocked_input,
                          self.conv1.weight, self.conv2.weight, self.conv3.weight, self.downsample1.weight,
                          self.bn1.weight, self.bn2.weight, self.bn3.weight, self.downsample2.weight,
                          self.bn1.bias, self.bn2.bias, self.bn3.bias, self.downsample2.bias,
                          self.bn1.mean, self.bn2.mean, self.bn3.mean, self.downsample2.mean,
                          self.bn1.var, self.bn2.var, self.bn3.var, self.downsample2.var]
            else:
                inputs = [blocked_input,
                          self.conv1.weight, self.conv2.weight, self.conv3.weight, self.dummy_tensor,
                          self.bn1.weight, self.bn2.weight, self.bn3.weight, self.dummy_tensor,
                          self.bn1.bias, self.bn2.bias, self.bn3.bias, self.dummy_tensor,
                          self.bn1.mean, self.bn2.mean, self.bn3.mean, self.dummy_tensor,
                          self.bn1.var, self.bn2.var, self.bn3.var, self.dummy_tensor]
        # Computations happen here
        #print("dbg: calling BottleneckApplyTPP inside BottleneckBNTPP")
        if self.use_groupnorm:
            output = BottleneckApplyGNTPP.apply(self.xsmm_handle, self.training, *inputs)
        else:
            output = BottleneckApplyBNTPP.apply(self.xsmm_handle, self.training, *inputs)

        #print("dbg: called BottleneckApplyTPP inside BottleneckBNTPP")
        blocked_output = blocked_layout.BlockedTensor(output, self.blocked_output_signature)

        if not self.use_groupnorm and self.training and self.bn_track_running_stats:
            #print("Updating running stats")
            #print("bn1.running_mean before = ", self.bn1.running_mean)
            self.bn1.running_mean = (1 - self.bn1.momentum) * self.bn1.running_mean + self.bn1.momentum * self.bn1.mean
            #print("bn1.momentum = ", self.bn1.momentum)
            #print("bn1.mean = ", self.bn1.mean)
            #print("bn1.running_mean = ", self.bn1.running_mean)
            #exit()
            self.bn1.running_var  = (1 - self.bn1.momentum) * self.bn1.running_var  + self.bn1.momentum * self.bn1.var
            self.bn2.running_mean = (1 - self.bn2.momentum) * self.bn2.running_mean + self.bn2.momentum * self.bn2.mean
            self.bn2.running_var  = (1 - self.bn2.momentum) * self.bn2.running_var  + self.bn2.momentum * self.bn2.var
            self.bn3.running_mean = (1 - self.bn3.momentum) * self.bn3.running_mean + self.bn3.momentum * self.bn3.mean
            self.bn3.running_var  = (1 - self.bn3.momentum) * self.bn3.running_var  + self.bn3.momentum * self.bn3.var
            if self.has_residual_conv == True:
                self.downsample2.running_mean = (1 - self.downsample2.momentum) * self.downsample2.running_mean + self.downsample2.momentum * self.downsample2.mean
                self.downsample2.running_var  = (1 - self.downsample2.momentum) * self.downsample2.running_var  + self.downsample2.momentum * self.downsample2.var

        #print("dbg: returning the blocked output from BottleneckBNTPP")

        return blocked_output


def init_libxsmm():
    pcl_cgbp_cpp.init_libxsmm()

# For using PCL PT conv module, a hack is done in main.py (not here in the definition of the context manager)
class ImplContextManager:
        def __init__(self, use_ref_conv = False, use_ref_bn = False, use_ref_gn = False, use_ref_pool = False, use_ref_fc = False):
            print("use_ref_conv:")
            print(use_ref_conv)
            print("use_ref_bn:")
            print(use_ref_bn)
            print("use_ref_gn:")
            print(use_ref_gn)
            print("use_ref_pool:")
            print(use_ref_pool)
            print("use_ref_fc:")
            print(use_ref_fc)

            #nn_Conv2d = torch.nn.Conv2d
            if use_ref_conv == True:
                print("Using reference PyTorch conv2d in the context manager")
            else:
                #torch.nn.Conv2d = XsmmConv2d
                #print("Using XsmmConv2d in the context manager")
                torch.nn.Conv2d = XsmmConv2dTPP
                print("Using XsmmConv2d TPP in the context manager")

            #nn_GroupNorm = torch.nn.GroupNorm
            if use_ref_gn == True:
                print("Using reference PyTorch GroupNorm in the context manager")
            else:
                #torch.nn.GroupNorm = XsmmGroupNorm
                #print("Using XsmmGroupNorm in the context manager")
                torch.nn.GroupNorm = XsmmGroupNormTPP
                print("Using XsmmGroupNormTPP in the context manager")

            #nn_BatchNorm2d = torch.nn.BatchNorm2d
            if use_ref_bn == True:
                print("Using reference PyTorch BatchNorm2d in the context manager")
            else:
                #torch.nn.BatchNorm2d = XsmmBatchNorm
                #print("Using XsmmBatchNorm in the context manager")
                torch.nn.BatchNorm2d = XsmmBatchNormTPP
                print("Using XsmmBatchNormTPP in the context manager")

            #nn_MaxPool2d = torch.nn.MaxPool2d
            if use_ref_pool == True:
                print("Using reference PyTorch MaxPool2d in the context manager")
            else:
                #torch.nn.MaxPool2d = XsmmMaxPool2d
                #print("Using XsmmMaxPool2d in the context manager")
                torch.nn.MaxPool2d = XsmmMaxPoolTPP2d
                print("Using XsmmMaxPoolTPP2d in the context manager")

            #nn_AvgPool2d = torch.nn.AvgPool2d
            if use_ref_pool == True:
                print("Using reference PyTorch AvgPool2d in the context manager")
            else:
                #torch.nn.AvgPool2d = XsmmAvgPool2d
                #print("Using XsmmAvgPool2d in the context manager")
                torch.nn.AvgPool2d = XsmmAvgPoolTPP2d
                print("Using XsmmAvgPoolTPP2d in the context manager")

            #nn_Fc = torch.nn.Linear
            if use_ref_fc == True:
                print("Using reference PyTorch Linear (Fc) in the context manager")
            else:
                torch.nn.Linear = XsmmLinearTPP
                print("Using XsmmLinearTPP in the context manager")

        def __enter__(self):
            print("Entering the context (ImplContextManager)...")
        def __exit__(self, exc_type, exc_value, exc_tb):
            print("Leaving the context (ImplContextManager)...")
            print(exc_type, exc_value, exc_tb, sep="\n")


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from pcl_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
    get_blocking_signature,
)
from pcl_pytorch_extension._C import _conv as conv_cpp
import time
from contextlib import contextmanager

USE_BF16_PARAMS = True

class DummyConvTPP(Function):
    @staticmethod
    def forward(ctx, param_struct, *inputs):

        output, = conv_cpp.conv_fwd(param_struct, inputs)

        ( input, weight) = inputs

        ctx.save_for_backward(input, weight)
        ctx.param_struct = param_struct

        print("debug: conv input shape = ",  input.shape)
        print("debug: conv output shape = ", output.shape)

        for i in range(10):
            ind = i + 0 #shift_output
            print("debug: i fwd input      = ", ind, input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_output
            print("debug: i fwd output     = ", ind, output.view(-1)[ind].item())

        return output

    @staticmethod
    def backward(ctx, *grad_outs): #grad_output):

        inputs = []
        inputs += [g.contiguous() for g in grad_outs]

        inputs += ctx.saved_tensors

        input, weight = ctx.saved_tensors
        param_struct = ctx.param_struct

        if input.requires_grad:
          grad_input, grad_weight = conv_cpp.conv_bwd(param_struct, inputs) #grad_output, input, weight)
        else:
          grad_input    = None
          [grad_weight] = conv_cpp.conv_bwd(param_struct, inputs) #handle.handle, grad_output, input, weight)

        return (grad_input, grad_weight)

class DummyConv2dTPP(BlockedModule, torch.nn.Conv2d):
    r"""PCL Conv2d module for using libxsmm Conv TPP"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', dtype=torch.float):
        self.C            = in_channels
        self.C_pad        = in_channels

        self.dtype        = dtype
        self.use_bf16     = True if self.dtype == torch.bfloat16 else False

        if self.use_bf16 and self.C_pad%2 != 0:
          self.C_pad = self.C_pad + 1

        #super(XsmmConv2dTPP, self).__init__()
        #nn_Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=None, dtype=dtype)
        torch.nn.Conv2d.__init__(self, self.C_pad, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=None, dtype=dtype)

        self.N            = 0
        self.K            = out_channels
        self.in_channels  = in_channels
        self.out_channels = out_channels
        #self.kernel_size  = _pair(kernel_size)
        self.R            = kernel_size
        self.S            = kernel_size
        #self.stride       = _pair(stride)
        self.stride       = stride
        #self.padding      = _pair(padding)
        self.pad_h        = padding
        self.pad_w        = padding
        self.strides      = stride
        self.pads         = padding
        #self.dilation     = _pair(dilation)
        self.config       = None

        self.weight = BlockedParameter(self.weight.data)
        if bias:
          self.bias = Parameter(torch.empty(self.K))

        [self.Cblock, self.Kblock, self.lp_block] = conv_cpp.conv_get_feature_map_blocks(self.C_pad, self.K, 0 if self.dtype == torch.float else 1)
        #print("debug: Cblock, Kblock, lp_block = ", self.Cblock, self.Kblock, self.lp_block)

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

        self.blocked_input_signature = get_blocking_signature(
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

        #if self.C != self.C_pad:
        if input.shape[1] != self.C_pad:
          #input = torch.cat((input, self.zero_pad), dim=-1)
          input = torch.cat((input, self.zero_pad), dim=1)

        if N != self.N:
          self.N = N

        if self.config == None:
            # 0s for physical padding
            self.config = conv_cpp.conv_setup(self.N, self.C, self.H, self.W, self.K, self.R, self.S, self.pad_h, self.pad_w, 0, 0, 0, 0, self.stride, 0 if self.dtype == torch.float else 1)

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.Cblock, None, None],
        )

        #output_size = pcl_cgbp_cpp.get_conv_tensor_layout(self.xsmm_handle.handle, "output");

        inputs = [blocked_input, self.weight]

        output = DummyConvTPP.apply(self.config, *inputs) #blocked_input, self.weight, self.xsmm_handle, output_size)

        blocked_output = BlockedTensor(output, self.blocked_output_signature)

        #for i in range(10):
        #    print("i blocked_output = ", i, blocked_output.unblocked_tensor().view(-1)[i].item())


        return blocked_output

@contextmanager
def pcl_impl(enable=True, use_bf16=False):
    pass

def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

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

class DummyConvTPP(Function):
    @staticmethod
    def forward(ctx, param_struct, tuning_params, tuning_string, tuning_timings_fwd, tuning_params_d, tuning_string_d, tuning_timings_d, tuning_params_w, tuning_string_w, tuning_timings_w, *inputs):

        output = conv_cpp.conv_fwd(param_struct, inputs)

        ( input, weight) = inputs

        ctx.save_for_backward(input, weight)
        ctx.param_struct = param_struct

        ctx.tuning_params_d    = tuning_params_d
        ctx.tuning_string_d    = tuning_string_d
        ctx.tuning_timings_d   = tuning_timings_d
        ctx.tuning_params_w    = tuning_params_w
        ctx.tuning_string_w    = tuning_string_w
        ctx.tuning_timings_w   = tuning_timings_w

        """
        print("debug: conv input shape = ",  input.shape)
        print("debug: conv output shape = ", output.shape)

        for i in range(10):
            ind = i + 0 #shift_input
            print("debug: i fwd input      = ", ind, input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_weight
            print("debug: i fwd weight     = ", ind, weight.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_output
            print("debug: i fwd output     = ", ind, output.view(-1)[ind].item())
        """

        return output

    @staticmethod
    def backward(ctx, *grad_outs): #grad_output):

        inputs = []
        inputs += [g.contiguous() for g in grad_outs]

        inputs += ctx.saved_tensors

        input, weight = ctx.saved_tensors
        grad_output, = grad_outs
        param_struct = ctx.param_struct

        tuning_params_d    = ctx.tuning_params_d
        tuning_string_d   = ctx.tuning_string_d
        tuning_timings_d   = ctx.tuning_timings_d
        tuning_params_w    = ctx.tuning_params_w
        tuning_string_w   = ctx.tuning_string_w
        tuning_timings_w   = ctx.tuning_timings_w

        """
        for i in range(10):
            ind = i + 0 #shift_weight
            print("debug: i bwd weight before = ", ind, weight.view(-1)[ind].item())
        """

        if input.requires_grad:
            if (tuning_params_d is None or tuning_string_d is None or len(tuning_params_d) == 0 or len(tuning_string_d) == 0 or tuning_timings_d is None):
                grad_input = conv_cpp.conv_bwd_d(param_struct, inputs) #handle.handle, grad_output, input, weight)
            else:
                grad_input = conv_cpp.conv_bwd_d_ext(param_struct, inputs, tuning_params_d, tuning_string_d, tuning_timings_d) #handle.handle, grad_output, input, weight)
        else:
            grad_input    = None

        if (tuning_params_w is None or tuning_string_w is None or len(tuning_params_w) == 0 or len(tuning_string_w) == 0 or tuning_timings_w is None):
            grad_weight = conv_cpp.conv_bwd_w(param_struct, inputs) #handle.handle, grad_output, input, weight)
        else:
            grad_weight = conv_cpp.conv_bwd_w_ext(param_struct, inputs, tuning_params_w, tuning_string_w, tuning_timings_w) #handle.handle, grad_output, input, weight)

        """
        if input.requires_grad:
          grad_input, grad_weight = conv_cpp.conv_bwd(param_struct, inputs) #grad_output, input, weight)
        else:
          grad_input    = None
          [grad_weight] = conv_cpp.conv_bwd(param_struct, inputs) #handle.handle, grad_output, input, weight)
        """

        """
        print("debug: pad_h = ", param_struct.pad_h)
        padding = [param_struct.pad_h, param_struct.pad_h, param_struct.pad_h, param_struct.pad_h]  #ctx.padding
        print("debug: input shape, grad_input shape = ", input.shape, grad_input.shape)
        if padding[0] != 0 or padding[1] != 0 or padding[2] != 0 or padding[3] != 0:
            [N, CP, ifhp, ifwp, bc] = input.shape
            shift_input  = (padding[0] * ifwp + padding[1])*bc - 5
            [N, KP, ofhp, ofwp, bk] = grad_output.shape
            shift_output = (padding[2] * ofwp + padding[3])*bk - 5
            print("shift_input shift_output = ", shift_input, shift_output)
        else:
            shift_input  = 0
            shift_output = 0

        for i in range(10):
            ind = i + shift_input
            print("debug: i bwd input        = ", ind, input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_weight
            print("debug: i bwd weight       = ", ind, weight.view(-1)[ind].item())
        for i in range(10):
            ind = i + shift_output
            print("debug: i bwd gradout      = ", ind, grad_output.view(-1)[ind].item())

        if input.requires_grad:
            for i in range(10):
                ind = i + shift_output
                print("debug: ind bwd grad_input = ", ind, grad_input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0 #shift_weight
            print("debug: i bwd grad weight  = ", ind, grad_weight.view(-1)[ind].item())
        """

        return (None, # for param_struct
                None, None, None, # for tuning_params, tuning_string and tuning_timings_fwd
                None, None, None, # for tuning_params_d, tuning_string_d, tuning_timings_d
                None, None, None, # for tuning_params_w, tuning_string_w and tuning_timings_w
                grad_input, grad_weight)

class DummyConv2dTPP(BlockedModule, torch.nn.Conv2d):
    r"""PCL Conv2d module for using libxsmm Conv TPP"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', dtype=torch.float, bc=None, bk=None):

        self.C            = in_channels
        self.C_pad        = in_channels

        self.dtype        = dtype
        self.use_bf16     = True if self.dtype == torch.bfloat16 else False

        if self.use_bf16 and self.C_pad%2 != 0:
          self.C_pad = self.C_pad + 1

        #super(XsmmConv2dTPP, self).__init__()
        #nn_Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=None, dtype=dtype)
        #print(torch.__version__)
        #print(torch.__file__)
        #for arg in (self.C_pad, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, None, dtype):
        #    print("type of arg, arg = ", type(arg), arg)
        #torch.nn.Conv2d.__init__(self, self.C_pad, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device=None, dtype=dtype)
        #torch.nn.Conv2d.__init__(self, in_channels=self.C_pad, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode, device=None, dtype=dtype)
        torch.nn.Conv2d.__init__(self, self.C_pad, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype=dtype)

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

        self.blocked_input_signature = get_blocking_signature(
            "NCHW", "NCHWC"
        )
        self.blocked_output_signature = self.blocked_input_signature

        self.bc = bc
        self.bk = bk

        if self.bc is not None or self.bk is not None:
            self.preset_blocksizes = True
            self.Cblock   = bc
            self.Kblock   = bk
            self.lp_block = 1 if self.dtype == torch.float else 2
        else:
            self.preset_blocksizes = False
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

    def maybe_block_params(self):
        self.weight.block()
        #self.bias.block()

    def forward(self, input,
                      tuning_params=None, tuning_string=None, tuning_timings_fwd=None,
                      tuning_params_d=None, tuning_string_d=None, tuning_timings_d=None,
                      tuning_params_w=None, tuning_string_w=None, tuning_timings_w=None):
        #print('Conv Input {} Padding:{} Stride:{}'.format(input.shape, self.padding, self.stride))
        self.maybe_block_params()

        N = input.size(0)
        self.H = input.size(2) - 2 * self.pad_h
        self.W = input.size(3) - 2 * self.pad_w

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
            # only physical padding is supported for now
            if self.preset_blocksizes:
                self.config = conv_cpp.conv_setup_preset(self.N, self.C, self.H, self.W, self.K, self.R, self.S,
                                                         self.pad_h, self.pad_w, self.pad_h, self.pad_w, self.pad_h, self.pad_w,
                                                         self.stride, 0 if self.dtype == torch.float else 1,
                                                         self.bc, self.bk) #, self.avoid_fmas_in_rim)
            else:
                self.config = conv_cpp.conv_setup(self.N, self.C, self.H, self.W, self.K, self.R, self.S,
                                                  self.pad_h, self.pad_w, self.pad_h, self.pad_w, self.pad_h, self.pad_w,
                                                  self.stride, 0 if self.dtype == torch.float else 1)

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.Cblock, None, None],
        )

        #output_size = pcl_cgbp_cpp.get_conv_tensor_layout(self.xsmm_handle.handle, "output");

        inputs = [blocked_input, self.weight]

        output = DummyConvTPP.apply(self.config, tuning_params, tuning_string, tuning_timings_fwd,
                                                tuning_params_d, tuning_string_d, tuning_timings_d,
                                                tuning_params_w, tuning_string_w, tuning_timings_w, *inputs ) #blocked_input, self.weight, self.xsmm_handle, output_size)

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

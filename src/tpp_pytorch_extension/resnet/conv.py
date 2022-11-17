import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import Conv2d as pytorch_conv2d
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

import numpy as np

class DummyConvTPP(Function):
    @staticmethod
    def forward(ctx, param_struct, tuning_params_fwd, tuning_string_fwd, tuning_timings_fwd, tuning_params_d, tuning_string_d, tuning_timings_d, tuning_params_w, tuning_string_w, tuning_timings_w, *inputs):


        if tuning_params_fwd is None or tuning_string_fwd is None or len(tuning_params_fwd) == 0:
            output = conv_cpp.conv_fwd(param_struct, inputs)
        else:
            if tuning_timings_fwd is None:
                tuning_timings_fwd = np.zeros(16, dtype=np.float32)
            output = conv_cpp.conv_fwd_ext(param_struct, inputs, tuning_params_fwd, tuning_string_fwd, tuning_timings_fwd)

        #output = conv_cpp.conv_fwd(param_struct, inputs)

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
            if (tuning_params_d is None or tuning_string_d is None or len(tuning_params_d) == 0 or len(tuning_string_d) == 0):
                if tuning_timings_d is None:
                    grad_input = conv_cpp.conv_bwd_d(param_struct, inputs)
                else:
                    print("Unsupported mode with tuning params_d empty but non-empty tuning_timings_d")
            else:
                if tuning_timings_d is None:
                    tuning_timings_d = np.zeros(16, dtype=np.float32)
                grad_input = conv_cpp.conv_bwd_d_ext(param_struct, inputs, tuning_params_d, tuning_string_d, tuning_timings_d)
        else:
            grad_input    = None

        if (tuning_params_w is None or tuning_string_w is None or len(tuning_params_w) == 0 or len(tuning_string_w) == 0):
            if tuning_timings_w is None:
                grad_weight = conv_cpp.conv_bwd_w(param_struct, inputs)
            else:
                print("Unsupported mode with tuning params_w empty but non-empty tuning_timings_w")
        else:
            if tuning_timings_w is None:
                tuning_timings_w = np.zeros(16, dtype=np.float32)
            grad_weight = conv_cpp.conv_bwd_w_ext(param_struct, inputs, tuning_params_w, tuning_string_w, tuning_timings_w)

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

class DummyConv2dTPP(BlockedModule, pytorch_conv2d):
    r"""PCL Conv2d module for using libxsmm Conv TPP"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros', dtype=torch.float,
                 bc=None, bk=None, logical_padding=None, use_hardcoded_tunings=None):

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
        pytorch_conv2d.__init__(self, self.C_pad, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype=dtype)

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

        self.logical_padding = logical_padding if logical_padding is not None else False
        self.use_hardcoded_tunings = use_hardcoded_tunings

        """
        # extra hack for the first convolution in resnet-50 which requires a logical padding and hardcoded block sizes
        if self.R == 7 and self.S == 7 and self.stride == 2 and (self.in_channels == 3 or self.in_channels == 4) and self.out_channels == 64:
            self.logical_padding = True
            bc = 4
            bk = 32
        """

        #print("dbg: for R = ", self.R, " S = ", self.S, " in the constructor of DummyConv2dTPP logical padding is set to ", self.logical_padding)

        self.pad_h_in = 0 if self.logical_padding else self.pad_h
        self.pad_w_in = 0 if self.logical_padding else self.pad_w
        self.pad_h_out = 0 if self.logical_padding else self.pad_h
        self.pad_w_out = 0 if self.logical_padding else self.pad_w

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

        #print("debug: preset_blocksizes = ", self.preset_blocksizes)
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

        self.tuning_params_fwd = None
        self.tuning_string_fwd = None
        self.tuning_params_d   = None
        self.tuning_string_d   = None
        self.tuning_params_w   = None
        self.tuning_string_w   = None

        # hardcoded for 56 threads on SPR
        if self.use_hardcoded_tunings:
            self.hybrid_cols = 14
            self.hybrid_rows = 4
            if self.use_bf16 == True:
                # bwd_d tunings are based on results in bottleneck_*_tuning_bwd_d_not1_0721.txt
                # bwd_w tunings are based on results in bottleneck_*_tuning_bwd_w_nohybrid_not1_0721.txt
                if self.R == 7 and self.S == 7 and (self.in_channels == 3 or self.in_channels == 4) and self.out_channels == 64: # First Resnet-50 v1.5 convolution (applied to 224x224 images)
                    self.tuning_params_fwd = [1, 2, 1, 1, # h,w,c,k block
                                              1, # h_in_gemm
                                              0 ] # pack_input
                    self.tuning_string_fwd = 'Afgbdced'
                    self.tuning_params_w = [1, # p_block
                                            0, # bf16_use_nchw_format
                                            1, 0, 0, # pack_input_upfront, fuse_upd_transposes, #use_f32_wt_reduction_and_external_wt_vnni
                                            1, 1, 0, # bf16_acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 56 ] # use_hybrid_imgfm_parallelization, n_img_teams, n_ofm_teams
                    self.tuning_string_w = 'C{C:56}A{R:1}bdef'

    def maybe_block_params(self):
        self.weight.block()
        #self.bias.block()

    def forward(self, input,
                      tuning_params_fwd=None, tuning_string_fwd=None, tuning_timings_fwd=None,
                      tuning_params_d=None, tuning_string_d=None, tuning_timings_d=None,
                      tuning_params_w=None, tuning_string_w=None, tuning_timings_w=None):

        l_tuning_params_fwd  = tuning_params_fwd if tuning_params_fwd is not None else self.tuning_params_fwd
        l_tuning_string_fwd  = tuning_string_fwd if tuning_string_fwd is not None else self.tuning_string_fwd
        l_tuning_params_d    = tuning_params_d if tuning_params_d is not None else self.tuning_params_d
        l_tuning_string_d    = tuning_string_d if tuning_string_d is not None else self.tuning_string_d
        l_tuning_params_w    = tuning_params_w if tuning_params_w is not None else self.tuning_params_w
        l_tuning_string_w    = tuning_string_w if tuning_string_w is not None else self.tuning_string_w

        #print('Conv Input {} Padding:{} Stride:{}'.format(input.shape, self.padding, self.stride))
        self.maybe_block_params()

        N = input.size(0)
        self.H = input.size(2) - 2 * self.pad_h_in
        self.W = input.size(3) - 2 * self.pad_w_in

        #input = input.to(global_dtype).contiguous()
        #weight = self.weight.to(global_dtype) #switched to self.weight below

        #if self.C != self.C_pad:
        if input.shape[1] != self.C_pad: # NCHW format for the input assumed (C is the second dim at least)
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
                self.config = conv_cpp.conv_setup_preset(self.N, self.C_pad, self.H, self.W, self.K, self.R, self.S,
                                                         self.pad_h, self.pad_w, self.pad_h_in, self.pad_w_in, self.pad_h_out, self.pad_w_out,
                                                         self.stride, 0 if self.dtype == torch.float else 1,
                                                         self.bc, self.bk) #, self.avoid_fmas_in_rim)
            else:
                self.config = conv_cpp.conv_setup(self.N, self.C_pad, self.H, self.W, self.K, self.R, self.S,
                                                  self.pad_h, self.pad_w, self.pad_h_in, self.pad_w_in, self.pad_h_out, self.pad_w_out,
                                                  self.stride, 0 if self.dtype == torch.float else 1)

        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, self.Cblock, None, None],
        )

        #output_size = pcl_cgbp_cpp.get_conv_tensor_layout(self.xsmm_handle.handle, "output");

        inputs = [blocked_input, self.weight]

        output = DummyConvTPP.apply(self.config, l_tuning_params_fwd, l_tuning_string_fwd, tuning_timings_fwd,
                                                 l_tuning_params_d, l_tuning_string_d, tuning_timings_d,
                                                 l_tuning_params_w, l_tuning_string_w, tuning_timings_w, *inputs ) #blocked_input, self.weight, self.xsmm_handle, output_size)

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

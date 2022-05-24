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
from pcl_pytorch_extension._C import _batchnorm as batchnorm_cpp
import time
from contextlib import contextmanager

USE_BF16_PARAMS = True

class DummyLinear(BlockedModule):
    def __init__(self, in_features, out_features, bias=True):
        super(DummyLinear, self).__init__()
        self.weight = BlockedParameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = BlockedParameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        raise NotImplemented
        return input

class DummyBatchNormFunction(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, p, training, need_attention_output, *inputs):
    def forward(ctx, training, relu, eltwise, eps, padding, *inputs):
        # print("DummyBatchNormFunction FWD Called")
        ( output, relu_mask ) = batchnorm_cpp.batchnorm_fwd(training, relu, eltwise, eps, padding, inputs)
        ( input, input_add, weight, bias, mean, var, invstd ) = inputs
        if training:
            ctx.save_for_backward(input, input_add, weight, mean, var, invstd, relu_mask, output)
        ctx.relu    = relu
        ctx.eltwise = eltwise
        ctx.eps     = eps
        ctx.padding = padding

        padding = ctx.padding
        print("debug: input shape, output shape = ", input.shape, output.shape)
        if ctx.padding[0] != 0:
            [N, CP, ifhp, ifwp, bc] = input.shape
            shift_input  = (padding[0] * ifwp + padding[1])*bc - 5
            [N, CP, ofhp, ofwp, bc] = output.shape
            shift_output = (padding[2] * ofwp + padding[3])*bc - 5
            print("shift_input shift_output = ", shift_input, shift_output)
        else:
            shift_input  = 0
            shift_output = 0

        print("debug: fwd relu eltwise eps = ", relu, eltwise, eps)
        for i in range(10):
            ind = i + shift_input
            print("debug: i fwd input      = ", ind, input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0
            print("debug: i fwd mean       = ", ind, mean.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0
            print("debug: i fwd var        = ", ind, var.view(-1)[ind].item())
        for i in range(10):
            ind = i + shift_output
            print("debug: i fwd output     = ", ind, output.view(-1)[ind].item())

        # print("Returning from DummyBatchNormFunction FWD")
        return output

    @staticmethod
    def backward(ctx, *grad_outs):
        # print("DummyBatchNormFunction BWD Called")
        inputs = []
        inputs += [g.contiguous() for g in grad_outs]

        inputs += ctx.saved_tensors
        (grad_input, grad_input_add, grad_weight, grad_bias) = batchnorm_cpp.batchnorm_bwd( ctx.relu, ctx.eltwise, ctx.eps, ctx.padding, inputs )

        (input, input_add, weight, mean, var, invstd, relu_mask, output) = ctx.saved_tensors

        padding = ctx.padding
        print("debug: input shape, grad_input shape = ", input.shape, grad_input.shape)
        if ctx.padding[0] != 0:
            [N, CP, ifhp, ifwp, bc] = input.shape
            shift_input  = (padding[0] * ifwp + padding[1])*bc - 5
            [N, CP, ofhp, ofwp, bc] = output.shape
            shift_output = (padding[2] * ofwp + padding[3])*bc - 5
            print("shift_input shift_output = ", shift_input, shift_output)
        else:
            shift_input  = 0
            shift_output = 0

        for i in range(10):
            ind = i + shift_output
            print("debug: ind bwd grad_output      = ", ind, grad_outs[0].view(-1)[ind].item())
        for i in range(10):
            ind = i + shift_input
            print("debug: ind bwd input            = ", ind, input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0
            print("debug: ind bwd weight           = ", ind, weight.view(-1)[ind].item())
        for i in range(10):
            ind = i + shift_input
            print("debug: ind bwd grad_input       = ", ind, grad_input.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0
            print("debug: ind bwd grad_weight      = ", ind, grad_weight.view(-1)[ind].item())
        for i in range(10):
            ind = i + 0
            print("debug: ind bwd grad_bias        = ", ind, grad_bias.view(-1)[ind].item())

        # print("Returning from DummyBatchNormFunction BWD")
        return (None, None, None, None, None, grad_input, grad_input_add, grad_weight, grad_bias, None, None, None)

class DummyBatchNormTPP(BlockedModule, torch.nn.BatchNorm2d):
    r"""PCL batchNorm TPP module for using libxsmm BN"""

    def __init__(self, num_channels, padding, eps, momentum=0.1, affine=True, track_running_stats=True, relu=False, eltwise=False, dtype=torch.float32):
        torch.nn.BatchNorm2d.__init__(self, num_channels, eps, momentum, affine, track_running_stats, device=None, dtype=dtype)

        self.C = num_channels
        self.padding = padding # a list of [pad_h_in, pad_w_in, pad_h_out, pad_w_out]
        self.N = 0
        self.affine = affine
        self.momentum = momentum
        self.xsmm_handle = None
        self.train_handle = None
        self.eval_handle = None
        self.relu = relu
        self.eltwise = eltwise
        self.track_running_stats = track_running_stats
        self.mean = torch.empty(num_channels)
        self.var = torch.empty(num_channels)
        self.invstd = torch.empty(num_channels)
        self.eps = eps
        self.dtype = dtype

        if len(padding) != 4:
            print("Error: padding must be supplied to DummyBatchNormTPP as a list of 4 elements")
            exit()

        self.blocked_input_signature = get_blocking_signature(
            "NCHW", "NCHWC"
        )
        self.blocked_output_signature = self.blocked_input_signature

        if self.affine:
          self.weight = Parameter(torch.Tensor(num_channels))
          self.bias   = Parameter(torch.Tensor(num_channels))
          #print("for self.affine = True dtype in XsmmBatchNormTPP constructor weight.dtype = ", dtype, self.weight.dtype)

          nn.init.constant_(self.weight, 1)
          nn.init.constant_(self.bias, 0)

        else:
          self.register_parameter('weight', None)
          self.register_parameter('bias', None)

        if self.track_running_stats:
          self.register_buffer('running_mean', torch.zeros(num_channels))
          self.register_buffer('running_var', torch.ones(num_channels))
          #self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        #else:
        #  self.register_parameter('running_mean', None)
        #  self.register_parameter('running_var', None)
        #  self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
          self.running_mean.zero_()
          self.running_var.fill_(1)
          #self.num_batches_tracked.zero_()

    def reset_parameters(self):
      self.reset_running_stats()
      if self.affine:
        self.weight.data.fill_(1.)
        self.bias.data.zero_()

    def forward(self, input, input_add = None):
        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        if input_add == None:
          input_add = torch.Tensor()

        #input = input.contiguous()
        #if self.eltwise:
        #  input_add = input_add.contiguous()

        if N != self.N:
          self.N = N
          self.Cblock = batchnorm_cpp.batchnorm_get_c_block(self.C)

        print("input shape = ", input.shape)

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

        #output_size = [self.N, self.C // self.Cblock, self.H, self.W, self.Cblock]

        if not self.training and self.track_running_stats: # using during evaluation the running_mean and running_var computed during training beforehand
            inputs = [ blocked_input, blocked_input_add, self.weight, self.bias, self.running_mean, self.running_var, self.invstd ]
            inputs = [ blocked_input, blocked_input_add, self.weight, self.bias, self.mean, self.var, self.invstd ]
            #output = XsmmBNTPP.apply(blocked_input, blocked_input_add, self.weight, self.bias, self.running_mean, self.running_var, self.invstd, self.xsmm_handle, output_size, self.training)
        else:
            inputs = [ blocked_input, blocked_input_add, self.weight, self.bias, self.mean, self.var, self.invstd ]
            #output = XsmmBNTPP.apply(blocked_input, blocked_input_add, self.weight, self.bias, self.mean, self.var, self.invstd, self.xsmm_handle, output_size, self.training)

        print("blocked_input shape = ", blocked_input.shape)

        output = DummyBatchNormFunction.apply(self.training, self.relu, self.eltwise, self.eps, self.padding, *inputs) #output_size, *inputs)

        print("momentum = ", self.momentum)

        for i in range(10):
            print("debug: i fwd running mean pre update     = ", i, self.running_mean.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd running var  pre update     = ", i, self.running_var.view(-1)[i].item())

        for i in range(10):
            print("debug: i fwd mean at update              = ", i, self.mean.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd var  at update              = ", i, self.var.view(-1)[i].item())

        if self.training and self.track_running_stats:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.var

        for i in range(10):
            print("debug: i fwd running mean post update    = ", i, self.running_mean.view(-1)[i].item())
        for i in range(10):
            print("debug: i bwd running var  post update    = ", i, self.running_var.view(-1)[i].item())

            # Unused?
            #if self.num_batches_tracked is not None:
            #    self.num_batches_tracked = self.num_batches_tracked + 1

        blocked_output = BlockedTensor(output, self.blocked_output_signature)

        return blocked_output

@contextmanager
def pcl_impl(enable=True, use_bf16=False):
    pass

def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

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

#import pcl_pytorch_extension
from pcl_pytorch_extension._C import _batchnorm as batchnorm_cpp
from pcl_pytorch_extension.resnet import batchnorm as batchnorm_py
from pcl_pytorch_extension.resnet import conv      as conv_py

from pcl_pytorch_extension._C import _bottleneck as bottleneck_cpp
from pcl_pytorch_extension._C import _conv       as conv_cpp
import time
from contextlib import contextmanager

#for debugging
import os
import numpy as np

import pcl_cgbp
import pcl_cgbp_cpp

# Generic base class for batchnorm/groupnorm bottleneck (with a control flag for the norm in the constructor)
# Copied from the CNN repo
class Bottleneck_base(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, eps, stride=1, downsample1=None, downsample2=None, use_groupnorm=False, dtype=torch.float):
        print("dbg: Bottleneck_base constructor called with inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm dtype = ",
                  inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm, dtype)
        super(Bottleneck_base, self).__init__()

        self.use_groupnorm = use_groupnorm
        self.dtype         = dtype

        # eltwise is accounted for in the forward()
        # but relu is created here for the PyTorch reference impl
        self.relu = nn.ReLU(inplace=False)

        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dtype=self.dtype)
        self.conv1 = conv_py.DummyConv2dTPP(inplanes, planes, kernel_size=1, bias=False, dtype=self.dtype)

        if self.use_groupnorm:
            self.bn1 = nn.GroupNorm(32, planes, eps)
        else:
            self.bn1 = batchnorm_py.DummyBatchNormTPP(planes, padding=[0, 0, 0, 0], eps=eps, relu=True, dtype=self.dtype)
            #self.bn1 = XsmmBatchNormTPP(planes, eps, relu=True, dtype=self.dtype)
            #self.bn1 = nn.BatchNorm2d(planes, eps)
            #self.bn1  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False, dtype=self.dtype)
        self.conv2 = conv_py.DummyConv2dTPP(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dtype=self.dtype)

        if self.use_groupnorm:
            self.bn2 = nn.GroupNorm(32, planes, eps, dtype=self.dtype)
        else:
            self.bn2 = batchnorm_py.DummyBatchNormTPP(planes, padding=[0, 0, 0, 0], eps=eps, relu=True, dtype=self.dtype)
            #self.bn2 = XsmmBatchNormTPP(planes, eps, relu=True, dtype=self.dtype)
            #self.bn2 = nn.BatchNorm2d(planes, eps, dtype=self.dtype)
            #self.bn2  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        #self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dtype=self.dtype)
        self.conv3 = conv_py.DummyConv2dTPP(planes, planes * 4, kernel_size=1, bias=False, dtype=self.dtype)

        if self.use_groupnorm:
            self.bn3 = nn.GroupNorm(32, planes * 4, eps, dtype=self.dtype)
        else:
            self.bn3 = batchnorm_py.DummyBatchNormTPP(planes * 4, padding=[0, 0, 0, 0], eps=eps, relu=True, eltwise=True, dtype=self.dtype)
            #self.bn3 = XsmmBatchNormTPP(planes * 4, eps, relu=True, eltwise=True, dtype=self.dtype)
            #self.bn3  = nn.BatchNorm2d(planes * 4, eps, dtype=self.dtype)
            #self.bn3  = nn.BatchNorm2d(planes * 4, eps, track_running_stats=False)

        self.stride = stride

        #self.downsample1 = downsample1 # this is the conv part of downsampling;
        #self.downsample2 = downsample2 # this is the bn   part of downsampling;

        # Otherwise loading a reference model's state_dict does not work
        if downsample1 != None and downsample2 != None:
            self.add_module("downsample1", downsample1)
            self.add_module("downsample2", downsample2)
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
        

        out = self.bn1(out)
        out = self.relu(out)

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
        

        out = self.bn2(out)
        out = self.relu(out)
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

        out = self.bn3(out)
        out += residual
        out = self.relu(out)

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

class BottleneckApplyBNTPP(Function):
    @staticmethod
    def forward(ctx, config, training, *inputs):

        #print("dbg: in bottleneck bn apply tpp forward")

        #bn_norm_type = 0 if training else 1

        #input_tensors = []
        #for entity in inputs:
        #    print("type of entity = ", type(entity))
        #    input_tensors.append(entity.data) # not necessary

        #print("type of inputs = ", type(inputs))
        #print("type of inputs[0] = ", type(inputs[0]))

        (input,
         c1w, c2w, c3w, c4w,
         b1w, b2w, b3w, b4w,
         b1b, b2b, b3b, b4b,
         b1m, b2m, b3m, b4m,
         b1n, b2n, b3n, b4n ) = inputs
         #c1s, c2s, c3s, c4s,
         #b1s, b2s, b3s, b4s ) = inputs

        #tmp_list = list(inputs)
        #tmp_list[0] = torch.ones_like(tmp_list[0])
        #tmp_list[1] = torch.ones_like(tmp_list[1])
        #inputs = tuple(tmp_list)

        #print("b4m = ", b4m);
        #print("b4n = ", b4n);

        #print("nan check in bottleneck for input before, nancount = ", torch.isnan(input.view(-1)).sum())
        output, conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out, bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out, b1s_out, b2s_out, b3s_out, b4s_out = bottleneck_cpp.bottleneck_bn_fwd(config, training, inputs) #_tensors)
        #print("dbg: bottleneck_forward_new called")

        if config.has_residual_conv == 0:
            dummy_tensor = torch.empty(1)
            bn4_relu_out = dummy_tensor
            conv4_out    = dummy_tensor

        (input,
         c1w, c2w, c3w, c4w,
         b1w, b2w, b3w, b4w,
         b1b, b2b, b3b, b4b,
         b1m, b2m, b3m, b4m,
         b1n, b2n, b3n, b4n ) = inputs
         #c1s, c2s, c3s, c4s, # should not be used as they were changed in fwd, _out are the actual versions at this moment
         #b1s, b2s, b3s, b4s ) = inputs

        """
        #print("nan check in bottleneck for input after, nancount = ", torch.isnan(input.view(-1)).sum())
        print("nan check in bottleneck for conv1_out, nancount = ", torch.isnan(conv1_out.view(-1)).sum())
        print("nan check in bottleneck for bn1_out, nancount = ", torch.isnan(bn1_out.view(-1)).sum())
        print("nan check in bottleneck for conv2_out, nancount = ", torch.isnan(conv2_out.view(-1)).sum())
        print("nan check in bottleneck for bn2_out, nancount = ", torch.isnan(bn2_out.view(-1)).sum())
        print("nan check in bottleneck for conv3_out, nancount = ", torch.isnan(conv3_out.view(-1)).sum())
        #print("nan check in bottleneck for residual, nancount = ", torch.isnan(residual.view(-1)).sum())
        print("nan check in bottleneck for conv4_out, nancount = ", torch.isnan(conv4_out.view(-1)).sum())
        print("nan check in bottleneck for bn4_out, nancount = ", torch.isnan(bn4_out.view(-1)).sum())
        print("nan check in bottleneck for bn3_out, nancount = ", torch.isnan(bn3_out.view(-1)).sum())
        print("nan check in bottleneck for output, nancount = ", torch.isnan(output.view(-1)).sum())
        """

        """
        dump = False

        global_tensor_x_counter = 0

        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if training:
            dump_file_suffix    = '_train_ext_tst' + '_rank_' + str(rank)
        else:
            dump_file_suffix    = '_eval_ext_tst' + '_rank_' + str(rank)

        if dump:

            #tmp_tensor = bn4_out.unblocked_tensor() if type(bn4_out) is BlockedTensor else bn4_out
            #np.savetxt('my_layer_bn4_fwd_output_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            #global_tensor_x_counter = global_tensor_x_counter + 1
            #exit()

            
            if type(c1w) is BlockedParameter:
                c1w.unblock()
            #else:
            tmp_tensor = c1w
            #tmp_tensor = c1w.unblocked_tensor() if type(c1w) is BlockedParameter else c1w
            np.savetxt('my_layer_conv1_forward_weight_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1
            

            tmp_tensor = input.unblocked_tensor() if type(input) is BlockedTensor else input
            np.savetxt('my_layer_conv1_forward_input_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            global_tensor_x_counter = global_tensor_x_counter + 1

            #print("types are ", type(conv1_out), type(bn1_out), type(conv2_out), type(bn2_out), type(conv3_out), type(bn4_out), type(output))

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

        ctx.config = config

        ctx.save_for_backward(input, c1w, c2w, c3w, c4w, b1w, b2w, b3w, b4w, b1b, b2b, b3b, b4b, b1m, b2m, b3m, b4m, b1n, b2n, b3n, b4n,
                              conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out,
                              bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
                              b1s_out, b2s_out, b3s_out, b4s_out)
                              #c1s, c2s, c3s, c4s, b1s_out, b2s_out, b3s_out, b4s_out) # FIXME, must be cNs_out!

        return output

    @staticmethod
    def backward(ctx, *grad_outs):
        #print("dbg: in bottleneck apply backward")
        inputs = []
        inputs += [g.contiguous() for g in grad_outs]

        inputs += ctx.saved_tensors

        config = ctx.config
        # FIXME: Not sure if necessary
        #del ctx.xsmm_handle

        #for entity in inputs:
        #    print("type of entity = ", type(entity))


        (grad_c1w, grad_c2w, grad_c3w, grad_c4w,
         grad_b1w, grad_b2w, grad_b3w, grad_b4w,
         grad_b1b, grad_b2b, grad_b3b, grad_b4b,
         grad_c1i, grad_c4i) = bottleneck_cpp.bottleneck_bn_bwd(config, inputs) #_tensors)

        #print("dbg: bottleneck_backward_new called")

        grad_input = grad_c1i + grad_c4i

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
        """
        """
        for i in range(10):
            ind = i
            print("ind grad_c1w grad_c2w grad_c3w ", ind, grad_c1w.view(-1)[ind].item(), grad_c2w.view(-1)[ind].item(), grad_c3w.view(-1)[ind].item())

        for i in range(10):
            ind = i
            print("ind grad_b1w grad_b2w grad_b3w ", ind, grad_b1w.view(-1)[ind].item(), grad_b2w.view(-1)[ind].item(), grad_b3w.view(-1)[ind].item())

        for i in range(10):
            ind = i
            print("ind grad_b1b grad_b2b grad_b3b ", ind, grad_b1b.view(-1)[ind].item(), grad_b2b.view(-1)[ind].item(), grad_b3b.view(-1)[ind].item())

        for i in range(10):
            ind = i + 0 #shift_input
            print("ind grad_c1i grad_c4i ", ind, grad_c1i.view(-1)[ind].item(), grad_c4i.view(-1)[ind].item())
        """


        return (None, None, # for handle and training arguments in forward
                grad_input,
                grad_c1w, grad_c2w, grad_c3w, grad_c4w,
                grad_b1w, grad_b2w, grad_b3w, grad_b4w,
                grad_b1b, grad_b2b, grad_b3b, grad_b4b,
                None,     None,     None,     None, # for means
                None,     None,     None,     None) # for vars
                #None,     None,     None,     None, # for conv scratches
                #None,     None,     None,     None) # for bn   scratches


# Generic monolithic bottleneck class for batchnorm/groupnorm bottleneck (with a control flag for the norm in the constructor and if-switches)
class BottleneckTPP(BlockedModule, Bottleneck_base):

    def __init__(self, inplanes, planes, eps, stride=1, use_physical_3x3_padding=False, downsample1=None, downsample2=None, use_groupnorm=False, dtype=torch.float):
        super(BottleneckTPP, self).__init__(inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm=use_groupnorm, dtype=dtype)

        print("debug: BottleneckTPP constructor called with inplanes, planes, eps, stride, downsample1, downsample2 use_groupnorm dtype = ",
                  inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm, dtype)

        #self.xsmm_handle = None
        self.config = None
        self.norm_eps = self.bn1.eps
        if not use_groupnorm:
            self.bn_momentum = self.bn1.momentum
            self.bn_track_running_stats = self.bn1.track_running_stats
        self.dtype    = dtype
        self.use_bf16 = True if self.dtype == torch.bfloat16 else False
        self.use_physical_3x3_padding = use_physical_3x3_padding
        self.use_groupnorm = use_groupnorm
        """
        self.conv1_scratch = torch.Tensor()
        self.conv2_scratch = torch.Tensor()
        self.conv3_scratch = torch.Tensor()
        self.conv4_scratch = torch.Tensor()
        self.bn1_scratch   = torch.Tensor()
        self.bn2_scratch   = torch.Tensor()
        self.bn3_scratch   = torch.Tensor()
        self.bn4_scratch   = torch.Tensor()
        """
        #print("dbg: dtype use_bf16 = " , self.dtype, self.use_bf16)

        self.inplanes  = inplanes
        self.planes    = planes
        self.expansion = 4
        if self.use_groupnorm:
            self.G = 32 # hardcoded for now

        self.blocked_input_signature = get_blocking_signature(
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

        if self.config == None:
            if self.use_groupnorm:
                print("use_groupnorm not implemeneted in the bottleneck in extensions")
                exit()
                """
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
                """
            else:
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                      print("BottleneckTPP Create BN-powered handle: ", N, self.inplanes, self.H, self.W, self.planes, self.stride, self.norm_eps, self.bn_momentum, self.bn_track_running_stats, self.dtype)
                self.config = bottleneck_cpp.bottleneck_bn_setup(N, self.inplanes, self.H, self.W, self.planes, self.stride, self.norm_eps, self.bn_momentum, self.bn_track_running_stats, self.expansion,
                                                             1 if self.use_physical_3x3_padding else 0, 0 if self.dtype == torch.float else 1)
                #self.xsmm_handle = BottleneckBNHandleTPP(N, self.inplanes, self.H, self.W, self.planes, self.stride,
                #                                         self.norm_eps, self.bn_momentum, 1 if self.bn_track_running_stats else 0,
                #                                         self.expansion, self.use_physical_3x3_padding, self.dtype)

            self.N = N
            #print("dbg: created the handle in BottleneckBNTPP")

        [conv1_Cblock, conv1_Kblock, conv1_lpblock] = conv_cpp.conv_get_feature_map_blocks(self.conv1.in_channels, self.conv1.out_channels, 0 if self.dtype == torch.float else 1)
        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, conv1_Cblock, None, None],
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
                          #self.conv1_scratch, self.conv2_scratch, self.conv3_scratch, self.conv4_scratch,
                          #self.bn1_scratch, self.bn2_scratch, self.bn3_scratch, self.bn4_scratch ]
            else:
                inputs = [blocked_input,
                          self.conv1.weight, self.conv2.weight, self.conv3.weight, self.dummy_tensor,
                          self.bn1.weight, self.bn2.weight, self.bn3.weight, self.dummy_tensor,
                          self.bn1.bias, self.bn2.bias, self.bn3.bias, self.dummy_tensor,
                          self.bn1.running_mean, self.bn2.running_mean, self.bn3.running_mean, self.dummy_tensor,
                          self.bn1.running_var, self.bn2.running_var, self.bn3.running_var, self.dummy_tensor]
                          #self.conv1_scratch, self.conv2_scratch, self.conv3_scratch, self.dummy_tensor,
                          #self.bn1_scratch, self.bn2_scratch, self.bn3_scratch, self.dummy_tensor ]
        else:
            if self.has_residual_conv == True:
                inputs = [blocked_input,
                          self.conv1.weight, self.conv2.weight, self.conv3.weight, self.downsample1.weight,
                          self.bn1.weight, self.bn2.weight, self.bn3.weight, self.downsample2.weight,
                          self.bn1.bias, self.bn2.bias, self.bn3.bias, self.downsample2.bias,
                          self.bn1.mean, self.bn2.mean, self.bn3.mean, self.downsample2.mean,
                          self.bn1.var, self.bn2.var, self.bn3.var, self.downsample2.var]
                          #self.conv1_scratch, self.conv2_scratch, self.conv3_scratch, self.conv4_scratch,
                          #self.bn1_scratch, self.bn2_scratch, self.bn3_scratch, self.bn4_scratch ]
            else:
                inputs = [blocked_input,
                          self.conv1.weight, self.conv2.weight, self.conv3.weight, self.dummy_tensor,
                          self.bn1.weight, self.bn2.weight, self.bn3.weight, self.dummy_tensor,
                          self.bn1.bias, self.bn2.bias, self.bn3.bias, self.dummy_tensor,
                          self.bn1.mean, self.bn2.mean, self.bn3.mean, self.dummy_tensor,
                          self.bn1.var, self.bn2.var, self.bn3.var, self.dummy_tensor]
                          #self.conv1_scratch, self.conv2_scratch, self.conv3_scratch, self.dummy_tensor,
                          #self.bn1_scratch, self.bn2_scratch, self.bn3_scratch, self.dummy_tensor ]
        # Computations happen here
        #print("dbg: calling BottleneckApplyTPP inside BottleneckBNTPP")
        if self.use_groupnorm:
            output = BottleneckApplyGNTPP.apply(self.config, self.training, *inputs)
        else:
            output = BottleneckApplyBNTPP.apply(self.config, self.training, *inputs)

        #print("dbg: self.conv1_scratch numel after forward = ", self.conv1_scratch.numel())
        #print("dbg: self.bn1_scratch   numel after forward = ", self.bn1_scratch.numel())

        #print("dbg: called BottleneckApplyTPP inside BottleneckBNTPP")
        blocked_output = BlockedTensor(output, self.blocked_output_signature)

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

@contextmanager
def pcl_impl(enable=True, use_bf16=False):
    pass

def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

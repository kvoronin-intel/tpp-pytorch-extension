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

# for debugging performance
import time

# Generic base class for batchnorm/groupnorm bottleneck (with a control flag for the norm in the constructor)
# Copied from the CNN repo
class Bottleneck_base(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, eps, stride=1, downsample1=None, downsample2=None, use_groupnorm=False, dtype=torch.float, bc_conv1=None, bc_conv2=None, bc_conv3=None, bk_conv3=None, avoid_fmas_in_rim=None ):

        #print("dbg: Bottleneck_base constructor called with inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm dtype = ",
        #          inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm, dtype)
        super(Bottleneck_base, self).__init__()

        self.use_groupnorm = use_groupnorm
        self.dtype         = dtype

        self.bc_conv1          = bc_conv1
        self.bc_conv2          = bc_conv2
        self.bc_conv3          = bc_conv3
        self.bk_conv3          = bk_conv3
        self.avoid_fmas_in_rim = avoid_fmas_in_rim if avoid_fmas_in_rim != None else False
        if self.bc_conv1 is not None and self.bc_conv2 is not None and self.bc_conv3 is not None and self.bk_conv3 is not None:
            self.preset_blocksizes = True
        else:
            self.preset_blocksizes = False

        # eltwise is accounted for in the forward()
        # but relu is created here for the PyTorch reference impl
        self.relu = nn.ReLU(inplace=False)

        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dtype=self.dtype)
        self.conv1 = conv_py.DummyConv2dTPP(inplanes, planes, bc=bc_conv1, bk=bc_conv2, kernel_size=1, bias=False, dtype=self.dtype)

        if self.use_groupnorm:
            self.bn1 = nn.GroupNorm(32, planes, eps)
        else:
            self.bn1 = batchnorm_py.DummyBatchNormTPP(planes, bc=bc_conv2, padding=[0, 0, 0, 0], eps=eps, relu=True, dtype=self.dtype)
            #self.bn1 = XsmmBatchNormTPP(planes, eps, relu=True, dtype=self.dtype)
            #self.bn1 = nn.BatchNorm2d(planes, eps)
            #self.bn1  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                       padding=1, bias=False, dtype=self.dtype)
        self.conv2 = conv_py.DummyConv2dTPP(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dtype=self.dtype,
                               bc=bc_conv2, bk=bc_conv3)

        if self.use_groupnorm:
            self.bn2 = nn.GroupNorm(32, planes, eps, dtype=self.dtype)
        else:
            self.bn2 = batchnorm_py.DummyBatchNormTPP(planes, bc=bc_conv3, padding=[0, 0, 0, 0], eps=eps, relu=True, dtype=self.dtype)
            #self.bn2 = XsmmBatchNormTPP(planes, eps, relu=True, dtype=self.dtype)
            #self.bn2 = nn.BatchNorm2d(planes, eps, dtype=self.dtype)
            #self.bn2  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        #self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dtype=self.dtype)
        self.conv3 = conv_py.DummyConv2dTPP(planes, planes * 4, bc=bc_conv3, bk=bk_conv3, kernel_size=1, bias=False, dtype=self.dtype)

        if self.use_groupnorm:
            self.bn3 = nn.GroupNorm(32, planes * 4, eps, dtype=self.dtype)
        else:
            self.bn3 = batchnorm_py.DummyBatchNormTPP(planes * 4, bc=bk_conv3, padding=[0, 0, 0, 0], eps=eps, relu=True, eltwise=True, dtype=self.dtype)
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
    def forward(ctx, config, training, tuning_params, tuning_strings, tuning_timings_fwd, tuning_params_d, tuning_strings_d, tuning_params_w, tuning_strings_w, tuning_timings_bwd, *inputs ):

        #print("dbg: in bottleneck bn apply tpp forward")
        time_start = time.time()

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
        """
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if rank == 0:
            c1w_nan_count = torch.isnan(c1w.view(-1)).sum()
            print("nan check before in bottleneck for c1w, nancount = ", c1w_nan_count)
            c2w_nan_count = torch.isnan(c2w.view(-1)).sum()
            print("nan check before in bottleneck for c2w, nancount = ", c1w_nan_count)
            c3w_nan_count = torch.isnan(c3w.view(-1)).sum()
            print("nan check in bottleneck for c3w, nancount = ", c3w_nan_count)
            c4w_nan_count = torch.isnan(c4w.view(-1)).sum()
            print("nan check in bottleneck for c4w, nancount = ", c4w_nan_count)
            if c1w_nan_count > 0 or c2w_nan_count > 0 or c3w_nan_count > 0 or c4w_nan_count > 0:
                print("Exiting before doing the forward because nan count in conv weights is not zero")
                exit(-1)
        """

        """
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if rank == 0:
            c1w_nan_count = torch.isnan(c1w.view(-1)).sum()
            print("nan check in bottleneck for c1w, nancount = ", c1w_nan_count)
            c2w_nan_count = torch.isnan(c2w.view(-1)).sum()
            print("nan check in bottleneck for c2w, nancount = ", c2w_nan_count)
            c3w_nan_count = torch.isnan(c3w.view(-1)).sum()
            print("nan check in bottleneck for c3w, nancount = ", c3w_nan_count)
            if config.has_residual_conv:
                c4w_nan_count = torch.isnan(c4w.view(-1)).sum()
                print("nan check in bottleneck for c4w, nancount = ", c4w_nan_count)
            else:
                c4w_nan_count = 0
            b1w_nan_count = torch.isnan(b1w.view(-1)).sum()
            print("nan check in bottleneck for b1w, nancount = ", b1w_nan_count)
            b2w_nan_count = torch.isnan(b2w.view(-1)).sum()
            print("nan check in bottleneck for b2w, nancount = ", b2w_nan_count)
            b3w_nan_count = torch.isnan(b3w.view(-1)).sum()
            print("nan check in bottleneck for b3w, nancount = ", b3w_nan_count)
            if config.has_residual_conv:
                b4w_nan_count = torch.isnan(b4w.view(-1)).sum()
                print("nan check in bottleneck for b4w, nancount = ", b4w_nan_count)
            else:
                b4w_nan_count = 0
            if c1w_nan_count > 0 or c2w_nan_count > 0 or c3w_nan_count > 0 or c4w_nan_count > 0 or b1w_nan_count > 0 or b2w_nan_count > 0 or b3w_nan_count > 0 or b4w_nan_count > 0:
                print("Exiting before doing fwd because nan count is not zero")
                exit(-1)
        """

        if tuning_params is None or tuning_strings is None or len(tuning_params) == 0 or len(tuning_strings) == 0:
            (output,
            conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out,
            bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
            b1s_out, b2s_out, b3s_out, b4s_out ) = bottleneck_cpp.bottleneck_bn_fwd(config, training, inputs) #_tensors)
        else:
            if tuning_timings_fwd is None:
                tuning_timings_fwd = np.zeros(16, dtype=np.float32)
            (output,
            conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out,
            bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
            b1s_out, b2s_out, b3s_out, b4s_out ) = bottleneck_cpp.bottleneck_bn_fwd_ext(config, training, inputs, tuning_params, tuning_strings, tuning_timings_fwd) #_tensors)
        #print("dbg: bottleneck_forward_new called")

        """
        print("perfdebug: checking for bottleneck in bwd with cfg C K H W stride: ", config.inplanes, config.planes, config.H, config.W, config.stride)
        print("PERFDUMP,FP,resnetconv,"  + str(config.N) + "," + str(config.N) + "," + str(config.inplanes) + "," + str(config.planes)   + "," + str(config.H) + "," + str(config.W) + "," + str(1) + "," + str(1) + "," + str(1)             + "," + str(0) + "," + str(0) + "," + str(tuning_timings_fwd[0]) + "," + str(1.0))
        print("PERFDUMP,FP,resnetconv,"  + str(config.N) + "," + str(config.N) + "," + str(config.planes)   + "," + str(config.planes)   + "," + str(config.H) + "," + str(config.W) + "," + str(3) + "," + str(3) + "," + str(config.stride) + "," + str(1) + "," + str(1) + "," + str(tuning_timings_fwd[1]) + "," + str(1.0))
        print("PERFDUMP,FP,resnetconv,"  + str(config.N) + "," + str(config.N) + "," + str(config.planes)   + "," + str(4*config.planes) + "," + str(config.H // config.stride) + "," + str(config.W // config.stride) + "," + str(1) + "," + str(1) + "," + str(1)             + "," + str(0) + "," + str(0) + "," + str(tuning_timings_fwd[2]) + "," + str(1.0))
        if config.has_residual_conv:
            print("PERFDUMP,FP,resnetconv,"  + str(config.N) + "," + str(config.N) + "," + str(config.inplanes) + "," + str(4*config.planes) + "," + str(config.H) + "," + str(config.W) + "," + str(1) + "," + str(1) + "," + str(config.stride) + "," + str(0) + "," + str(0) + "," + str(tuning_timings_fwd[3]) + "," + str(1.0))

        print("PERFDUMP,FP,resnetbn,"    + str(config.N) + "," + str(config.N) + "," + str(config.planes)   + "," + str(config.planes)   + "," + str(config.H)                  + "," + str(config.W)                  + "," + "na" + "," + "na" + "," + "na" + "," + str(0) + "," + str(1) + "," + str(tuning_timings_fwd[4]) + "," + str(1.0) + ',' + str(1) + ',' + str(0) + ',' + str(training))
        print("PERFDUMP,FP,resnetbn,"    + str(config.N) + "," + str(config.N) + "," + str(config.planes)   + "," + str(config.planes)   + "," + str(config.H // config.stride) + "," + str(config.W // config.stride) + "," + "na" + "," + "na" + "," + "na" + "," + str(1) + "," + str(0) + "," + str(tuning_timings_fwd[5]) + "," + str(1.0) + ',' + str(1) + ',' + str(0) + ',' + str(training))
        print("PERFDUMP,FP,resnetbn,"    + str(config.N) + "," + str(config.N) + "," + str(4*config.planes) + "," + str(4*config.planes) + "," + str(config.H // config.stride) + "," + str(config.W // config.stride) + "," + "na" + "," + "na" + "," + "na" + "," + str(0) + "," + str(0) + "," + str(tuning_timings_fwd[6]) + "," + str(1.0) + ',' + str(1) + ',' + str(1) + ',' + str(training))
        if config.has_residual_conv:
            print("PERFDUMP,FP,resnetbn,"    + str(config.N) + "," + str(config.N) + "," + str(4*config.planes) + "," + str(4*config.planes) + "," + str(config.H // config.stride) + "," + str(config.W // config.stride)                  + "," + "na" + "," + "na" + "," + "na" + "," + str(0) + "," + str(0) + "," + str(tuning_timings_fwd[7]) + "," + str(1.0) + ',' + str(0) + ',' + str(0) + ',' + str(training))
        """
        #print("time: conv = ", config.inplanes, config.planes, config.H, config.W, 1, 1, tuning_timings_fwd[0], "(c1)")
        #print("time: conv = ", config.planes, config.planes, config.H, config.W, 1, config.stride, tuning_timings_fwd[1], "(c2)")
        #print("time: conv = ", config.planes, 4*config.planes, config.H, config.W, 1, 1, tuning_timings_fwd[2], "(c3)")
        #print("time: conv = ", config.inplanes, 4*config.planes, config.H, config.W, 1, config.stride, tuning_timings_fwd[3], "(c4)")
        #print("time: b1 = ", tuning_timings_fwd[4])
        #print("time: b2 = ", tuning_timings_fwd[5])
        #print("time: b3 = ", tuning_timings_fwd[6])
        #print("time: b4 = ", tuning_timings_fwd[7])
        #print("time: c1b1 = ", tuning_timings_fwd[8])
        #print("time: c2b2 = ", tuning_timings_fwd[9])
        #print("time: c3b3 = ", tuning_timings_fwd[10])
        #print("time: c4b4 = ", tuning_timings_fwd[11])

        if config.has_residual_conv == 0:
            dummy_tensor = torch.empty(1)
            bn4_relu_out = dummy_tensor
            conv4_out    = dummy_tensor


        """
        for i in range(64):
            ind = i
            print("ind c1w b1w c3w b3w", ind, c1w.view(-1)[ind].item(), b1w.view(-1)[ind].item(), c3w.view(-1)[ind].item(), b3w.view(-1)[ind].item())

        if config.has_residual_conv:
            for i in range(64):
                ind = i
                print("ind c1o b1o c2o b2o c3o b3o c4o b4o", ind, conv1_out.view(-1)[ind].item(), bn1_out.view(-1)[ind].item(), conv2_out.view(-1)[ind].item(), bn2_out.view(-1)[ind].item(), conv3_out.view(-1)[ind].item(), bn3_out.view(-1)[ind].item(), conv4_out.view(-1)[ind].item(), bn4_out.view(-1)[ind].item())
        else:
            for i in range(64):
                ind = i
                print("ind c1o b1o c2o b2o c3o b3o", ind, conv1_out.view(-1)[ind].item(), bn1_out.view(-1)[ind].item(), conv2_out.view(-1)[ind].item(), bn2_out.view(-1)[ind].item(), conv3_out.view(-1)[ind].item(), bn3_out.view(-1)[ind].item())

        for i in range(64):
            ind = i
            print("ind out", ind, output.view(-1)[ind].item())
        """

        """
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if rank == 0:
            c1w_nan_count = torch.isnan(c1w.view(-1)).sum()
            print("nan check in bottleneck for c1w, nancount = ", c1w_nan_count)
            conv1_nan_count = torch.isnan(conv1_out.view(-1)).sum()
            #print("nan check in bottleneck for input after, nancount = ", torch.isnan(input.view(-1)).sum())
            print("nan check in bottleneck for conv1_out, nancount = ", conv1_nan_count)
            bn1_nan_count = torch.isnan(bn1_out.view(-1)).sum()
            print("nan check in bottleneck for bn1_out, nancount = ", bn1_nan_count)
            c2w_nan_count = torch.isnan(c2w.view(-1)).sum()
            print("nan check in bottleneck for c2w, nancount = ", c2w_nan_count)
            conv2_full_nan_count = torch.isnan(conv2_out.view(-1)).sum()
            print("nan check in bottleneck for full conv2_out, nancount = ", conv2_full_nan_count)
            if conv2_full_nan_count > 0:
                conv2_out_zeroed_rim = torch.zeros_like(conv2_out)
                output_hw_padding = [1, 1, 1, 1]
                nchwc_shape = conv2_out.shape
                print("debug: nchwc shape = ", nchwc_shape)
                outH = nchwc_shape[2] - output_hw_padding[0] - output_hw_padding[1]
                outW = nchwc_shape[3] - output_hw_padding[2] - output_hw_padding[3]
                print("range = ", 'full', ' ', 'full', output_hw_padding[0], outH + output_hw_padding[0], output_hw_padding[2], outW + output_hw_padding[2])
                conv2_out_zeroed_rim[:,:,output_hw_padding[0]:outH + output_hw_padding[0],output_hw_padding[2]:outW + output_hw_padding[2]] = conv2_out[:,:,output_hw_padding[0]:outH + output_hw_padding[0],output_hw_padding[2]:outW + output_hw_padding[2]]

                conv2_nan_count = torch.isnan(conv2_out_zeroed_rim.view(-1)).sum()
            else:
                conv2_nan_count = conv2_full_nan_count
            print("nan check in bottleneck for zeroed-rim conv2_out, nancount = ", conv2_nan_count)
            bn2_nan_count = torch.isnan(bn2_out.view(-1)).sum()
            print("nan check in bottleneck for bn2_out, nancount = ", bn2_nan_count)
            c3w_nan_count = torch.isnan(c3w.view(-1)).sum()
            print("nan check in bottleneck for c3w, nancount = ", c3w_nan_count)
            conv3_nan_count = torch.isnan(conv3_out.view(-1)).sum()
            print("nan check in bottleneck for conv3_out, nancount = ", conv3_nan_count)
            #print("nan check in bottleneck for residual, nancount = ", torch.isnan(residual.view(-1)).sum())
            if config.has_residual_conv:
                c4w_nan_count = torch.isnan(c4w.view(-1)).sum()
                print("nan check in bottleneck for c4w, nancount = ", c4w_nan_count)
            else:
                c42_nan_count = 0
            conv4_nan_count = torch.isnan(conv4_out.view(-1)).sum()
            print("nan check in bottleneck for conv4_out, nancount = ", conv4_nan_count)
            if config.has_residual_conv:
                bn4_nan_count = torch.isnan(bn4_out.view(-1)).sum()
                print("nan check in bottleneck for bn4_out, nancount = ", bn4_nan_count)
            else:
                bn4_nan_count = 0
            bn3_nan_count = torch.isnan(bn3_out.view(-1)).sum()
            print("nan check in bottleneck for bn3_out, nancount = ", bn3_nan_count)
            output_nan_count = torch.isnan(output.view(-1)).sum()
            print("nan check in bottleneck for output, nancount = ", torch.isnan(output.view(-1)).sum())
            if conv1_nan_count > 0 or conv2_nan_count > 0 or conv3_nan_count > 0 or conv4_nan_count > 0 or bn1_nan_count > 0 or bn2_nan_count > 0 or bn3_nan_count > 0 or bn4_nan_count > 0 or c1w_nan_count > 0 or c2w_nan_count > 0 or c3w_nan_count > 0 or c4w_nan_count > 0:
                print("Exiting after doing fwd because nan count is not zero")
                exit(-1)
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

        ctx.tuning_params_d    = tuning_params_d
        ctx.tuning_strings_d   = tuning_strings_d
        ctx.tuning_params_w    = tuning_params_w
        ctx.tuning_strings_w   = tuning_strings_w
        ctx.tuning_timings_bwd = tuning_timings_bwd

        #print("dbg: in fwd tuning_params_d, tuning_strings_d = ", tuning_params_d, tuning_strings_d)

        ctx.save_for_backward(input, c1w, c2w, c3w, c4w, b1w, b2w, b3w, b4w, b1b, b2b, b3b, b4b, b1m, b2m, b3m, b4m, b1n, b2n, b3n, b4n,
                              conv1_out, bn1_out, conv2_out, bn2_out, conv3_out, bn3_out, conv4_out, bn4_out,
                              bn1_relu_out, bn2_relu_out, bn3_relu_out, bn4_relu_out,
                              b1s_out, b2s_out, b3s_out, b4s_out)
                              #c1s, c2s, c3s, c4s, b1s_out, b2s_out, b3s_out, b4s_out) # FIXME, must be cNs_out!

        time_btlnk_fwd = time.time() - time_start

        #print("time_btlnk_fwd (C K H W stride) = ", config.inplanes, config.planes, config.H, config.W, config.stride, time_btlnk_fwd)

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

        tuning_params_d    = ctx.tuning_params_d
        tuning_strings_d   = ctx.tuning_strings_d
        tuning_params_w    = ctx.tuning_params_w
        tuning_strings_w   = ctx.tuning_strings_w
        tuning_timings_bwd = ctx.tuning_timings_bwd

        #print("dbg: in bwd tuning_params_d, tuning_strings_d = ", tuning_params_d, tuning_strings_d)
        #print("dbg: in bwd tuning_params_w, tuning_strings_w = ", tuning_params_w, tuning_strings_w)

        #for entity in inputs:
        #    print("type of entity = ", type(entity))

        if (tuning_params_d is None or tuning_strings_d is None or len(tuning_params_d) == 0 or len(tuning_strings_d) == 0):
            if (tuning_params_w is None or tuning_strings_w is None or len(tuning_params_w) == 0 or len(tuning_strings_w) == 0):
                if tuning_timings_bwd is None:
                    (grad_c1w, grad_c2w, grad_c3w, grad_c4w,
                     grad_b1w, grad_b2w, grad_b3w, grad_b4w,
                     grad_b1b, grad_b2b, grad_b3b, grad_b4b,
                     grad_c1i, grad_c4i) = bottleneck_cpp.bottleneck_bn_bwd(config, inputs) #_tensors)
                else:
                    print("Unsupported mode with tuning params for both w and d empty but non-empty tuning_timings")
            else:
                if tuning_timings_bwd is None:
                    tuning_timings_bwd = np.zeros(16, dtype=np.float32)
                (grad_c1w, grad_c2w, grad_c3w, grad_c4w,
                 grad_b1w, grad_b2w, grad_b3w, grad_b4w,
                 grad_b1b, grad_b2b, grad_b3b, grad_b4b,
                 grad_c1i, grad_c4i) = bottleneck_cpp.bottleneck_bn_bwd_defaultd_ext(config, inputs, tuning_params_w, tuning_strings_w, tuning_timings_bwd)
        else:
            if tuning_timings_bwd is None:
                tuning_timings_bwd = np.zeros(16, dtype=np.float32)
            if (tuning_params_w is None or tuning_strings_w is None or len(tuning_params_w) == 0 or len(tuning_strings_w) == 0):
                (grad_c1w, grad_c2w, grad_c3w, grad_c4w,
                 grad_b1w, grad_b2w, grad_b3w, grad_b4w,
                 grad_b1b, grad_b2b, grad_b3b, grad_b4b,
                 grad_c1i, grad_c4i) = bottleneck_cpp.bottleneck_bn_bwd_defaultw_ext(config, inputs, tuning_params_d, tuning_strings_d, tuning_timings_bwd)
            else:
                (grad_c1w, grad_c2w, grad_c3w, grad_c4w,
                 grad_b1w, grad_b2w, grad_b3w, grad_b4w,
                 grad_b1b, grad_b2b, grad_b3b, grad_b4b,
                 grad_c1i, grad_c4i) = bottleneck_cpp.bottleneck_bn_bwd_ext(config, inputs, tuning_params_d, tuning_strings_d, tuning_params_w, tuning_strings_w, tuning_timings_bwd)

        #print("dbg: bottleneck_backward_new called")

        grad_input = grad_c1i + grad_c4i

        """
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if rank == 0:
            grad_c1w_nan_count = torch.isnan(grad_c1w.view(-1)).sum()
            print("nan check in bottleneck for grad_c1w, nancount = ", grad_c1w_nan_count)
            grad_b1w_nan_count = torch.isnan(grad_b1w.view(-1)).sum()
            print("nan check in bottleneck for grad_b1w_out, nancount = ", grad_b1w_nan_count)
            grad_c2w_nan_count = torch.isnan(grad_c2w.view(-1)).sum()
            print("nan check in bottleneck for grad_c2w, nancount = ", grad_c2w_nan_count)
            grad_b2w_nan_count = torch.isnan(grad_b2w.view(-1)).sum()
            print("nan check in bottleneck for grad_b2w_out, nancount = ", grad_b2w_nan_count)
            grad_c3w_nan_count = torch.isnan(grad_c3w.view(-1)).sum()
            print("nan check in bottleneck for grad_c3w, nancount = ", grad_c3w_nan_count)
            if ctx.config.has_residual_conv == 0:
                grad_c4w_nan_count = torch.isnan(grad_c4w.view(-1)).sum()
                print("nan check in bottleneck for grad_c4w, nancount = ", grad_c4w_nan_count)
                grad_b4w_nan_count = torch.isnan(grad_b4w.view(-1)).sum()
                print("nan check in bottleneck for grad_b4w_out, nancount = ", grad_b4w_nan_count)
            else:
                grad_c4w_nan_count = 0
                grad_b4w_nan_count = 0
            grad_b3w_nan_count = torch.isnan(grad_b3w.view(-1)).sum()
            print("nan check in bottleneck for grad_b3w_out, nancount = ", grad_b3w_nan_count)
            if grad_b1w_nan_count > 0 or grad_b2w_nan_count > 0 or grad_b3w_nan_count > 0 or grad_b4w_nan_count > 0 or grad_c1w_nan_count > 0 or grad_c2w_nan_count > 0 or grad_c3w_nan_count > 0 or grad_c4w_nan_count > 0:
                print("Exiting because nan count is not zero")
                exit(-1)
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


        return (None, None,       # for handle and training arguments in forward
                None, None, None, # for tuning_params, tuning_strings and tuning_timings_fwd
                None, None, None, None, None, # for tuning_params_d, tuning_strings_d, tuning_params_w, tuning_strings_w and tuning_timings_bwd
                grad_input,
                grad_c1w, grad_c2w, grad_c3w, grad_c4w,
                grad_b1w, grad_b2w, grad_b3w, grad_b4w,
                grad_b1b, grad_b2b, grad_b3b, grad_b4b,
                None,     None,     None,     None, # for means
                None,     None,     None,     None,  # for vars
                None, None) # for tuning_params, tuning_strings
                #None,     None,     None,     None, # for conv scratches
                #None,     None,     None,     None) # for bn   scratches


# Generic monolithic bottleneck class for batchnorm/groupnorm bottleneck (with a control flag for the norm in the constructor and if-switches)
class BottleneckTPP(BlockedModule, Bottleneck_base):

    def __init__(self, inplanes, planes, eps, stride=1, use_physical_3x3_padding=False, downsample1=None, downsample2=None, use_groupnorm=False, dtype=torch.float,
                 bc_conv1=None, bc_conv2=None, bc_conv3=None, bk_conv3=None, avoid_fmas_in_rim=None, use_hardcoded_tunings=False):
        super(BottleneckTPP, self).__init__(inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm=use_groupnorm, dtype=dtype,
                                            bc_conv1=bc_conv1, bc_conv2=bc_conv2, bc_conv3=bc_conv3, bk_conv3=bk_conv3, avoid_fmas_in_rim=avoid_fmas_in_rim)

        print("debug: BottleneckTPP constructor called with inplanes, planes, eps, stride, downsample1, downsample2 use_groupnorm dtype bc_conv1 bc_conv2 bc_conv3 bk_conv3 avoid_fmas_in_rim use_hardcoded_tunings = ",
                  inplanes, planes, eps, stride, downsample1, downsample2, use_groupnorm, dtype, bc_conv1, bc_conv2, bc_conv3, bk_conv3, avoid_fmas_in_rim, use_hardcoded_tunings)

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

        #self.tuning_params  = None
        #self.tuning_strings = None
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

        self.use_hardcoded_tunings = use_hardcoded_tunings

        self.tuning_params_fwd  = None
        self.tuning_strings_fwd = None
        self.tuning_params_d    = None
        self.tuning_strings_d   = None
        self.tuning_params_w    = None
        self.tuning_strings_w   = None

        # hardcoded for 56 threads on SPR
        if self.use_hardcoded_tunings:
            self.hybrid_cols = 14
            self.hybrid_rows = 4
            if self.use_bf16 == True:
                # bwd_d tunings are based on results in bottleneck_*_tuning_bwd_d_not1_0721.txt
                # bwd_w tunings are based on results in bottleneck_*_tuning_bwd_w_nohybrid_not1_0721.txt
                if self.inplanes == 64 and self.planes == 64: # Bottleneck type #0
                    self.tuning_params_fwd = [4, 1, 4, 1, 4, 1, 4, 1, # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1, # c,k blocks
                                              1, 1, 1, 1, # h_in_gemms
                                              0, 0 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbdced', 'Afgbdced', 'Afgbdced', 'Afgbdced'] #['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_d = [7, 1, 7, 1, 7, 1, 7, 1, # h,w blocks
                                            1, 1, 1, 1, 1, 1, 1, 1, # c,k blocks
                                            1, 1, 1, 1] # h_in_gemms
                    #self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_d = ['Afgcbded', 'Afgcbded', 'Afgcbded', 'Afgcbded']
                    self.tuning_params_w = [1, 0, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            1, 0, 0, 0, 0, 1,  1, 56, 1,  0, 0, 1,
                                            1, 0, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            1, 0, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1]
                    #self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                    self.tuning_strings_w = ['Aefcbd', 'A{R:56}C{C:1}dbef', 'Aefcbd', 'Aefcbd']
                elif self.inplanes == 256 and self.planes == 64:  # Bottleneck type #1
                    self.tuning_params_fwd = [4, 1, 4, 1, 4, 1, 4, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              1, 1, 1, 1, # h_in_gemms
                                              0, 0 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbdced', 'Afgbdced', 'Afgbdced', 'Afgbdced'] #['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    #self.tuning_strings_fwd = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                            1, 1, 1, 1] # h_in_gemms
                    #self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_d = ['Afgcbde', 'Afgcbde', 'Afgcbde', 'Afgcbde']
                    #self.tuning_params_w = [1, 1, 1, 1, # p blocks
                    #                        1, 1, 1, 1, # use nchw formats
                    #                        0, 0, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                    #                        0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                    #                        0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_params_w = [1, 0, 0, 0, 0, 0,  0, 1, 1,  1, 0, 1,
                                            1, 0, 0, 0, 0, 0,  0, 1, 1,  1, 0, 1,
                                            1, 0, 0, 0, 0, 0,  0, 1, 1,  1, 0, 1,
                                            1, 0, 0, 0, 0, 0,  0, 1, 1,  1, 0, 1] # last row is a dummy (no c4)
                    #self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                    self.tuning_strings_w = ['Aefdbc', 'Acbdef', 'Aefdbc', 'Aefdbc']
                elif self.inplanes == 256 and self.planes == 128:  # Bottleneck type #2
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              1, 1, 1, 1, # h_in_gemms
                                              0, 0 ] # pack_input, fuse_stats
                    #self.tuning_strings_fwd = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_fwd = ['Afgbedc', 'Afgbedc', 'Afgbedc', 'Afgbedc'] #['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                            1, 1, 1, 1] # h_in_gemms
                    #self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_d = ['Afgcbde', 'Afgcbde', 'Afgcbde', 'Afgcbde']
                    #self.tuning_params_w = [1, 1, 1, 1, # p blocks
                    #                        1, 0, 1, 1, # use nchw formats
                    #                        1, 0, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                    #                        1, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                    #                        0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_params_w = [1, 0, 1, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            0, 0, 1, 1, 1, 0,  0, 4, 14,  0, 0, 1,
                                            1, 0, 1, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            1, 0, 1, 0, 1, 1,  0, 1, 1,   1, 0, 1]
                    #self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                    #self.tuning_strings_w = ['Aefcdb', 'abEFdc', 'Aefcdb', 'Aefcdb']
                    self.tuning_strings_w = ['Aefcdb', 'A{R:4}C{C:14}bdef', 'Aefcdb', 'Aefcdb']
                elif self.inplanes == 512 and self.planes == 128:  # Bottleneck type #3
                    self.tuning_params_fwd = [7, 1, 7, 1, 7, 1, 7, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              1, 1, 1, 1, # h_in_gemms
                                              0, 0 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbdecd', 'Afgbdecd', 'Afgbdecd', 'Afgbdecd']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 2, 1, 1, 1 , # c,k blocks
                                            1, 1, 1, 1] # h_in_gemms
                    #self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_d = ['Afgcdeb', 'Afgcdeb', 'Afgcdeb', 'Afgcdeb']
                    #self.tuning_params_w = [1, 1, 1, 1, # p blocks
                    #                        1, 1, 1, 1, # use nchw formats
                    #                        0, 1, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                    #                        0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                    #                        0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_params_w = [1, 1, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            1, 0, 0, 0, 0, 1,  1, 56, 1,  0, 0, 1,
                                            1, 1, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            1, 0, 1, 0, 1, 1,  0, 1, 1,   1, 0, 1] # last row is a dummy (no c4)
                    #self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                    #self.tuning_strings_w = ['Aefcbd', 'Acdbef', 'Aefcbd', 'Aefcbd']
                    self.tuning_strings_w = ['Aefcbd', 'A{R:56}C{C:1}dbef', 'Aefcbd', 'Aefcbd']
                elif self.inplanes == 512 and self.planes == 256:  # Bottleneck type #4
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 8, 1, 8 , # c,k blocks
                                              1, 1, 2, 2, # h_in_gemms
                                              1, 0 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbcdce', 'Afgbcdce', 'Afgbcdce', 'Afgbcdce' ]
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 4, 1, 4, 1 , # c,k blocks
                                            1, 1, 2, 1] # h_in_gemms
                    #self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_d = ['Afgcebd', 'Afgcebd', 'Afgcebd', 'Afgcebd']
                    #self.tuning_params_w = [1, 1, 1, 1, # p blocks
                    #                        1, 0, 1, 1, # use nchw formats
                    #                        1, 1, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                    #                        0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                    #                        0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_params_w = [1, 1, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            0, 0, 1, 0, 1, 0,  0, 1, 1,   0, 0, 1,
                                            1, 1, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            1, 0, 0, 0, 1, 1,  0, 1, 1,   1, 0, 1]
                    #self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                    #self.tuning_strings_w = ['Aefbcd', 'abEfcd', 'Aefbcd', 'Aefbcd']
                    self.tuning_strings_w = ['Aefbcd', 'cAEBfd', 'Aefbcd', 'Aefbcd']
                elif self.inplanes == 1024 and self.planes == 256:  # Bottleneck type #5
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 2, 1, 1, 1, 2, 1, 1 , # c,k blocks
                                              2, 2, 2, 2, # h_in_gemms
                                              0, 0 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbcecd', 'Afgbcecd', 'Afgbcecd', 'Afgbcecd']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            8, 1, 1, 1, 8, 1, 1, 1 , # c,k blocks
                                            2, 2, 2, 2] # h_in_gemms
                    #self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_d = ['Afgcbdeb', 'Afgcbde', 'Afgcbde', 'Afgcbde']
                    #self.tuning_params_w = [1, 1, 1, 1, # p blocks
                    #                        1, 1, 1, 1, # use nchw formats
                    #                        0, 1, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                    #                        0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                    #                        0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_params_w = [1, 1, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            1, 0, 0, 0, 0, 1,  1, 56, 1,  0, 0, 1,
                                            1, 1, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            1, 1, 0, 0, 0, 0,  0, 1, 1,   1, 0, 1]  # last row is a dummy (no c4)
                    #self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                    self.tuning_strings_w = ['Aefdcb', 'A{R:56}C{C:1}dbef', 'Aefdcb', 'Aefdcb']
                elif self.inplanes == 1024 and self.planes == 512:  # Bottleneck type #6
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 8, 1, 4, 1, 8, 1, 4 , # c,k blocks
                                              2, 1, 7, 7, # h_in_gemms
                                              1, 0 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbcecd', 'Afgbcecd', 'Afgbcecd', 'Afgbcecd']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            4, 1, 8, 1, 8, 1, 16, 1 , # c,k blocks
                                            2, 1, 7, 1] # h_in_gemms
                    #self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_d = ['Afgcbde', 'Afgcbde', 'Afgcbde', 'Afgcbde']
                    #self.tuning_params_w = [1, 1, 1, 1, # p blocks
                    #                        1, 0, 1, 1, # use nchw formats
                    #                        1, 0, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                    #                        1, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                    #                        0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_params_w = [1, 0, 1, 0, 0, 0,  0, 1, 1,   1, 0, 1,
                                            0, 0, 1, 0, 1, 0,  0, 1, 1,   1, 0, 1,
                                            1, 0, 0, 0, 0, 1,  1, 14, 4,  0, 0, 1,
                                            1, 0, 0, 0, 1, 1,  1, 8, 7,   1, 0, 1]
                    #self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                    self.tuning_strings_w = ['Aefdbc', 'ABEFcd', 'A{R:14}C{C:4}dbef', 'A{R:8}C{C:7}dbef']
                elif self.inplanes == 2048 and self.planes == 512:  # Bottleneck type #7
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              7, 7, 7, 7, # h_in_gemms
                                              0, 0 ] # pack_input, fuse_stats
                    #self.tuning_strings_fwd = ['ACfgbdec', 'ACfgbdec', 'ACfgbdec', 'ACfgbdec']
                    self.tuning_strings_fwd = ['A{C:' + str(self.hybrid_cols) + '}C{R:' + str(self.hybrid_rows) +'}fgbde',
                                               'A{C:' + str(self.hybrid_cols) + '}C{R:' + str(self.hybrid_rows) +'}fgbde',
                                               'A{C:' + str(self.hybrid_cols) + '}C{R:' + str(self.hybrid_rows) +'}fgbde',
                                               'A{C:' + str(self.hybrid_cols) + '}C{R:' + str(self.hybrid_rows) +'}fgbde'] # last one is a dummy
                    #self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                    #                        4, 1, 4, 1, 4, 1, 1, 1 , # c,k blocks
                    #                          7, 1, 7, 7] # h_in_gemms
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              7, 7, 7, 7] # h_in_gemms
                    #self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    #self.tuning_strings_d = ['BAfgcedb', 'BAfgcedb', 'BAfgcedb', 'BAfgcedb']
                    self.tuning_strings_d = ['A{C:' + str(self.hybrid_cols) + '}B{R:' + str(self.hybrid_rows) +'}fgcbde',
                                               'A{C:' + str(self.hybrid_cols) + '}B{R:' + str(self.hybrid_rows) +'}fgcbde',
                                               'A{C:' + str(self.hybrid_cols) + '}B{R:' + str(self.hybrid_rows) +'}fgcbde',
                                               'A{C:' + str(self.hybrid_cols) + '}B{R:' + str(self.hybrid_rows) +'}fgcbde'] # last one is dummy
                    #self.tuning_params_w = [1, 1, 1, 1, # p blocks
                    #                        1, 0, 1, 1, # use nchw formats
                    #                        0, 0, 0, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                    #                        0, 0, 1, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                    #                        1, 14, 4] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_params_w = [1, 0, 0, 0, 0, 1,  1, 14, 4,  0, 0, 1,
                                            0, 0, 0, 0, 0, 0,  0, 1, 1,   0, 0, 1,
                                            1, 0, 0, 0, 0, 1,  1, 14, 4,  0, 0, 1,
                                            1, 0, 0, 0, 1, 1,  0, 1, 1,   1, 0, 1] # last row is a dummy (no c4)
                    #self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                    #self.tuning_strings_w = ['Aefcdb', 'Adbcef', 'Aefcdb', 'Aefcdb']
                    self.tuning_strings_w = ['A{R:14}C{C:4}dbef',
                                              'ABEFcd',
                                              'A{R:14}C{C:4}dbef',
                                              'Aefcdb']
            else: # fp32
                # fwd tunings are based on results in bottleneck_*_tuning_fp32_dbg.txt
                # bwd tunings are defaults
                if self.inplanes == 64 and self.planes == 64: # Bottleneck type #0
                    self.tuning_params_fwd = [4, 1, 4, 1, 4, 1, 4, 1, # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1, # c,k blocks
                                              1, 1, 1, 1, # h_in_gemms
                                              0, 0 ] # pack_input, fuse_stats
                    #self.tuning_strings_fwd = ['Afgbcded', 'Afgbcded', 'Afgbcded', 'Afgbcded'] #['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    #self.tuning_params_fwd = [4, 1, 4, 1, 1, 2, 1, 2, # h,w blocks
                    #                          1, 1, 1, 1, 1, 1, 1, 1, # c,k blocks
                    #                          1, 1, 1, 1, # h_in_gemms
                    #                          0, 0 ] # pack_input, fuse_stats
                    #self.tuning_strings_fwd = ['Afgbcded', 'Afgbcded', 'Afgcbde', 'Afgcbde'] #['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1, # h,w blocks
                                            1, 1, 1, 1, 1, 1, 1, 1, # c,k blocks
                                            1, 1, 1, 1] # h_in_gemms
                    self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_w = [1, 1, 1, 1, # p blocks
                                            1, 1, 1, 1, # use nchw formats
                                            0, 0, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                                            0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                elif self.inplanes == 256 and self.planes == 64:  # Bottleneck type #1
                    self.tuning_params_fwd = [7, 1, 7, 1, 7, 1, 7, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              1, 1, 1, 1, # h_in_gemms
                                              0, 0 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbdecd', 'Afgbdecd', 'Afgbdecd', 'Afgbdecd'] #['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    #self.tuning_params_fwd = [1, 2, 7, 1, 1, 2, 1, 2 , # h,w blocks
                    #                          1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                    #                          1, 1, 1, 1, # h_in_gemms
                    #                          0, 0 ] # pack_input, fuse_stats
                    #self.tuning_strings_fwd = ['Afgcbde', 'Afgbdecd', 'Afgcbde', 'Afgcbde'] #['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    #self.tuning_strings_fwd = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                            1, 1, 1, 1] # h_in_gemms
                    self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_w = [1, 1, 1, 1, # p blocks
                                            1, 1, 1, 1, # use nchw formats
                                            0, 0, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                                            0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                elif self.inplanes == 256 and self.planes == 128:  # Bottleneck type #2
                    self.tuning_params_fwd = [7, 1, 4, 1, 4, 1, 4, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              1, 1, 1, 1, # h_in_gemms
                                              0, 1 ] # pack_input, fuse_stats
                    #self.tuning_strings_fwd = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_strings_fwd = ['Afgbdced', 'Afgbdced', 'Afgbdced', 'Afgbdced'] #['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                            1, 1, 1, 1] # h_in_gemms
                    self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_w = [1, 1, 1, 1, # p blocks
                                            1, 0, 1, 1, # use nchw formats
                                            1, 0, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                                            1, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                elif self.inplanes == 512 and self.planes == 128:  # Bottleneck type #3
                    self.tuning_params_fwd = [7, 1, 7, 1, 7, 1, 7, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 2, 1, 1 , # c,k blocks
                                              1, 1, 1, 1, # h_in_gemms
                                              0, 1 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbcdced', 'Afgbcdced', 'Afgbcdced', 'Afgbcdced']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 2, 1, 1, 1 , # c,k blocks
                                            1, 1, 1, 1] # h_in_gemms
                    self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_w = [1, 1, 1, 1, # p blocks
                                            1, 1, 1, 1, # use nchw formats
                                            0, 1, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                                            0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                elif self.inplanes == 512 and self.planes == 256:  # Bottleneck type #4
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              1, 1, 2, 2, # h_in_gemms
                                              1, 1 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbcde', 'Afgbcde', 'Afgbcde', 'Afgbcde' ]
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            1, 1, 1, 1, 4, 1, 4, 1 , # c,k blocks
                                            1, 1, 2, 1] # h_in_gemms
                    self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_w = [1, 1, 1, 1, # p blocks
                                            1, 0, 1, 1, # use nchw formats
                                            1, 1, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                                            0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                elif self.inplanes == 1024 and self.planes == 256:  # Bottleneck type #5
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 4, 1, 1, 1, 4, 1, 1 , # c,k blocks
                                              2, 1, 2, 2, # h_in_gemms # ! fixed h_in_gemm for 3x3 manually (was 2)
                                              0, 1 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbcdec', 'Afgbcdec', 'Afgbcdec', 'Afgbcdec']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            8, 1, 1, 1, 8, 1, 1, 1 , # c,k blocks
                                            2, 2, 2, 2] # h_in_gemms
                    self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_w = [1, 1, 1, 1, # p blocks
                                            1, 1, 1, 1, # use nchw formats
                                            0, 1, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                                            0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                elif self.inplanes == 1024 and self.planes == 512:  # Bottleneck type #6
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 8, 1, 8, 1, 8, 1, 16 , # c,k blocks
                                              2, 1, 7, 7, # h_in_gemms
                                              1, 0 ] # pack_input, fuse_stats
                    self.tuning_strings_fwd = ['Afgbcecd', 'Afgbcecd', 'Afgbcecd', 'Afgbcecd']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            4, 1, 8, 1, 8, 1, 16, 1 , # c,k blocks
                                            2, 1, 7, 1] # h_in_gemms
                    self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_w = [1, 1, 1, 1, # p blocks
                                            1, 0, 1, 1, # use nchw formats
                                            1, 0, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                                            1, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']
                elif self.inplanes == 2048 and self.planes == 512:  # Bottleneck type #7
                    self.tuning_params_fwd = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                              1, 1, 1, 1, 1, 1, 1, 1 , # c,k blocks
                                              7, 1, 7, 7, # h_in_gemms # ! fixed h_in_gemm for 3x3 manually (was 7)
                                              0, 0 ] # pack_input, fuse_stats
                    #self.tuning_strings_fwd = ['ACfgbced', 'ACfgbced', 'ACfgbced', 'ACfgbced']
                    self.tuning_strings_fwd = ['A{C:' + str(self.hybrid_cols) + '}C{R:' + str(self.hybrid_rows) +'}fgbcde',
                                               'A{C:' + str(self.hybrid_cols) + '}C{R:' + str(self.hybrid_rows) +'}fgbcde',
                                               'A{C:' + str(self.hybrid_cols) + '}C{R:' + str(self.hybrid_rows) +'}fgbcde',
                                               'A{C:' + str(self.hybrid_cols) + '}C{R:' + str(self.hybrid_rows) +'}fgbcde']
                    self.tuning_params_d = [1, 1, 1, 1, 1, 1, 1, 1 , # h,w blocks
                                            4, 1, 4, 1, 4, 1, 1, 1 , # c,k blocks
                                            7, 1, 7, 7] # h_in_gemms
                    self.tuning_strings_d = ['Abcdefg', 'Abcdefg', 'Abcdefg', 'Abcdefg']
                    self.tuning_params_w = [1, 1, 1, 1, # p blocks
                                            1, 1, 1, 1, # use nchw formats
                                            0, 0, 1, # pack_input_upfront, fuse_upd_transposes, use_f32_wt_reduction_and_external_wt_vnni
                                            0, 0, 0, # acc_nw, par_over_h_pixels, compute_full_wt_output_block
                                            0, 1, 1] #hybrid, n_img_teams, n_ofm_teams
                    self.tuning_strings_w = ['Aefbcd', 'Aefbcd', 'Aefbcd', 'Aefbcd']



    def maybe_block(self):
        for m in self.modules():
            if hasattr(m, "maybe_block_params"):
                m.maybe_block_params()

    #def set_tuning_params(self, tuning_params):
    #    self.tuning_params = tuning_params
    #def set_tuning_strings(self, tuning_strings):
    #    self.tuning_strings = tuning_strings

    def forward(self, input, tuning_params_fwd=None, tuning_strings_fwd=None, tuning_timings_fwd=None, tuning_params_d=None, tuning_strings_d=None, tuning_params_w=None, tuning_strings_w=None, tuning_timings_bwd=None):

        #print("bn1.running_mean at the start of forward = ", self.bn1.running_mean)
        """
        rank = int(os.environ.get("PMI_RANK", -1))
        if rank < 0:
            rank = 0
        if rank == 0:
            print("debug: BottleneckTPP forward is called with inplanes, planes, stride, downsample1, downsample2 use_groupnorm dtype bc_conv1 bc_conv2 bc_conv3 bk_conv3 = ",
                  self.inplanes, self.planes, self.stride, self.downsample1, self.downsample2, self.use_groupnorm, self.dtype, self.bc_conv1, self.bc_conv2, self.bc_conv3, self.bk_conv3)
        """

        #print("in btlnk forward(), use_hardcoded_tunings, self.tuning_params_fwd tuning_params_fwd = ", self.use_hardcoded_tunings, self.tuning_params_fwd, tuning_params)

        l_tuning_params_fwd  = tuning_params_fwd if tuning_params_fwd is not None else self.tuning_params_fwd
        l_tuning_strings_fwd = tuning_strings_fwd if tuning_strings_fwd is not None else self.tuning_strings_fwd
        l_tuning_params_d    = tuning_params_d if tuning_params_d is not None else self.tuning_params_d
        l_tuning_strings_d   = tuning_strings_d if tuning_strings_d is not None else self.tuning_strings_d
        l_tuning_params_w    = tuning_params_w if tuning_params_w is not None else self.tuning_params_w
        l_tuning_strings_w   = tuning_strings_w if tuning_strings_w is not None else self.tuning_strings_w

        #print("in btlnk forward(), l_tuning_params_fwd = ", l_tuning_params_fwd)

        self.maybe_block()

        N = input.size(0)
        self.H = input.size(2)
        self.W = input.size(3)

        if self.config is None:
            if self.use_groupnorm:
                print("use_groupnorm not implemented in the bottleneck in extensions")
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
                      print("BottleneckTPP Create BN-powered handle: ", N, self.inplanes, self.H, self.W, self.planes, self.stride, self.norm_eps, self.bn_momentum, self.bn_track_running_stats, self.dtype, self.preset_blocksizes)
                if self.preset_blocksizes:
                    self.config = bottleneck_cpp.bottleneck_bn_setup_fused_fwd_tuner(N, self.inplanes, self.H, self.W, self.planes, self.stride, self.norm_eps, self.bn_momentum, self.bn_track_running_stats, self.expansion,
                                                                 1 if self.use_physical_3x3_padding else 0, 0 if self.dtype == torch.float else 1,
                                                                 self.bc_conv1, self.bc_conv2, self.bc_conv3, self.bk_conv3, 1 if self.avoid_fmas_in_rim else 0)
                else:
                    self.config = bottleneck_cpp.bottleneck_bn_setup(N, self.inplanes, self.H, self.W, self.planes, self.stride, self.norm_eps, self.bn_momentum, self.bn_track_running_stats, self.expansion,
                                                                 1 if self.use_physical_3x3_padding else 0, 0 if self.dtype == torch.float else 1)
                #self.xsmm_handle = BottleneckBNHandleTPP(N, self.inplanes, self.H, self.W, self.planes, self.stride,
                #                                         self.norm_eps, self.bn_momentum, 1 if self.bn_track_running_stats else 0,
                #                                         self.expansion, self.use_physical_3x3_padding, self.dtype)

            self.N = N
            #print("dbg: created the handle in BottleneckBNTPP")

        """
        [conv1_Cblock, conv1_Kblock, conv1_lpblock] = conv_cpp.conv_get_feature_map_blocks(self.conv1.in_channels, self.conv1.out_channels, 0 if self.dtype == torch.float else 1)
        self.conv1_Cblock = conv1_Cblock # used in the early version of the fwd tuner test
        blocked_input = self.get_blocked_tensor(
            input,
            self.blocked_input_signature,
            [None, conv1_Cblock, None, None],
        )
        """
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
            #output = BottleneckApplyBNTPP.apply(self.config, self.training, *inputs, tuning_params=tuning_params, tuning_strings=tuning_strings)
            #output = BottleneckApplyBNTPP.apply(self.config, self.training, *inputs, tuning_params, tuning_strings)
            output = BottleneckApplyBNTPP.apply(self.config, self.training,
                                                l_tuning_params_fwd, l_tuning_strings_fwd, tuning_timings_fwd,
                                                l_tuning_params_d, l_tuning_strings_d, l_tuning_params_w, l_tuning_strings_w, tuning_timings_bwd, *inputs )

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

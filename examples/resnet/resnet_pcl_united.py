import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import pcl_cgbp

from pcl_cgbp import BottleneckTPP

import blocked_layout
from blocked_layout import BlockedTensor, BlockedParameter

# Required to use the bottleneck implementation from the extension repo
#import tpp_pytorch_extension_resnet
#from tpp_pytorch_extension_resnet import bottleneck
#import tpp_pytorch_extension_resnet
#from tpp_pytorch_extension_resnet import bottleneck
import tpp_pytorch_extension
from tpp_pytorch_extension.resnet import bottleneck

# for validate_fwd = True case (to do a check for the distributed training and get the rank)
import os

import numpy as np

# Note: only resnet50_bn/gn have been tested
__all__ = ['ResNet', 'resnet50_bn', 'resnet50_gn', 'resnet101_bn', 'resnet101_gn', 'resnet152_bn']

# Used in blocks for printing tensors inside the Bottleneck module (disabled by default)
#global_tensor_x_counter = 0
#global_resnet_forward_counter = 0
#global_block_forward_counter = 0

#global_time = 0

"""
def conv2d_init(m):
    assert isinstance(m, nn.Conv2d)
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))
"""

def gn_init(m, zero_init=False):
    #assert isinstance(m, nn.GroupNorm) or isinstance(m, pcl_cgbp.nn_GroupNorm)
    m.weight.data.fill_(0. if zero_init else 1.)
    m.bias.data.zero_()

class BottleneckBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, eps, stride=1, downsample1=None, downsample2=None, use_ref_conv=False, use_ref_norm=False, dtype=torch.float):
        #print("BottleneckBN constructor called with inplanes, planes, eps, stride, downsample1, downsample2, use_ref_conv, use_ref_norm, dtype = ", inplanes, planes, eps, stride, downsample1, downsample2, use_ref_conv, use_ref_norm, dtype)
        super(BottleneckBN, self).__init__()

        self.use_ref_conv = use_ref_conv
        self.use_ref_bn   = use_ref_norm
        self.dtype        = dtype

        # eltwise is accounted for in the forward()
        # but relu is created here for the PyTorch reference impl
        if self.use_ref_bn == True:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dtype=self.dtype)
        if self.use_ref_bn != True:
            self.bn1 = nn.BatchNorm2d(planes, eps, relu=True, dtype=self.dtype)
            #self.bn1  = nn.BatchNorm2d(planes, eps, dtype=self.dtype)
        else:
            self.bn1  = nn.BatchNorm2d(planes, eps, dtype=self.dtype)
            #self.bn1  = nn.BatchNorm2d(planes, eps, track_running_stats=False)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dtype=self.dtype)

        if self.use_ref_bn != True:
            self.bn2 = nn.BatchNorm2d(planes, eps, relu=True, dtype=self.dtype)
            #self.bn2 = nn.BatchNorm2d(planes, eps, dtype=self.dtype)
        else:
            self.bn2  = nn.BatchNorm2d(planes, eps, dtype=self.dtype)
            #self.bn2  = nn.BatchNorm2d(planes, eps, track_running_stats=False, dtype=self.dtype)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dtype=self.dtype)

        if self.use_ref_bn != True:
            self.bn3 = nn.BatchNorm2d(planes * 4, eps, relu=True, eltwise=True, dtype=self.dtype)
            #self.bn3  = nn.BatchNorm2d(planes * 4, eps, dtype=self.dtype)
        else:
            self.bn3  = nn.BatchNorm2d(planes * 4, eps, dtype=self.dtype)
            #self.bn3  = nn.BatchNorm2d(planes * 4, eps, track_running_stats=False, dtype=self.dtype)
        self.downsample1 = downsample1 # this is the conv part of downsampling;
        self.downsample2 = downsample2 # this is the bn   part of downsampling;
        self.stride = stride


    def forward(self, x):

        #global global_tensor_x_counter
        #global global_block_forward_counter

        #print("debug: in bottleneck forward x.shape = ", x.shape)
        residual = x

        """
        self.rank = int(os.environ.get("PMI_RANK", -1))
        if self.rank < 0:
            self.rank = 0
        if self.training:
            #self.dump_file_suffix    = '_train_sfx' + '_rank_' + str(self.rank)
            self.dump_file_suffix    = '_train_tst' + '_rank_' + str(self.rank)
        else:
            #self.dump_file_suffix    = '_eval_sfx' + '_rank_' + str(self.rank)
            self.dump_file_suffix    = '_eval_tst' + '_rank_' + str(self.rank)
        """

        #if type(self.conv1.weight) is BlockedParameter:
        #    self.conv1.weight.unblock()
        #np.savetxt('my_layer_conv1_forward_weight' + str(global_tensor_x_counter)  + self.dump_file_suffix + '.txt', self.conv1.weight.contiguous().view(-1).detach().to(torch.float).numpy())

        #tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
        #np.savetxt('my_layer_conv1_forward_input_x_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1


        out = self.conv1(x)

        #tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
        #np.savetxt('my_layer_conv1_forward_output_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1

        if self.use_ref_bn == True:
            out = self.bn1(out)
            out = self.relu(out)
        else:
            out = self.bn1(out)

        #tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
        #np.savetxt('my_layer_conv2_forward_input_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1

        #print("before conv2, out shape = ", out.shape)

        out = self.conv2(out)

        #print("after conv2, out shape = ", out.shape)

        #tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
        #np.savetxt('my_layer_conv2_forward_output_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1

        if self.use_ref_bn == True:
            out = self.bn2(out)
            out = self.relu(out)
        else:
            out = self.bn2(out)
        #out = self.bn2(out)

        #print("after bn2 (+relu), out shape = ", out.shape)
        #exit()

        #tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
        #np.savetxt('my_layer_conv3_forward_input_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1

        out = self.conv3(out)

        #tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
        #np.savetxt('my_layer_conv3_forward_output_out_' + str(global_tensor_x_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1

        if self.downsample1 is not None and self.downsample2 is not None:
            residual1 = self.downsample1(x)
            residual  = self.downsample2(residual1)

        if self.use_ref_bn == True:
            out = self.bn3(out)
            out += residual
            out = self.relu(out)
        else:
            out = self.bn3(out, residual)

        """
        if self.use_ref_bn == True or self.use_ref_conv == True:
            if self.downsample1 is not None and self.downsample2 is not None:
                residual1 = self.downsample1(x)
                residual  = self.downsample2(residual1)

            #np.savetxt('my_block_bn3_input_' + str(global_block_forward_counter) + self.dump_file_suffix + '.txt', out.contiguous().view(-1).detach().to(torch.float).numpy())
            #global_block_forward_counter = global_block_forward_counter + 1
            #np.savetxt('my_block_bn3_residual_' + str(global_block_forward_counter) + self.dump_file_suffix + '.txt', out.contiguous().view(-1).detach().to(torch.float).numpy())
            #global_block_forward_counter = global_block_forward_counter + 1

            if self.use_ref_bn == True:
                out = self.bn3(out)
                out += residual
                out = self.relu(out)
            else:
                out = self.bn3(out, residual)

            #np.savetxt('my_block_bn3_output_' + str(global_block_forward_counter) + self.dump_file_suffix + '.txt', out.contiguous().view(-1).detach().to(torch.float).numpy())
            #global_block_forward_counter = global_block_forward_counter + 1
        else:

            if self.downsample2 is not None:
                print("Error, when batchnorm and conv are optimized, downsample2 must be None!")
                exit()

            if self.downsample1 is not None:
                residual = self.downsample1(x)


            #if self.downsample1 is not None and self.downsample2 is not None:
            #    residual1 = self.downsample1(x)
            #    residual2 = self.downsample2(residual1)
            #    residual = residual2

            out = self.bn3(out, residual)
        """

        #tmp_tensor = out.unblocked_tensor() if type(out) is BlockedTensor else out
        #np.savetxt('my_layer_block_forward_output_out_' + str(global_block_forward_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #global_block_forward_counter = global_block_forward_counter + 1
        #exit()

        return out

class BottleneckGN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, eps, stride=1, downsample1=None, downsample2=None, use_ref_conv=False, use_ref_norm=False, dtype=torch.float):
        #print("BottleneckGN constructor called with inplanes, planes, eps, stride, downsample1, downsample2, use_ref_conv, use_ref_norm, dtype = ", inplanes, planes, eps, stride, downsample1, downsample2, use_ref_conv, use_ref_norm, dtype)
        super(BottleneckGN, self).__init__()

        self.use_ref_conv = use_ref_conv
        self.use_ref_gn   = use_ref_norm
        self.dtype        = dtype

        # eltwise is accounted for in the forward()
        # but relu is created here for the PyTorch reference impl
        if self.use_ref_gn == True:
            self.relu = nn.ReLU(inplace=False) # False just for simplicity, could be True

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, dtype=self.dtype)
        if self.use_ref_gn != True:
            self.bn1 = nn.GroupNorm(32, planes, eps, relu=True, dtype=self.dtype)
        else:
            self.bn1 = nn.GroupNorm(32, planes, eps, dtype=self.dtype)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, dtype=self.dtype)
        if self.use_ref_gn != True:
            self.bn2 = nn.GroupNorm(32, planes, eps, relu=True, dtype=self.dtype)
        else:
            self.bn2 = nn.GroupNorm(32, planes, eps, dtype=self.dtype)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, dtype=self.dtype)
        if self.use_ref_gn != True:
            self.bn3 = nn.GroupNorm(32, planes * 4, eps, relu=True, eltwise=True, dtype=self.dtype)
        else:
            self.bn3 = nn.GroupNorm(32, planes * 4, eps, dtype=self.dtype)
        self.downsample1 = downsample1 # this is the conv part of downsampling
        self.downsample2 = downsample2 # this is the gn   part of downsampling
        self.stride = stride

        gn_init(self.bn1)
        gn_init(self.bn2)
        gn_init(self.bn3, zero_init=True)

    def forward(self, x):

        residual = x

        #np.savetxt('my_conv1_forward_input_x_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', x.contiguous().view(-1).detach().numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1
        out = self.conv1(x)
        #np.savetxt('my_conv1_forward_output_out_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', out.contiguous().view(-1).detach().numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1

        if self.use_ref_gn == True:
            out = self.bn1(out)
            out = self.relu(out)
        else:
            out = self.bn1(out)

        #np.savetxt('my_conv2_forward_input_out_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', out.contiguous().view(-1).detach().numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1
        out = self.conv2(out)
        #np.savetxt('my_conv2_forward_output_out_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', out.contiguous().view(-1).detach().numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1

        if self.use_ref_gn == True:
            out = self.bn2(out)
            out = self.relu(out)
        else:
            out = self.bn2(out)

        #np.savetxt('my_conv3_forward_input_out_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', out.contiguous().view(-1).detach().numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1
        out = self.conv3(out)
        #np.savetxt('my_conv3_forward_output_out_' + str(global_tensor_x_counter) + dump_file_suffix + '.txt', out.contiguous().view(-1).detach().numpy())
        #global_tensor_x_counter = global_tensor_x_counter + 1

        if self.downsample1 is not None and self.downsample2 is not None:
            residual1 = self.downsample1(x)
            residual  = self.downsample2(residual1)

        if self.use_ref_gn == True:
            out = self.bn3(out)
            out += residual
            out = self.relu(out)
        else:
            out = self.bn3(out, residual)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, use_ref_conv=False, use_ref_norm=False, use_ref_pool=False, use_ref_fc=False,
                  validate_fwd=False, pad_input_for_bf16_ref=False,
                  use_bottleneck_tpp=False, use_physical_3x3_padding=False, use_groupnorm=False,
                  use_hardcoded_tunings=None, channel_block_size=None, inference_mode=False, dtype = torch.float ):
        print("ResNet constructor called with block layers num_classes use_ref_conv use_ref_norm use_ref_pool use_ref_fc validate_fwd pad_input... use_bottleneck_tpp use_3x3_physical use_groupnorm use_hardcoded_tunings inference_mode dtype = ",
               block, layers, num_classes, use_ref_conv, use_ref_norm, use_ref_pool, use_ref_fc,
                validate_fwd, pad_input_for_bf16_ref,
                use_bottleneck_tpp, use_physical_3x3_padding, use_groupnorm,
                use_hardcoded_tunings, inference_mode, dtype)

        self.inplanes = 64
        self.use_ref_conv = use_ref_conv
        #self.use_ref_bn   = use_ref_bn
        #self.use_ref_gn   = use_ref_gn
        self.use_ref_norm = use_ref_norm
        self.use_ref_pool = use_ref_pool
        self.use_ref_fc   = use_ref_fc
        self.validate_fwd = validate_fwd
        self.pad_input_for_bf16_ref   = pad_input_for_bf16_ref
        self.use_bottleneck_tpp       = use_bottleneck_tpp
        self.use_physical_3x3_padding = use_physical_3x3_padding
        self.use_groupnorm            = use_groupnorm
        self.use_hardcoded_tunings    = use_hardcoded_tunings
        self.channel_block_size       = channel_block_size
        self.inference_mode           = inference_mode
        self.dtype                    = dtype

        #self.eps = 1.e-05
        self.eps = 1.e-07
        super(ResNet, self).__init__()
        if self.validate_fwd:
            torch.manual_seed(0)

        if self.pad_input_for_bf16_ref and self.use_ref_conv: # and self.validate.fwd:
            print("Doing (optional but necessary for comparison) channel padding for the first layer for the reference PT conv")
            self.padded_nchannels = 3 + 1 # works for both 2 and 4 as VNNI blocking size
            self.conv1 = nn.Conv2d(self.padded_nchannels, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, dtype=self.dtype)
        elif self.use_ref_conv:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                   bias=False, dtype=self.dtype)
        else:
            # Notice bk = 64 (showed 2ms better performance in the standalone testing) while the bottlenecks will use 32
            if self.inference_mode:
                self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False, relu=True, dtype=self.dtype, bc=4, bk=self.channel_block_size, logical_padding=True, use_hardcoded_tunings=self.use_hardcoded_tunings)
            else:
                self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                       bias=False, dtype=self.dtype, bc=4, bk=self.channel_block_size, logical_padding=True, use_hardcoded_tunings=self.use_hardcoded_tunings)
                                       #bias=False, dtype=self.dtype, bc=4, bk=32, logical_padding=True, use_hardcoded_tunings=self.use_hardcoded_tunings)

        if self.use_groupnorm:
            if self.use_ref_norm != True:
                self.bn1 = nn.GroupNorm(32, 64, self.eps, relu=True, dtype=self.dtype)
                self.relu = None # used in the forward() for the inference
            else:
                self.bn1 = nn.GroupNorm(32, 64, self.eps, dtype=self.dtype)
                self.relu = nn.ReLU(inplace=True)
        else: # batchnorm
            if self.use_ref_norm != True:
                self.bn1 = nn.BatchNorm2d(64, self.eps, relu=True, dtype=self.dtype, bc=self.channel_block_size)
                #self.bn1  = nn.BatchNorm2d(64, self.eps)
                #self.relu = nn.ReLU(inplace=False)
                self.relu = None # used in the forward() for the inference
            else:
                self.bn1  = nn.BatchNorm2d(64, self.eps, dtype=self.dtype)
                #self.bn1  = nn.BatchNorm2d(64, self.eps, track_running_stats=False)
                self.relu = nn.ReLU(inplace=False)

        if self.use_ref_pool != True:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dtype=self.dtype, bc=self.channel_block_size)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if self.use_ref_pool != True:
            self.avgpool = nn.AvgPool2d(7, stride=1, dtype=self.dtype, bc=self.channel_block_size)
        else:
            self.avgpool = nn.AvgPool2d(7, stride=1)

        if self.use_ref_fc != True:
            self.fc = nn.Linear(512 * block.expansion, num_classes, bn=torch.get_num_threads(), bc=self.channel_block_size, bk=100, dtype=self.dtype)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes, dtype=self.dtype)

        if self.validate_fwd:
            self.tensor_dump_counter = 0
            self.fwd_counter         = 0

        if self.use_groupnorm:
            gn_init(self.bn1)
        else:
            for m in self.modules():
                if isinstance(m, BottleneckBN) or isinstance(m, BottleneckTPP):
                    print("INIT BN ", m)
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        #downsample = None
        downsample1 = None
        downsample2 = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.use_ref_conv:
                downsample1 = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, dtype=self.dtype)
            else:
                downsample1 = nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, dtype=self.dtype, bc=self.channel_block_size, bk=self.channel_block_size)
            if self.use_groupnorm:
                downsample2 = nn.GroupNorm(32, planes * block.expansion, self.eps, dtype=self.dtype)
                gn_init(downsample2)
            else:
                if self.use_ref_norm:
                    downsample2 = nn.BatchNorm2d(planes * block.expansion, self.eps, dtype=self.dtype)
                else:
                    downsample2 = nn.BatchNorm2d(planes * block.expansion, self.eps, dtype=self.dtype, bc=self.channel_block_size)

        #print("in _make_layer downsample1, downsample2 = ", downsample1, downsample2)

        layers = []
        if self.use_bottleneck_tpp:
            if self.use_hardcoded_tunings is not None:
                layers.append(block(self.inplanes, planes, self.eps, stride, use_physical_3x3_padding=self.use_physical_3x3_padding,
                                    downsample1=downsample1, downsample2=downsample2, use_groupnorm = self.use_groupnorm, use_hardcoded_tunings = self.use_hardcoded_tunings,
                                    bc_conv1=self.channel_block_size, bc_conv2=self.channel_block_size, bc_conv3=self.channel_block_size, bk_conv3=self.channel_block_size,
                                    dtype=self.dtype))
            else:
                layers.append(block(self.inplanes, planes, self.eps, stride, use_physical_3x3_padding=self.use_physical_3x3_padding,
                                    downsample1=downsample1, downsample2=downsample2, use_groupnorm = self.use_groupnorm,
                                    bc_conv1=self.channel_block_size, bc_conv2=self.channel_block_size, bc_conv3=self.channel_block_size, bk_conv3=self.channel_block_size,
                                    dtype=self.dtype))
        else:
            layers.append(block(self.inplanes, planes, self.eps, stride, downsample1, downsample2,
                          use_ref_conv = self.use_ref_conv, use_ref_norm = self.use_ref_norm, dtype=self.dtype))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if self.use_bottleneck_tpp:
                if self.use_hardcoded_tunings is not None:
                    layers.append(block(self.inplanes, planes, self.eps, use_physical_3x3_padding=self.use_physical_3x3_padding,
                                    downsample1=None, downsample2=None, use_groupnorm = self.use_groupnorm, use_hardcoded_tunings = self.use_hardcoded_tunings,
                                    bc_conv1=self.channel_block_size, bc_conv2=self.channel_block_size, bc_conv3=self.channel_block_size, bk_conv3=self.channel_block_size,
                                    dtype=self.dtype))
                else:
                    layers.append(block(self.inplanes, planes, self.eps, use_physical_3x3_padding=self.use_physical_3x3_padding,
                                    downsample1=None, downsample2=None, use_groupnorm = self.use_groupnorm,
                                    bc_conv1=self.channel_block_size, bc_conv2=self.channel_block_size, bc_conv3=self.channel_block_size, bk_conv3=self.channel_block_size,
                                    dtype=self.dtype))
            else:
                layers.append(block(self.inplanes, planes, self.eps, use_ref_conv = self.use_ref_conv, use_ref_norm = self.use_ref_norm, dtype=self.dtype))

        return nn.Sequential(*layers)

    def forward(self, x):

        if self.validate_fwd:
            self.rank = int(os.environ.get("PMI_RANK", -1))
            if self.rank < 0:
                self.rank = 0
            if self.training:
                #self.dump_file_suffix    = '_train_sfx2' + '_rank_' + str(self.rank)
                self.dump_file_suffix    = '_train_tst_newfc' + '_rank_' + str(self.rank)
            else:
                #self.dump_file_suffix    = '_eval_sfx2' + '_rank_' + str(self.rank)
                #self.dump_file_suffix    = '_eval_tstext32' + '_rank_' + str(self.rank)
                #self.dump_file_suffix    = '_inf_ref' + '_rank_' + str(self.rank)
                self.dump_file_suffix    = '_inf_tst2' + '_rank_' + str(self.rank)

        #time_start = time.time()

        """
        if x.dim() == 4:
            size = [x.size(0), 1, x.size(1), x.size(2), x.size(3)]
            x = x.view(size).permute([0,1,3,4,2]).contiguous()
        """

        if self.validate_fwd and self.rank == 0 and self.fwd_counter == 1:
            if type(self.conv1.weight.grad) is BlockedParameter:
                self.conv1.weight.grad.unblock()
            #np.savetxt('my_conv1_forward_weight_grad_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', self.conv1.weight.grad.contiguous().view(-1).detach().to(torch.float).numpy())
            if type(self.conv1.weight.grad) is BlockedParameter:
                self.conv1.weight.grad.block()
            #self.tensor_dump_counter = self.tensor_dump_counter + 1

        if self.validate_fwd and self.rank == 0:
            tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            #np.savetxt('my_conv1_forward_input_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        if self.pad_input_for_bf16_ref and self.use_ref_conv:
          pad_shape = list(x.shape)
          pad_shape[1] = 1
          #print("debug: pad_shape = ", pad_shape)
          zero_pad = x.new_zeros(pad_shape)
          #print("debug: zero_pad shape = ", self.zero_pad.shape)

          #input = torch.cat((input, self.zero_pad), dim=-1)
          x = torch.cat((x, zero_pad), dim=1)
          #print("debug: input shape after padding = ", input.shape)

          #if self.validate_fwd and self.rank == 0:
          #    tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
          #    np.savetxt('my_conv1_forward_special_input_after_pad_resnet_sfx.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
          #    #self.tensor_dump_counter = self.tensor_dump_counter + 1

        if self.validate_fwd and self.rank == 0:
            if type(self.conv1.weight) is BlockedParameter:
                self.conv1.weight.unblock()
            #np.savetxt('my_conv1_forward_weight' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', self.conv1.weight.contiguous().view(-1).detach().to(torch.float).numpy())
            #self.tensor_dump_counter = self.tensor_dump_counter + 1
            #exit()
        x = self.conv1(x)

        #print("conv1 called")

        #print("before bn1, x.dtype x.shape = ", x.dtype, x.shape)

        if self.validate_fwd and self.rank == 0:
            #print("type(x) = ", type(x))
            tmp_tensor = x.unblocked_tensor() if callable(getattr(x,"unblocked_tensor", None)) else x #hasattr(x,"unblocked_tensor") else x
            #np.savetxt('my_conv1_forward_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        #exit()

        #if self.validate_fwd:
        #    if self.fwd_counter == 1:
        #        print("exiting because after conv1 of the self.fwd_counter (validate_fwd is enabled)")
        #        exit()

        if self.use_ref_norm == True:
            x = self.bn1(x)
            x = self.relu(x)
        else:
            x = self.bn1(x)
            #if isinstance(self.bn1, torch.nn.modules.linear.Identity):
            #    print("dbg resnet: calling relu as batchnorm has been folded into the identity")
            #    if self.relu is None:
            #        self.relu = nn.ReLU(inplace=False)
            #        x = self.relu(x)
        #exit()
        #print("bn1 called")

        #if self.validate_fwd and self.rank == 0:
        #    #np.savetxt('my_bn1_forward_runmean_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', self.bn1.running_mean.contiguous().view(-1).detach().to(torch.float).numpy())
        #    self.tensor_dump_counter = self.tensor_dump_counter + 1
        #    #np.savetxt('my_bn1_forward_runvar_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', self.bn1.running_var.contiguous().view(-1).detach().to(torch.float).numpy())
        #    self.tensor_dump_counter = self.tensor_dump_counter + 1

        if self.validate_fwd and self.rank == 0:
            tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            np.savetxt('my_bn1_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        #print("before maxpool, x.dtype x.shape = ", x.dtype, x.shape)

        x = self.maxpool(x)

        #print("maxpool called")

        if self.validate_fwd and self.rank == 0:
            tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            #np.savetxt('my_maxpool_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        x = self.layer1(x)

        #print("layer1 called")

        if self.validate_fwd and self.rank == 0:
            tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            #np.savetxt('my_layer1_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        x = self.layer2(x)
        if self.validate_fwd and self.rank == 0:
            tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            #np.savetxt('my_layer2_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        #print("layer2 called")

        x = self.layer3(x)
        if self.validate_fwd and self.rank == 0:
            tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            #np.savetxt('my_layer3_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        #print("layer3 called")

        x = self.layer4(x)
        if self.validate_fwd and self.rank == 0:
            tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            #np.savetxt('my_layer4_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        #print("layer4 called")

        x = self.avgpool(x)

        #print("avgpool called")

        if self.validate_fwd and self.rank == 0:
            #tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            #np.savetxt('my_avgpool_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        if self.use_ref_fc == True:
            x = x.view(x.size(0), -1).contiguous() #.to(torch.float32)?
        else: # converting to NCnc format with C for fc equal to CHW in regular terms
            x = torch.squeeze(x)

        #print("x.shape before fc = ", x.shape)
        #tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
        #np.savetxt('my_fc_input_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #self.tensor_dump_counter = self.tensor_dump_counter + 1
        if self.validate_fwd and self.rank == 0:
            #np.savetxt('my_fc_weight_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', self.fc.weight.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1
            #np.savetxt('my_fc_bias_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', self.fc.bias.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        #if self.validate_fwd and self.rank == 0 and self.fwd_counter == 1:
        #    if type(self.fc.weight.grad) is BlockedParameter:
        #        self.fc.weight.grad.unblock()
        #    np.savetxt('my_fc_forward_weight_grad_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', self.fc.weight.grad.contiguous().view(-1).detach().to(torch.float).numpy())
        #    if type(self.fc.weight.grad) is BlockedParameter:
        #        self.fc.weight.grad.block()
        #    self.tensor_dump_counter = self.tensor_dump_counter + 1

        #print("before fc, x.dtype x.shape = ", x.dtype, x.shape)
        #print("before fc, fc.weight.dtype fc.weight shape = ", self.fc.weight.dtype, self.fc.weight.shape)

        x = self.fc(x)

        #print("x.shape after fc = ", x.shape)
        #if self.validate_fwd and self.rank == 0:
        #    tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
        #    np.savetxt('my_fc_output_x_beforereshape' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
        #    self.tensor_dump_counter = self.tensor_dump_counter + 1

        if self.validate_fwd and self.rank == 0:
            tmp_tensor = x.unblocked_tensor() if hasattr(x,"unblocked_tensor") else x
            np.savetxt('my_fc_output_x_' + str(self.tensor_dump_counter) + self.dump_file_suffix + '.txt', tmp_tensor.contiguous().view(-1).detach().to(torch.float).numpy())
            self.tensor_dump_counter = self.tensor_dump_counter + 1

        #time_end = time.time()
        #global_time = global_time + (time_end - time_start)

        #exit()

        if self.validate_fwd:
            self.fwd_counter = self.fwd_counter + 1
            if self.fwd_counter == 2:
                print("exiting because of the self.fwd_counter (validate_fwd is enabled)")
                exit()

        return x

def resnet50_bn(use_ref_conv, use_ref_bn, use_ref_pool, use_ref_fc, validate_fwd, pad_input_for_bf16_ref,
                use_bottleneck_tpp, use_physical_3x3_padding, dtype, use_ext_bottleneck, use_hardcoded_tunings,
                channel_block_size, inference_mode,
                **kwargs):
    if use_bottleneck_tpp:
        if use_ext_bottleneck:
            bottleneck_module = tpp_pytorch_extension.resnet.bottleneck.BottleneckTPP
            model = ResNet(bottleneck_module, [3, 4, 6, 3], use_ref_conv=use_ref_conv, use_ref_norm=use_ref_bn, use_ref_pool=use_ref_pool, use_ref_fc=use_ref_fc, validate_fwd=validate_fwd,
                            pad_input_for_bf16_ref=pad_input_for_bf16_ref, use_bottleneck_tpp=use_bottleneck_tpp, use_physical_3x3_padding=use_physical_3x3_padding, use_groupnorm=False, dtype=dtype,
                            use_hardcoded_tunings=use_hardcoded_tunings,
                            channel_block_size=channel_block_size, inference_mode=inference_mode,
                             **kwargs)
        else:
            if inference_mode:
                print("Error: inference_mode = True is not supported by resnet50_bn when use_bottleneck_tpp and use_ext_bottleneck = ", use_bottleneck_tpp, use_ext_bottleneck)
                exit()
            bottleneck_module = BottleneckTPP
            model = ResNet(bottleneck_module, [3, 4, 6, 3], use_ref_conv=use_ref_conv, use_ref_norm=use_ref_bn, use_ref_pool=use_ref_pool, use_ref_fc=use_ref_fc, validate_fwd=validate_fwd,
                            pad_input_for_bf16_ref=pad_input_for_bf16_ref, use_bottleneck_tpp=use_bottleneck_tpp, use_physical_3x3_padding=use_physical_3x3_padding, use_groupnorm=False, dtype=dtype,
                            channel_block_size=channel_block_size,
                             **kwargs)
    else:
        if inference_mode:
            print("Warning: inference_mode = True is ignored by resnet50_bn when use_bottleneck_tpp = ", use_bottleneck_tpp)
        model = ResNet(BottleneckBN,    [3, 4, 6, 3], use_ref_conv=use_ref_conv, use_ref_norm=use_ref_bn, use_ref_pool=use_ref_pool, use_ref_fc=use_ref_fc, validate_fwd=validate_fwd,
                        pad_input_for_bf16_ref=pad_input_for_bf16_ref, use_bottleneck_tpp=use_bottleneck_tpp, use_physical_3x3_padding=False, use_groupnorm=False, dtype=dtype,
                           **kwargs)
    return model

def resnet50_gn(use_ref_conv, use_ref_gn, use_ref_pool, use_ref_fc, validate_fwd, pad_input_for_bf16_ref, use_bottleneck_tpp, use_physical_3x3_padding, dtype, inference_mode, **kwargs):
    if inference_mode:
        print("Error: inference_mode = True is not supported by resnet50_gn")
        exit()
    if use_bottleneck_tpp:
        if use_ext_bottleneck:
            print("Error: groupnorm is not supported for the extension bottleneck")
            exit()
            #bottleneck_module = tpp_pytorch_extension.BottleneckTPP
        else:
            bottleneck_module = BottleneckTPP
        model = ResNet(bottleneck_module, [3, 4, 6, 3], use_ref_conv=use_ref_conv, use_ref_norm=use_ref_gn, use_ref_pool=use_ref_pool, use_ref_fc=use_ref_fc, validate_fwd=validate_fwd,
                        pad_input_for_bf16_ref=pad_input_for_bf16_ref, use_bottleneck_tpp=use_bottleneck_tpp, use_physical_3x3_padding=use_physical_3x3_padding, use_groupnorm=True, dtype=dtype,
                           **kwargs)
    else:
        model = ResNet(BottleneckGN,    [3, 4, 6, 3], use_ref_conv=use_ref_conv, use_ref_norm=use_ref_gn, use_ref_pool=use_ref_pool, use_ref_fc=use_ref_fc, validate_fwd=validate_fwd,
                        pad_input_for_bf16_ref=pad_input_for_bf16_ref, use_bottleneck_tpp=use_bottleneck_tpp, use_physical_3x3_padding=False, use_groupnorm=True, dtype=dtype,
                           **kwargs)
    return model

def resnet101_bn(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152_bn(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

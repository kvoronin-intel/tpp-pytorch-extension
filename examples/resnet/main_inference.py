import argparse
import os
import random
import shutil
import time
import warnings
import sys
import psutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from resnet_pcl_united import resnet50_bn, resnet50_gn

import tpp_pytorch_extension
#from tpp_pytorch_extension._C import _conv as conv_cpp
from tpp_pytorch_extension.resnet import conv as conv_py

from tpp_pytorch_extension import optim as optim_py

import pcl_cgbp
from pcl_cgbp import ImplContextManager, block

try:
  import torch_ccl
except ImportError as e:
  print(e)
  print("falling back to importing oneccl_bindings_for_pytorch")
  try:
    import oneccl_bindings_for_pytorch
  except ImportError as e2:
    print(e2)

import pcl_extend_profiler

import blocked_layout

from collections import OrderedDict

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--iterations', default=100, type=int, metavar='N',
                    help='number of total iterations to run (synthetic data only)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)', dest='print_freq')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:12356', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--use-gpu", action="store_true", default=False)
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--perf-only', action='store_true', default=False, help='performance run only')
parser.add_argument("--enable-profiling", action="store_true", default=False)
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--use-bf16', action="store_true", default=False)
parser.add_argument("--use-ref-conv", action="store_true", default=False, help='uses reference (PyTorch) conv implementation in the opt resnet', dest='use_ref_conv')
parser.add_argument("--use-ref-bn", action="store_true", default=False, help='uses reference (PyTorch) bn implementation in the opt resnet', dest='use_ref_bn')
parser.add_argument("--use-ref-gn", action="store_true", default=False, help='uses reference (PyTorch) gn implementation in the opt resnet', dest='use_ref_gn')
parser.add_argument("--use-ref-pool", action="store_true", default=False, help='uses reference (PyTorch) pool implementation in the opt resnet', dest='use_ref_pool')
parser.add_argument("--use-ref-fc", action="store_true", default=False, help='uses reference (PyTorch) fc(linear) implementation in the opt resnet', dest='use_ref_fc')
parser.add_argument("--validate-fwd", action="store_true", default=False, help='enables functional validation mode which does two iterations and dumps output', dest='validate_fwd')

parser.add_argument("--synthetic", action="store_true", default=False, help='if true, uses synthetic data instead of the real dataset', dest='synthetic')

parser.add_argument("--pad-input-for-bf16-ref", action="store_true", default=False, help='enables physical channel padding for the first conv layer to compare fp32 ref with bf16 opt', dest='pad_input_for_bf16_ref')

parser.add_argument("--use-bottleneck-tpp", action="store_true", default=False, help='switches from bottleneck module implemented in PT to a monolithic TPP-based module', dest='use_bottleneck_tpp')

parser.add_argument("--use-phys3x3-padding", action="store_true", default=False, help='enables usage of physical 3x3 padding in the monolithic bottleneck tpp module', dest='use_physical_3x3_padding')

parser.add_argument("--resnet50-gn", action="store_true", default=False, help='enables usage of groupnorm instead of the batchnorm', dest='use_groupnorm')

parser.add_argument("--use-ext-bottleneck", action="store_true", default=False, help='enables usage of bottleneck module implemented in the extension repo', dest='use_ext_bottleneck')

parser.add_argument("--use-hardcoded-tunings", action="store_true", default=False, help='enables usage of hardcoded tuning parameters in ext bottleneck modules', dest='use_hardcoded_tunings')

parser.add_argument("--channel-block-size", default=None, type=int, help='block size for channels', dest='channel_block_size')

parser.add_argument("--use-new-conv2d", action="store_true", default=False, help='enables usage of TPPConv2D (Conv from PCL PT extensions) instead of nn conv2d (and pcl_cgbp XsmmConv2d)', dest='use_new_conv2d')

best_acc1 = 0

model_global_dtype = torch.float

class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

pcl_cgbp.init_libxsmm()

def set_inference_mode(module):
    print("dbg: folding: calling set_inference_mode for the module = ", module, type(module))
    #Given a model, set inference attribute to True in all bottleneck layers
    if hasattr(module, "enable_inference_mode"):
        print("dbg: folding: setting inference mode, module, type(module) = ", module,  type(module))
        module.enable_inference_mode()
    for name, child_module in module.named_children():
        set_inference_mode(child_module)
    return

def replace_batchnorms_with_identities(module):
    #print("dbg: folding: calling replace_batchnorm_with_identities for the module = ", module, type(module))
    #Given a model, replace all BatchNorm2d layers with Identity layers
    if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, tpp_pytorch_extension.resnet.batchnorm.TPPBatchNormTPP):
        #print("dbg: folding: replacing batchnorm with identity, module, type(module) = ", module,  type(module))
        return torch.nn.Identity()
    for name, child_module in module.named_children():
        child_module_mod = replace_batchnorms_with_identities(child_module)
        setattr(module, name, child_module_mod)
    return module

def fold_batchnorm(model):
    # Fold the batchnorm and reset BN params
    model.eval()
    conv = None
    conv_name = ""
    bn   = None
    for name,module in model.named_modules():
        #print("dbg: folding: name, module, type(module) = ", name, module, type(module))
        if type(module) in [torch.nn.Conv2d, tpp_pytorch_extension.resnet.conv.TPPConv2dTPP]:
            conv = module
            conv_name = name
        if type(module) in [torch.nn.BatchNorm2d, tpp_pytorch_extension.resnet.batchnorm.TPPBatchNormTPP]:
            bn = module.type(conv.weight.dtype)
            # Fusing when batch norm layer is encountered
            #print("dbg: conv bias before folding = ", conv.bias)
            #print("dbg: conv weight dtype before folding = ", conv.weight.dtype)
            fused_conv = torch.nn.utils.fusion.fuse_conv_bn_eval(conv,bn)
            #print("dbg: conv bias after folding = ", fused_conv.bias)
            #print("dbg: conv weight dtype after folding = ", fused_conv.weight.dtype)
            #print("dbg: conv bias dtype after folding = ", fused_conv.bias.dtype)

            # Copy weight and bias to conv
            with torch.no_grad():
                conv.weight.data.copy_(fused_conv.weight.data)
                if conv.bias is not None:
                    conv.bias.data.copy_(fused_conv.bias.data)
                else:
                    conv.bias = fused_conv.bias

    #exit()

    replace_batchnorms_with_identities(model)

    set_inference_mode(model)

    return model


def worker_init_fn(worker_id):
    cpu_aff = psutil.Process().cpu_affinity()
    cpu_aff_new = {cpu_aff[0] - worker_id - 1}
    try:
        psutil.Process().cpu_affinity(cpu_aff_new)
    except:
        print("Unable to set worker affinity {} for worker {}".format(cpu_aff_new, worker_id))

def main():
    global model_global_dtype
    if args.use_bf16:
        #pcl_cgbp.global_dtype = torch.bfloat16
        model_global_dtype    = torch.bfloat16
    else:
        #pcl_cgbp.global_dtype = torch.float
        model_global_dtype    = torch.float

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("PMI_SIZE", -1))
        if args.world_size == -1: args.world_size = int(os.environ["WORLD_SIZE"])
        #args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global model_norm_name
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for inference".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ.get("PMI_RANK", -1))
            if args.rank == -1: args.rank = int(os.environ["RANK"])
            #args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.use_groupnorm:
        model_norm_name = 'gn'
        print("=> creating model '{}'".format('resnet50_' + model_norm_name))
        model = resnet50_gn(use_ref_conv = args.use_ref_conv, use_ref_gn = args.use_ref_gn, use_ref_pool = args.use_ref_pool, use_ref_fc = args.use_ref_fc,
                            validate_fwd = args.validate_fwd, pad_input_for_bf16_ref = args.pad_input_for_bf16_ref, use_bottleneck_tpp = args.use_bottleneck_tpp,
                            use_physical_3x3_padding = args.use_physical_3x3_padding, dtype = model_global_dtype, use_ext_bottleneck = args.use_ext_bottleneck, inference_mode=True)
    else:
        model_norm_name = 'bn'
        print("=> creating model '{}'".format('resnet50_' + model_norm_name))
        model = resnet50_bn(use_ref_conv = args.use_ref_conv, use_ref_bn = args.use_ref_bn, use_ref_pool = args.use_ref_pool, use_ref_fc = args.use_ref_fc,
                            validate_fwd = args.validate_fwd, pad_input_for_bf16_ref = args.pad_input_for_bf16_ref, use_bottleneck_tpp = args.use_bottleneck_tpp,
                            use_physical_3x3_padding = args.use_physical_3x3_padding, dtype = model_global_dtype, use_ext_bottleneck = args.use_ext_bottleneck,
                            use_hardcoded_tunings = args.use_hardcoded_tunings, channel_block_size=args.channel_block_size, inference_mode=True)
    if args.distributed:
        if args.rank == 0:
            print(model)
    else:
        print(model)

    print("Folding batchnorm")
    model = fold_batchnorm(model)

    block(model)

    if args.use_gpu and ngpus_per_node > 0:
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
    elif args.distributed:
        # args.batch_size = int(args.batch_size / ngpus_per_node)
        #if args.rank == 0:
        #    print("No DDP (DistributedDataParallel) while debugging")
        #pass
        model = torch.nn.parallel.DistributedDataParallel(model)

    #exit()

    if args.distributed:
        checkpoint_name = 'checkpoint_at_init_distr_' + model_norm_name + '_N' + str(args.batch_size) + '_inference.pth.tar';
    else:
        checkpoint_name = 'checkpoint_at_init_' + model_norm_name + '_N'  + str(args.batch_size) + '_inference.pth.tar';

    
    # Saving model data
    if args.use_ref_conv and ((args.use_groupnorm and args.use_ref_gn) or (not args.use_groupnorm and args.use_ref_bn)) and args.use_ref_pool and args.use_ref_fc:
        print("Saving initialized model")
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            torch.save(model.state_dict(), checkpoint_name)
        elif args.distributed and args.rank == 0:
            torch.save(model.state_dict(), checkpoint_name)
        #exit()
    

    """
    if args.rank == 0:
        print("model state keys = ", model.state_dict().keys())
        print("Model's state_dict:")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    """
    # Loading model
    if not args.use_ref_conv or not args.use_ref_bn or not args.use_ref_pool or not args.use_ref_fc:
        print("Loading initialized model from a checkpoint ", checkpoint_name)
        checkpoint = torch.load(checkpoint_name)
        #print("checkpoint keys before = ", checkpoint.keys())
        if args.use_bf16:

            #print("type of checkpoint = ", type(checkpoint))
            state_before = model.state_dict()
            #print("dbg: loaded checkpoint keys = ", checkpoint.keys())
            checkpoint_downconverted_bf16 = OrderedDict()
            for param_tensor in checkpoint:
                #checkpoint_bf16 +=
                #print("dbg:", param_tensor, "\t", checkpoint[param_tensor].size(), "\t", checkpoint[param_tensor].dtype, "\t", state_before[param_tensor].dtype)
                checkpoint_downconverted_bf16[param_tensor] = checkpoint[param_tensor].to(state_before[param_tensor].dtype)
            #exit()
            model.load_state_dict(checkpoint_downconverted_bf16)
            """
            print("after loading model state dict:")
            state_after = model.state_dict()
            for param_tensor in state_after:
                #checkpoint_bf16 +=
                #print("param_tensor, type = ", param_tensor, type(param_tensor))
                print("param_tensor", "\t", state_after[param_tensor].size(), "\t", state_after[param_tensor].dtype)
            exit()
            """
        else:
            model.load_state_dict(checkpoint)
    
    #exit()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.use_gpu and ngpus_per_node > 0:
        criterion = criterion.cuda(args.gpu)

    if args.gpu:
      cudnn.benchmark = True

    # Data loading code
    if not args.synthetic:
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    if args.synthetic:
        val_loader = None
    else:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=True, worker_init_fn=worker_init_fn, persistent_workers=True)

    validate(val_loader, model, criterion, args)
    return

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    #with torch.no_grad():
    #with torch.no_grad() and torch.inference_mode():
    #with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=record_shapes) if args.enable_profiling else dummy_context_mgr() as prof:
    #with torch.no_grad() and torch.inference_mode() and (torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=False) if args.enable_profiling else dummy_context_mgr() as prof): # bad syntax
    with torch.no_grad() and torch.inference_mode():
      record_shapes=False
      with torch.autograd.profiler.profile(enabled=True, use_cuda=False, record_shapes=record_shapes) if args.enable_profiling else dummy_context_mgr() as prof:

        if args.enable_profiling:
            tpp_pytorch_extension.reset_debug_timers()

        end = time.time()

        if args.synthetic:
            input_ref  = torch.randn(args.batch_size, 3, 224, 224)
            target_ref = torch.randint(0,999,(args.batch_size,))#.to(torch.float)

            data_iterator = range(args.iterations)
        else:
            data_iterator = enumerate(train_loader)

        #for i, (input, target) in enumerate(val_loader):

        # for i, (input, target) in enumerate(val_loader):
        # data_item is i for synthetic and a tuple (i, input) for real dataset
        for data_item in data_iterator: #range(args.iterations):

            i = data_item[0] if type(data_item) is tuple else data_item

            #print("dbg: i = ", i)

            if args.synthetic:
                input  = input_ref.detach()
                target = target_ref.detach()
            else:
                (input, target) = data_item[1]

            if args.use_gpu:
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            if args.use_bf16:
                input_bf16  = input.to(torch.bfloat16)
                #target_bf16 = target.to(torch.bfloat16)

                model_input = input_bf16
            else:
                model_input = input

            # compute output
            model_output = model(model_input)
            if args.use_bf16:
                #if hasattr(model_output,"unblocked_tensor"):
                #    output = model_output.unblocked_tensor().to(torch.float)
                #else:
                #    output = model_output.to(torch.float)
                output = model_output.to(torch.float)
            else:
                output = model_output
            #output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.rank == 0:
              print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader) if val_loader else args.iterations, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

            if args.validate_fwd and i == 10:
              #if i == 10:
              print("Breaking for loop in validate() due to the early stop condition for i")
              break

        if args.rank == 0:
          print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if args.enable_profiling:
            tpp_pytorch_extension.print_debug_timers()

      if args.enable_profiling:
        if not args.distributed or (args.distributed and args.rank == 0):
            print("Writing profiler data")

        with open('rn50_' + model_norm_name + '_inference_profile', 'w') as f:
            f.write(prof.nested_key_averages(only_top_level=True).table(sort_by="self_cpu_time_total")) # was True
            f.flush()

        if prof:
            file_prefix = "my_prefix_" #"squad_time%s" % ("_r%d" % args.local_rank if args.local_rank >= 0 else "")
            with open("%s.prof" % file_prefix, "w") as prof_f:
                prof_f.write(prof.key_averages(group_by_input_shape=record_shapes).table(sort_by="cpu_time_total"))
            try:
                with open("%s.nested.prof" % file_prefix, "w") as prof_f:
                    #prof_f.write(prof.nested_key_averages().table(sort_by="cpu_time_total"))
                    prof_f.write(prof.nested_key_averages().table(sort_by=None, row_limit=1000))
                with open("%s.top_level.prof" % file_prefix, "w") as prof_f:
                    prof_f.write(prof.nested_key_averages(only_top_level=True).table(sort_by="cpu_time_total"))
                    prof.print_op_timings(prof, prefix=file_prefix)
            except:
                pass


    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    args = parser.parse_args()

    if args.pad_input_for_bf16_ref and not args.use_ref_conv:
        print("Error: pad_input_for_bf16_ref should only be enabled when reference PT conv is used")
        exit()
    if args.pad_input_for_bf16_ref and args.use_bf16:
        print("Error: pad_input_for_bf16_ref only makes sense when running fp32 (so use_bf16 must be switched off)")
        exit()
    if args.use_bottleneck_tpp and (args.use_ref_bn or args.use_ref_gn):
        print("Error: use_bottleneck_tpp implies that TPP batchnorm (groupnorm) are used")
        exit()
    if args.use_physical_3x3_padding and not args.use_bottleneck_tpp:
        print("Error: use_physical_3x3_padding implies that monolithic TPP bottlenneck module is used (args.use_bottleneck_tpp)")
        exit()
    if args.use_ext_bottleneck  and not args.use_bottleneck_tpp:
        print("Error: use_ext_bottleneck implies that monolithic TPP bottlenneck module is used (args.use_bottleneck_tpp)")
        exit()
    if args.use_ext_bottleneck and not args.use_physical_3x3_padding:
        print("Error: use_ext_bottleneck implies that use_physical_3x3_padding is enabled (as only physical padding is supported by it)")
        exit()
    if not args.use_ext_bottleneck and args.use_hardcoded_tunings:
        print("Error: use_ext_bottleneck must be enabled for use_hardcoded_tunings option to make sense")
        exit()
    if not args.use_bottleneck_tpp and args.channel_block_size != None:
        print("Error: use_bottleneck_tpp must be enabled for non-default channel_block_size to make sense")
        exit()
    #print(args)
    with ImplContextManager(args.use_ref_conv, args.use_ref_bn, args.use_ref_gn, args.use_ref_pool, args.use_ref_fc):
        # extra (w.r.t to ImplContextManager definition) overwrite = a hack to use TPPConv2dTPP from PCL PT extensions
        # in place of nn.Conv2D (and not XsmmConv2d from pcl_cgbp)
        if args.use_new_conv2d:
            torch.nn.Conv2d = conv_py.TPPConv2dTPP
        #print("dbg: torch.nn.Conv2d = ", torch.nn.Conv2d)
        main()


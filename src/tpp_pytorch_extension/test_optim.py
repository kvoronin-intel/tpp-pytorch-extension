import argparse
import torch
from torch.autograd import Variable
import numpy as np
import pdb

from collections import OrderedDict

import copy


import pcl_pytorch_extension
from pcl_pytorch_extension import optim as optim_py
#import conv as conv_py

#import optim
#from pcl_optim import SGD_fb_enhanced, SGD_bf16_enhanced, SGD_bf16fb_enhanced

parser = argparse.ArgumentParser(description='PCL PyTorch extension standalone testing for optimizers')

parser.add_argument('--test-optim', default=None, type=str,
                    help='module to test against the reference', dest='test_optim')

parser.add_argument('--use-bf16-opt', action="store_true", default=False, dest='use_bf16_opt')

parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, dtype=torch.float):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=True, device=None, dtype=dtype)

    def forward(self, x):
        out = self.linear(x)
        return out

def main():

    torch.manual_seed(0)

    #pdb.set_trace()

    # Create dummy data for training
    x_values = [i for i in range(11)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    print("x_values = ", x_values)
    print("y_values = ", y_values)

    # Creating a simple toy model
    inputDim     = 1        # takes variable 'x'
    outputDim    = 1        # takes variable 'y'
    epochs       = 10

    model_torch = LinearRegression(inputDim, outputDim, torch.float)
    if args.use_bf16_opt:
        print("Saving initialized fp32 model")
        torch.save(model_torch.state_dict(), 'checkpoint_toy_model_test_optim.pth.tar')
        print("Loading initialized fp32 model and converting to bf16")
        checkpoint = torch.load('checkpoint_toy_model_test_optim.pth.tar')
        checkpoint_bf16 = OrderedDict()
        for param_tensor in checkpoint:
            checkpoint_bf16[param_tensor] = checkpoint[param_tensor].to(torch.bfloat16)
        #exit()
        model_opt = LinearRegression(inputDim, outputDim, torch.bfloat16)
        model_opt.load_state_dict(checkpoint_bf16)
    else:
        model_opt = copy.deepcopy(model_torch)

    criterion = torch.nn.MSELoss()

    #optimizer_opt   = torch.optim.SGD(model_opt.parameters(), lr=args.lr)
    if args.test_optim == 'SGD_fb_enhanced':
        optimizer_opt = optim_py.SGD_fb_enhanced(model_opt.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.test_optim == 'SGD_bf16_enhanced':
        optimizer_opt = optim_py.SGD_bf16_enhanced(model_opt.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.test_optim == 'SGD_bf16fb_enhanced':
        optimizer_opt = optim_py.SGD_bf16fb_enhanced(model_opt.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.test_optim == 'SplitSGD_bf16fb_enhanced':
        optimizer_opt = optim_py.SplitSGD_bf16fb_enhanced(model_opt.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.test_optim == 'ref':
        optimizer_opt = torch.optim.SGD(model_opt.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise RuntimeError('unrecognized args.test_optim value = ', args.test_optim)

    #optimizer_torch = torch.optim.AdamW(model_torch.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    optimizer_torch = torch.optim.SGD(model_torch.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    print("optimizer_opt   = ", optimizer_opt)
    print("optimizer_torch = ", optimizer_torch)

    epoch_to_checkpoint = epochs // 2

    checkpoint_name_torch = "checkpoint_torch.pt"
    checkpoint_name_opt   = "checkpoint_opt.pt"

    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        inputs = Variable(torch.from_numpy(x_train))
        if args.use_bf16_opt:
            inputs_opt = copy.deepcopy(inputs).to(torch.bfloat16)
        else:
            inputs_opt = copy.deepcopy(inputs)

        labels = Variable(torch.from_numpy(y_train))
        if args.use_bf16_opt:
            labels_opt = copy.deepcopy(labels).to(torch.bfloat16)
        else:
            labels_opt = copy.deepcopy(labels)
        #print("inputs = ", inputs)
        #print("labels = ", labels)

        # Checkpoint loading
        if epoch == epoch_to_checkpoint + 1:
            print("Loading a checkpoint (for stock setup) for epoch = ", epoch);
            checkpoint_torch = torch.load(checkpoint_name_torch)
            #print("dbg: checkpoint torch = ", checkpoint_torch)
            #print("dbg: checkpoint_torch keys = ", checkpoint_torch.keys())

            model_state_dict_before = model_torch.state_dict()
            model_state_dict_downconverted_torch = OrderedDict()
            for param_tensor in checkpoint_torch['model_state_dict']:
                #print("dbg: param_tensor, type = ", param_tensor, type(param_tensor))
                #checkpoint_bf16 +=
                #print("dbg:", param_tensor, "\t", checkpoint_torch['model_state_dict'][param_tensor].shape, "\t", checkpoint_torch['model_state_dict'][param_tensor].dtype)
                model_state_dict_downconverted_torch[param_tensor] = checkpoint_torch['model_state_dict'][param_tensor].to(model_state_dict_before[param_tensor].dtype)
            #exit()
            model_torch.load_state_dict(model_state_dict_downconverted_torch)
            optimizer_torch.load_state_dict(checkpoint_torch['optimizer_state_dict'])
            loss_torch = checkpoint_torch['loss']

            print("Loading a checkpoint (for optimized setup) for epoch = ", epoch);
            checkpoint_opt = torch.load(checkpoint_name_opt)

            model_state_dict_before = model_opt.state_dict()
            model_state_dict_downconverted_opt = OrderedDict()
            for param_tensor in checkpoint_opt['model_state_dict']:
                #print("dbg: param_tensor, type = ", param_tensor, type(param_tensor))
                #checkpoint_bf16 +=
                #print("dbg:", param_tensor, "\t", checkpoint_opt['model_state_dict'][param_tensor].shape, "\t", checkpoint_opt['model_state_dict'][param_tensor].dtype)
                model_state_dict_downconverted_opt[param_tensor] = checkpoint_opt['model_state_dict'][param_tensor].to(model_state_dict_before[param_tensor].dtype)
            #exit()
            model_opt.load_state_dict(model_state_dict_downconverted_opt)
            optimizer_opt.load_state_dict(checkpoint_opt['optimizer_state_dict'])
            loss_opt = checkpoint_opt['loss']

            #exit()

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer_torch.zero_grad()
        optimizer_opt.zero_grad()

        # get output from the model, given the inputs
        outputs_torch = model_torch(inputs)
        outputs_opt   = model_opt(inputs_opt)

        # get loss for the predicted output
        loss_torch = criterion(outputs_torch, labels)
        #loss_torch = criterion(outputs_torch.to(torch.bfloat16), labels)
        #print(loss_torch)
        #loss_opt   = criterion(outputs_opt, labels_opt)
        loss_opt   = criterion(outputs_opt.to(torch.float), labels_opt.to(torch.float))
        #print(loss_opt)

        #print('epoch {}, loss_torch {}'.format(epoch, loss_torch.item()))
        print('epoch {}, loss_torch {}, loss_opt {}'.format(epoch, loss_torch.item(), loss_opt.item()))

        # get gradients w.r.t to parameters
        loss_torch.backward()
        loss_opt.backward()

        # update parameters
        optimizer_torch.step()
        optimizer_opt.step()

        #print("Optimizer ref's state_dict:")
        #for var_name in optimizer_torch.state_dict():
        #    print(var_name, "\t", optimizer_torch.state_dict()[var_name])

        #print("Optimizer opt's state_dict:")
        #for var_name in optimizer_opt.state_dict():
        #    print(var_name, "\t", optimizer_opt.state_dict()[var_name])


        if epoch == epoch_to_checkpoint:
            #print("")
            print("Saving a checkpoint for epoch = (stock setup)", epoch);

            #print("dbg: optimizer state dict torch = ", optimizer_torch.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_torch.state_dict(),
                'optimizer_state_dict': optimizer_torch.state_dict(),
                'loss': loss_torch,
            }, checkpoint_name_torch)
            """
            save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model_torch.state_dict(),
                    'optimizer_state_dict' : optimizer_torch.state_dict(),
                    'loss': loss_torch,
                })
            """

            #print("dbg: optimizer state dict dbg opt = ", optimizer_opt.state_dict_dbg())
            #exit()

            print("Saving a checkpoint for epoch = (optimized setup)", epoch);
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_opt.state_dict(),
                #'optimizer_state_dict': optimizer_opt.state_dict_dbg(),
                'optimizer_state_dict': optimizer_opt.state_dict(),
                'loss': loss_opt,
            }, checkpoint_name_opt)
            """
            save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model_opt.state_dict(),
                    'optimizer_state_dict' : optimizer_opt.state_dict(),
                    'loss': loss_opt,
                })
            """
        #print("dbg: model torch weight =  grad = ", model_torch.linear.weight, model_torch.linear.weight.grad)
        #print("dbg: model opt   weight =  grad = ", model_opt.linear.weight,   model_opt.linear.weight.grad)

        #state_dict_dbg2 = optimizer_opt.state_dict_dbg()
        #print("state_dict_dbg here 2 = ", state_dict_dbg1)

        # This is done to check that the state dict is correct in the optimizer
        torch.save({
          'epoch': epoch + 1,
          #'state_dict': model.state_dict(),
          #'best_acc1': best_acc1,
          'optimizer' : optimizer_opt.state_dict(),
        }, 'checkpoint_test_optim.pth.tar')


if __name__ == "__main__":
    args = parser.parse_args()
    main()


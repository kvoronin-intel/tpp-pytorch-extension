import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
from pcl_pytorch_extension.gnn.gat import fused_GAT as pcl_gat
from dgl.nn.pytorch.conv import GATConv
from pcl_pytorch_extension.gnn.common import gnn_utils

import pcl_pytorch_extension as ppx
import os
import psutil

from contextlib import contextmanager

@contextmanager
def pcl_impl(enable=True, use_bf16=False):
    try:
        global GATConv
        orig_GATConv = GATConv
        try:
            if enable:
                if use_bf16:
                    GATConv = pcl_gat.GATConvOptBF16

                else:
                    GATConv = pcl_gat.GATConvOpt
            yield
        finally:
            GATConv = orig_GATConv
    except ImportError as e:
        pass


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 num_heads,
                 activation):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(GATConv((in_feats, in_feats), n_hidden, num_heads=num_heads, activation=activation))
        for i in range(1, n_layers - 1):
            self.layers.append(GATConv((n_hidden * num_heads, n_hidden * num_heads), n_hidden,
                                             num_heads=num_heads, activation=activation))
        self.layers.append(GATConv((n_hidden * num_heads, n_hidden * num_heads), n_classes,
                                         num_heads=num_heads, activation=None))
        


    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst), False).flatten(1)
            else:
                h = layer(block, (h, h_dst), False)
        h = h.mean(1)
        return h.log_softmax(dim=-1)

  ## pcl code
    def inference_full(self, g, x, device):
        h = x.to(device)
        h_dst = h
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                h = layer(g, (h, h), False).flatten(1)
            else:
                h = layer(g, (h, h), False)
                h = h.mean(1)
                h = h.log_softmax(dim=-1)
        return h



    def inference(self, g, x, num_heads, device):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        if device.type != "cpu":
            for l, layer in enumerate(self.layers):
                if args.use_bf16:
                    if l < self.n_layers - 1:
                        y = (th.zeros(g.num_nodes(), self.n_hidden * num_heads if l != len(self.layers) - 1 else self.n_classes))#.to(th.bfloat16)
                    else:
                        y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)
                else:
                    if l < self.n_layers - 1:
                        y = th.zeros(g.num_nodes(), self.n_hidden * num_heads if l != len(self.layers) - 1 else self.n_classes)
                    else:
                        y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    th.arange(g.num_nodes()),
                    sampler,
                    batch_size=200000, #500000, 
                    # batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.num_workers)#,
                    # use_cpu_worker_affinity=True)

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    block = blocks[0].int().to(device)
                    if args.use_bf16:
                        x = x.to(th.bfloat16)
                    h = x[input_nodes]
                    h_dst = h[:block.num_dst_nodes()]
                    if l < self.n_layers - 1:
                        h = layer(block, (h, h_dst)).flatten(1) 
                    else:
                        h = layer(block, (h, h_dst)).to(th.float32)
                        h = h.mean(1)
                        h = h.log_softmax(dim=-1)

                    y[output_nodes] = h.cpu()

            x = y
            return y.to(device)
        else:
            if args.use_bf16:
                x = x.to(th.bfloat16)

            for l, layer in enumerate(self.layers):
                x = layer(g, (x, x), train=False)

            return x

     ## pcl code
    def inference_full(self, g, x, device):
        h = x.to(device)
        h_dst = h
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                h = layer(g, (h, h), train=False).flatten(1)
            else:
                h = layer(g, (h, h), train=False)
                h = h.mean(1)
                h = h.log_softmax(dim=-1)
        return h
        
 


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, test_nid, num_heads, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    ## =============================
    with th.no_grad():
        pred = model.inference(g, nfeat, num_heads, device).to(th.float32)
    model.train()
    ## =============================
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(pred[test_nid], labels[test_nid]), pred

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate_(model, g, nfeat, labels, val_nid, test_nid, num_heads, device):
    model.eval()
    with th.no_grad():
        pred = model(nfeat)
        pred_val = pred[val_nid]
        labels_val = labels[val_nid]
        pred_tst = pred[test_nid]
        labels_tst = labels[test_nid]
        return accuracy(pred_val, labels_val), accuracy(pred_tst, labels_tst), pred

def evaluate_full(model, g, nfeat, labels, val_nid, device):
    model.eval()
    with th.no_grad():
        pred = model.inference_full(g, nfeat, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
    #return correct.item() * 1.0 / len(labels), correct.item(), len(labels)

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    #if args.opt_mlp:
    #    if use_bf16:
    #        batch_inputs = nfeat[input_nodes].to(th.bfloat16)
            #batch_inputs = gnn_utils.gather_features(nfeat, input_nodes)#.to(th.bfloat16)
    #    else:
            #batch_inputs = gnn_utils.gather_features(nfeat, input_nodes)
    #        batch_inputs = nfeat[input_nodes]
    #else:
    #    if use_bf16:
    #    batch_inputs = nfeat[input_nodes].to(th.bfloat16)
    #    else:
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]

    return batch_inputs, batch_labels

def worker_init_fn(worker_id):
    cpu_aff = psutil.Process().cpu_affinity()
    cpu_aff_new = {cpu_aff[0] - worker_id - 1}
    try:
        psutil.Process().cpu_affinity(cpu_aff_new)
        print(
            "Worker {} with pid {} called, new affinity = {}".format(
                worker_id, os.getpid(), psutil.Process().cpu_affinity()
            )
        )
    except:
        print(
            "Unable to set worker affinity {} for worker {}".format(
                cpu_aff_new, worker_id
            )
        )
class AverageMeter(object):
 
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

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    th.save(state, filename, _use_new_zipfile_serialization=True)


#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g, num_heads = data


    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    if args.dataset == "ogbn-papers100M":
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            train_nid,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            use_cpu_worker_affinity=args.cpu_worker_aff,
            persistent_workers=args.cpu_worker_aff,
            #formats=["csc"],
        )
    else:
       dataloader = dgl.dataloading.NodeDataLoader(
          g,
          train_nid,
          sampler,
          batch_size=args.batch_size,
          shuffle=True,
          drop_last=False,
          num_workers=args.num_workers,
          persistent_workers=args.cpu_worker_aff,
          use_cpu_worker_affinity=args.cpu_worker_aff
       )

    # Define model and optimizer
    with pcl_impl(args.opt_mlp, args.use_bf16):
        model = GAT(in_feats, args.num_hidden, n_classes, args.num_layers, num_heads, F.relu)
        model = model.to(device)
        print("Model path ", GATConv)


    if args.opt_mlp:
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.wd,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = ppx.optim.AdamW(
            optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

    # Training loop
    avg = 0
    avgb = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    if args.opt_mlp:
        ppx.manual_seed(args.seed)
    # else:
    #     manual_seed(args.seed)
    record_shapes = False
    with th.autograd.profiler.profile(
        enabled=args.profile, use_cuda=(args.gpu > 0), record_shapes=record_shapes
    ) as prof:
        # if prof and args.opt_mlp:
        #     ppx.reset_debug_timers()
        if args.opt_mlp: ppx.reset_debug_timers()
        best_acc = 0

        for epoch in range(args.num_epochs):
            batch_fwd_time = AverageMeter()
            batch_bwd_time = AverageMeter()
            loss_time = AverageMeter()
            optim_zero_time = AverageMeter()
            optim_step_time = AverageMeter()
            data_time = AverageMeter()
            gather_time = AverageMeter()
 
 
            iter_time = []

            tic = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            
            end = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # ppx.reset_debug_timers()
                if args.opt_mlp and epoch == 0 and step == 0:
                    cores = int(os.environ["OMP_NUM_THREADS"])
                    gnn_utils.affinitize_cores(cores, args.num_workers)

                t0 = time.time()
                if step > 0: data_time.update(t0 - end)

                # copy block to gpu
                blocks = [blk.to(device) for blk in blocks]

                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(nfeat, labels, seeds, input_nodes)

                t1 = time.time()
                if step > 0: gather_time.update(t1 - t0)
                t2 = time.time()

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs).to(th.float32)
                t3 = time.time()
                if step > 0: batch_fwd_time.update(t3 - t2)

                loss = F.nll_loss(batch_pred, batch_labels)
                t4 = time.time()
                if step > 0: loss_time.update(t4 - t3)

                optimizer.zero_grad()
                t5 = time.time()
                if step > 0: optim_zero_time.update(t5 - t4)

                # t1 = time.time()
                loss.backward()
                t6 = time.time()
                if step > 0: batch_bwd_time.update(t6 - t5)

                # print("loss time: ", time.time()- t1)
                optimizer.step()
                t7 = time.time()
                if step > 0: optim_step_time.update(t7 - t6)

                end = time.time()
                if step > 0:
                   iter_time.append(data_time.val + gather_time.val + batch_fwd_time.val + loss_time.val + optim_zero_time.val + batch_bwd_time.val + optim_step_time.val)                   
                #iter_tput.append(len(seeds) / (time.time() - tic_step))
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    print(
                        #"Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} |"
                        #"Nodes: Input {:d} H1 {:d} H2 {:d} Out {:d} |"
                        "Epoch {:05d} | Step {:05d} |"
                        "DL (s) {data_time.val:.3f} ({data_time.avg:.3f}) |"
                        "GT (s) {gather_time.val:.3f} ({gather_time.avg:.3f}) |"
                        "FWD (s) {batch_fwd_time.val:.3f} ({batch_fwd_time.avg:.3f}) |"
                        "Loss (s) {loss_time.val:.4f} ({loss_time.avg:.4f}) |"
                        "Optim Zero (s) {optim_zero_time.val:.4f} ({optim_zero_time.avg:.4f}) |"
                        "BWD (s) {batch_bwd_time.val:.3f} ({batch_bwd_time.avg:.3f}) |"
                        "Optim Step (s) {optim_step_time.val:.4f} ({optim_step_time.avg:.4f}) |".format(
                            epoch,
                            step,
                            loss.item(),
                            acc.item(),
                            #blocks[0].num_src_nodes(),
                            #blocks[1].num_src_nodes(),
                            #blocks[2].num_src_nodes(),
                            #blocks[2].num_dst_nodes(),
                            data_time=data_time,
                            gather_time=gather_time,
                            batch_fwd_time=batch_fwd_time,
                            loss_time=loss_time,
                            optim_zero_time=optim_zero_time,
                            batch_bwd_time=batch_bwd_time,
                            optim_step_time=optim_step_time,
                        )
                    )

            
            print("Epoch Time(s): {:.4f}".format(np.sum(iter_time)))
            process = psutil.Process(os.getpid())
            #print('Mem usage in GB: ', process.memory_info().rss/1e9, flush=True)  # in bytes
            print('Mem usage in GB: ', process.memory_info().rss/1e9)  # in bytes


                    #gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                    #print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                     #   epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
                # if prof and args.opt_mlp:
                # ppx.print_debug_timers(0)
            # # ""
            # if prof:
            #   with open("gat_prof.prof", "w") as prof_f:
            #     prof_f.write(prof.key_averages(group_by_input_shape=record_shapes).table(sort_by="cpu_time_total"))
            #   if ppx.extend_profiler:
            #     with open("gat_nested.prof", "w") as prof_f:
            #       prof_f.write(prof.nested_key_averages().table(sort_by=None, row_limit=1000))
            #     with open("gat_top_level.prof", "w") as prof_f:
            #       prof_f.write(prof.nested_key_averages(only_top_level=True).table(sort_by="cpu_time_total"))
            #     prof.print_op_timings(prefix="gsage_time_")
            # # """

            toc = time.time()
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch >= 1:
                avgb += toc - tic

            ## single node inference
             # cum_acc_val = evaluate_full(model, g, features_g, labels_g, val_nid_g, device)
            #acc = evaluate(model, g_orig, features_g, labels_g, test_nid_g, device)
             # cum_acc_test = evaluate_full(model, g, features_g, labels_g, test_nid_g, device)
            ###############################
            # cum_acc_val = evaluate(model, g_orig, features_g, labels_g, val_nid_g, num_heads, device)
            # cum_acc_test = evaluate(model, g_orig, features_g, labels_g, test_nid_g, num_heads, device)
 
            ################################
             # if best_acc < cum_acc_test: best_acc = cum_acc_test
             # if args.rank == 0:
             #   print("#############################################################", flush=True)
             #   print("[SS] Epoch: {} Single node val accuracy Avg: {:0.4f} and test accuracy: "
             #         "Avg: {:0.4f}, best_acc: {:.2f} with lr: {}".
             #         format(epoch, float(cum_acc_val)*100, float(cum_acc_test)*100,
             #                float(best_acc)*100,args.lr), flush=True)
             #   print("#############################################################", flush=True)
 

            if args.dataset != 'ogbn-papers100M':
                if not args.profile and epoch % args.eval_every == 0 and epoch != 0:
                   #eval_acc, test_acc, pred = evaluate(model, g, nfeat, labels, val_nid, test_nid, num_heads, device)
                   ## single node inference
                   cum_acc_val = evaluate_full(model, g, nfeat, labels, val_nid, device) 
                   #acc = evaluate(model, g_orig, features_g, labels_g, test_nid_g, device)
                   cum_acc_test = evaluate_full(model, g, nfeat, labels, test_nid, device) 

                   if best_acc < cum_acc_test: best_acc = cum_acc_test
                   #print("#############################################################", flush=True)
                   print("#############################################################")
                   print("[SS] Epoch: {} Single node val accuracy Avg: {:0.4f} and test accuracy: "
                           "Avg: {:0.4f}, best_acc: {:.2f} with lr: {}".
                           format(epoch, float(cum_acc_val)*100, float(cum_acc_test)*100,
                                  float(best_acc)*100,args.lr), flush=True)
                   print("#############################################################")
 

                   # eval_acc = accuracy()
                   # if args.save_pred: 
                   #     np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
                   #print('Eval Acc {:.4f}'.format(eval_acc))
                   #if eval_acc > best_eval_acc:
                   #   best_eval_acc = eval_acc
                   #   best_test_acc = test_acc
                   #print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))
            else:
                #save_checkpoint(
                #       {"state_dict": model.state_dict()},
                #       filename="ogbp100M" + str(epoch) + ".pth.tar",
                #    )
               if  epoch % args.eval_every == 0 or epoch == args.n_epochs - 1:
                   if args.checkpoint:
                        checkpoint_file = args.checkpoint_dir + "/" + args.checkpoint_file + str(epoch)
                        print("Saving model: ")
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': model.layers.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, filename=checkpoint_file)
                        print("Model saved! in ", checkpoint_file, flush=True)     

    # if prof and args.opt_mlp:
        # ppx.print_debug_timers(0)
    if args.opt_mlp: ppx.print_debug_timers(0)
    # ppx.print_debug_timers(0)

    if prof:
        with open("gat.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=record_shapes).table(
                    sort_by="cpu_time_total"
                )
            )
        if ppx.extend_profiler:
            with open("gat_nested.prof", "w") as prof_f:
                prof_f.write(
                    prof.nested_key_averages().table(sort_by=None, row_limit=1000)
                )
            with open("gat_top_level.prof", "w") as prof_f:
                prof_f.write(
                    prof.nested_key_averages(only_top_level=True).table(
                        sort_by="cpu_time_total"
                    )
                )
            prof.print_op_timings(prefix="gat_time_")

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    if args.dataset == "ogbn-products":
        return best_test_acc, avgb / (epoch)
    elif args.dataset == "ogbn-papers100M":
        return avgb / (epoch)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=-1,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='10,10,10')
    argparser.add_argument('--batch-size', type=int, default=512)
    argparser.add_argument('--val-batch-size', type=int, default=512)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--num-workers', type=int, default=8,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--save-pred', type=str, default='')
    argparser.add_argument('--head', type=int, default=4)
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument(
        "--opt_mlp",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    argparser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    argparser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    argparser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    argparser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    argparser.add_argument(
        "--cpu-worker-aff",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="default dataset name",
    )
    argparser.add_argument('--checkpoint', action="store_true", help="checkpoint or not")
    argparser.add_argument("--checkpoint-file", type=str, default="ogbn-papers-mini-gat.model", help="model file")
    argparser.add_argument("--checkpoint-dir", type=str, default=".",
                        help="model dir")

    args = argparser.parse_args()
    
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load data
    data = DglNodePropPredDataset(name=args.dataset)
    #data = DglNodePropPredDataset(name='ogbn-products')
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]
    n_classes = data.num_classes
    if args.dataset == 'ogbn-papers100M':
       ed = graph.edges()
       graph = dgl.add_edges(graph, ed[1], ed[0])
       labels = labels[:, 0].long()
       in_feats = graph.ndata["feat"].shape[1]
       nfeat = graph.ndata["feat"]
       if args.use_bf16:
            nfeat = nfeat.to(th.bfloat16)
       del ed
    elif args.dataset == "ogbn-products":
        nfeat = graph.ndata.pop("feat").to(device)
        if args.use_bf16:
            nfeat = nfeat.to(th.bfloat16)
        labels = labels[:, 0].to(device)
        in_feats = nfeat.shape[1]
        n_classes = (labels.max() + 1).item()

    #print("graph: ", graph)
    #nfeat = graph.ndata.pop('feat').to(device)
    #labels = labels[:, 0].to(device)
    #print("nfeat: ", nfeat.shape)

    print('Total edges before adding self-loop {}'.format(graph.num_edges()))
    graph = graph.remove_self_loop().add_self_loop()
    print('Total edges after adding self-loop {}'.format(graph.num_edges()))

    in_feats = nfeat.shape[1]
    n_classes = (labels.max() + 1).item()

    # Create csr/coo/csc formats before launching sampling processes
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph, args.head

   
    if not args.profile:
        # Run 10 times
        test_accs = []
        epoch_time = []
        if args.dataset == "ogbn-products":
            for i in range(1):
                # test_accs.append(run(args, device, data).cpu().numpy())
                acc, et = run(args, device, data)
                test_accs.append(acc)
                epoch_time.append(et)
                print(
                    "Average test accuracy:", np.mean(test_accs), "±", np.std(test_accs)
                )
                print(
                    "Average epoch time:", np.mean(epoch_time), "±", np.std(epoch_time)
                )
        elif args.dataset == "ogbn-papers100M":
            for i in range(1):
                et = run(args, device, data)
                epoch_time.append(et)
                print(
                    "Average epoch time:", np.mean(epoch_time), "±", np.std(epoch_time)
                )

        #for i in range(1):
        #    test_accs.append(run(args, device, data).cpu().numpy())
        #    print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
    else:
       run(args, device, data)

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
# import gatconv as dglnn
import time
import argparse
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
# from pcl_pytorch_extension.gnn.gat import fused_gat as pcl_gat
from pcl_pytorch_extension.gnn.gat import fused_GAT as pcl_gat
import pcl_pytorch_extension as ppx
import os
import psutil


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
        print("Other Parms: ", (in_feats, in_feats), " ", n_hidden)
        self.layers.append(dglnn.GATConv((in_feats, in_feats), n_hidden, num_heads=num_heads, activation=activation))
        print(in_feats, in_feats, n_hidden, num_heads)
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_hidden,
                                             num_heads=num_heads, activation=activation))
            print(n_hidden*num_heads, n_hidden*num_heads, n_hidden, num_heads)
        self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_classes,
                                         num_heads=num_heads, activation=None))
        

        # print(n_hidden*num_heads, n_hidden*num_heads, n_classes, num_heads)

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
                h = layer(block, (h, h_dst)).flatten(1)
            else:
                h = layer(block, (h, h_dst))
        h = h.mean(1)
        return h.log_softmax(dim=-1)

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
                batch_size=500000, #500000, 
                # batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)#,
                # use_cpu_worker_affinity=True)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)
                if args.use_bf16:
                    x = x.to(th.bfloat16)
                # print("inp nodes", input_nodes.shape, " ", x.shape, " ", output_nodes.shape)
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

def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
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


#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g, num_heads = data

    # print("num_head ========= ", num_heads)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_cpu_worker_affinity=args.cpu_worker_aff
    )

    # Define model and optimizer
    with pcl_gat.pcl_impl(args.use_pcl, args.use_bf16):
        model = GAT(in_feats, args.num_hidden, n_classes, args.num_layers, num_heads, F.relu)
        model = model.to(device)

    if args.use_pcl:
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
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    if args.use_pcl:
        ppx.manual_seed(args.seed)
    # else:
    #     manual_seed(args.seed)
    record_shapes = False
    with th.autograd.profiler.profile(
        enabled=args.profile, use_cuda=(args.gpu > 0), record_shapes=record_shapes
    ) as prof:
        # if prof and args.use_pcl:
        #     ppx.reset_debug_timers()
        if args.use_pcl: ppx.reset_debug_timers()

        for epoch in range(args.num_epochs):
            tic = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                # ppx.reset_debug_timers()
                tic_step = time.time()
                # copy block to gpu
                blocks = [blk.to(device) for blk in blocks]

                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(nfeat, labels, seeds, input_nodes)

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs).to(th.float32)
                loss = F.nll_loss(batch_pred, batch_labels)
                optimizer.zero_grad()
                # t1 = time.time()
                loss.backward()
                # print("loss time: ", time.time()- t1)
                optimizer.step()

                iter_tput.append(len(seeds) / (time.time() - tic_step))
                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                        epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
                # if prof and args.use_pcl:
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
            if not args.profile and epoch % args.eval_every == 0 and epoch != 0:
                eval_acc, test_acc, pred = evaluate(model, g, nfeat, labels, val_nid, test_nid, num_heads, device)
                # eval_acc = accuracy()
                # if args.save_pred: 
                #     np.savetxt(args.save_pred + '%02d' % epoch, pred.argmax(1).cpu().numpy(), '%d')
                print('Eval Acc {:.4f}'.format(eval_acc))
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_test_acc = test_acc
                print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))

    # if prof and args.use_pcl:
        # ppx.print_debug_timers(0)
    if args.use_pcl: ppx.print_debug_timers(0)
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
    return best_test_acc

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
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
        "--use_pcl",
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

    args = argparser.parse_args()
    
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    # load data
    data = DglNodePropPredDataset(name='ogbn-products')
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    graph, labels = data[0]
    nfeat = graph.ndata.pop('feat').to(device)
    labels = labels[:, 0].to(device)

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
        for i in range(1):
            test_accs.append(run(args, device, data).cpu().numpy())
            print('Average test accuracy:', np.mean(test_accs), '±', np.std(test_accs))
    else:
        run(args, device, data)

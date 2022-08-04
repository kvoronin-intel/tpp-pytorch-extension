import argparse
from itertools import chain
from timeit import default_timer
from typing import Callable, Tuple, Union

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils
import model as base_model
from contextlib import contextmanager
from pcl_pytorch_extension.gnn.rgcn_hetero_ogb_mag import fused_rgcn
from pcl_pytorch_extension.gnn.common import gnn_utils
import pcl_pytorch_extension as ppx
import os, time


@contextmanager
def opt_impl(enable=True, use_bf16=False):
    try:
        import dgl

        orig_EntityClassify = base_model.EntityClassify
        orig_RelGraphEmbedding = base_model.RelGraphEmbedding

        try:
            if enable:
                if use_bf16:
                    base_model.EntityClassify = fused_rgcn.OptEntityClassifyBF16
                    base_model.RelGraphEmbedding = fused_rgcn.OptRelGraphEmbedBF16
                else:
                    base_model.EntityClassify = fused_rgcn.OptEntityClassify
                    base_model.RelGraphEmbedding = fused_rgcn.OptRelGraphEmbed
            yield
        finally:
            base_model.EntityClassify = orig_EntityClassify
            base_model.RelGraphEmbedding = orig_RelGraphEmbedding
    except ImportError as e:
        pass


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()


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


def train(
    embedding_layer: nn.Module,
    model: nn.Module,
    device: Union[str, torch.device],
    embedding_optimizer: torch.optim.Optimizer,
    model_optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    labels: torch.Tensor,
    predict_category: str,
    dataloader: dgl.dataloading.NodeDataLoader,
    epoch,
) -> Tuple[float]:
    model.train()

    total_loss = 0
    total_accuracy = 0

    start = default_timer()

    embedding_layer = embedding_layer.to(device)
    model = model.to(device)
    loss_function = loss_function.to(device)

    batch_fwd_time = AverageMeter()
    batch_bwd_time = AverageMeter()
    loss_time = AverageMeter()
    optim_zero_time = AverageMeter()
    optim_step_time = AverageMeter()
    data_time = AverageMeter()
    gather_time = AverageMeter()
    init_time = AverageMeter()

    iter_time = []

    record_shapes = False
    with torch.autograd.profiler.profile(
        enabled=args.profile, use_cuda=False, record_shapes=record_shapes
    ) as prof:
        if args.mlp_profile and args.opt_mlp:
            ppx.reset_debug_timers()
        end = time.time()
        for step, (in_nodes, out_nodes, blocks) in enumerate(dataloader):
            if args.opt_mlp and epoch == 0 and step == 0:
                cores = int(os.environ["OMP_NUM_THREADS"])
                gnn_utils.affinitize_cores(cores, args.num_workers)
            t0 = time.time()
            if step > 0:
                data_time.update(t0 - end)
            t1 = time.time()
            embedding_optimizer.zero_grad(set_to_none=True)
            model_optimizer.zero_grad()
            t2 = time.time()
            if step > 0:
                optim_zero_time.update(t2 - t1)

            in_nodes = {rel: nid.to(device) for rel, nid in in_nodes.items()}
            out_nodes = out_nodes[predict_category].to(device)
            blocks = [block.to(device) for block in blocks]

            batch_labels = labels[out_nodes].to(device)
            t3 = time.time()
            if step > 0:
                init_time.update(t3 - t2)

            embedding = embedding_layer(in_nodes=in_nodes, device=device)
            t4 = time.time()
            if step > 0:
                gather_time.update(t4 - t3)

            logits = model(blocks, embedding)[predict_category]
            t5 = time.time()
            if step > 0:
                batch_fwd_time.update(t5 - t4)

            loss = loss_function(logits, batch_labels)
            t6 = time.time()
            if step > 0:
                loss_time.update(t6 - t5)

            loss.backward()
            t7 = time.time()
            if step > 0:
                batch_bwd_time.update(t7 - t6)

            model_optimizer.step()
            embedding_optimizer.step()
            t8 = time.time()
            if step > 0:
                optim_step_time.update(t8 - t7)
            end = time.time()

            indices = logits.argmax(dim=-1)
            correct = torch.sum(indices == batch_labels)
            accuracy = correct.item() / len(batch_labels)
            total_loss += loss.item()
            total_accuracy += accuracy

            if step > 0 and step % 100 == 0:
                print(
                    "Step {:02d} |"
                    "DL (s) {data_time.val:.3f} ({data_time.avg:.3f}) |"
                    "OptZ (s) {optim_zero_time.val:.4f} ({optim_zero_time.avg:.4f}) |"
                    "Init (s) {init_time.val:.4f} ({init_time.avg:.4f}) |"
                    "GT (s) {gather_time.val:.3f} ({gather_time.avg:.3f}) |"
                    "FWD (s) {batch_fwd_time.val:.3f} ({batch_fwd_time.avg:.3f}) |"
                    "Loss (s) {loss_time.val:.4f} ({loss_time.avg:.4f}) |"
                    "BWD (s) {batch_bwd_time.val:.3f} ({batch_bwd_time.avg:.3f}) |"
                    "OptS (s) {optim_step_time.val:.4f} ({optim_step_time.avg:.4f}) |".format(
                        step,
                        data_time=data_time,
                        optim_zero_time=optim_zero_time,
                        init_time=init_time,
                        gather_time=gather_time,
                        batch_fwd_time=batch_fwd_time,
                        loss_time=loss_time,
                        batch_bwd_time=batch_bwd_time,
                        optim_step_time=optim_step_time,
                    )
                )
            # print('Step {:02d} | Loss {:.4f} | TLoss {:.4f} | Acc {:.4f} | TAcc {:.4f}'.format(
            #  step, loss.item(), total_loss, accuracy, total_accuracy))

    if args.mlp_profile and args.opt_mlp:
        ppx.print_debug_timers(0)

    if prof:
        with open("rgcn.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=record_shapes).table(
                    sort_by="cpu_time_total"
                )
            )
        if ppx.extend_profiler:
            with open("rgcn_nested.prof", "w") as prof_f:
                prof_f.write(
                    prof.nested_key_averages().table(sort_by=None, row_limit=1000)
                )
            with open("rgcn_top_level.prof", "w") as prof_f:
                prof_f.write(
                    prof.nested_key_averages(only_top_level=True).table(
                        sort_by="cpu_time_total"
                    )
                )

    stop = default_timer()
    t = stop - start

    total_loss /= step + 1
    total_accuracy /= step + 1

    return t, total_loss, total_accuracy


def validate(
    embedding_layer: nn.Module,
    model: nn.Module,
    device: Union[str, torch.device],
    inference_mode: str,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    hg: dgl.DGLHeteroGraph,
    labels: torch.Tensor,
    epoch,
    predict_category: str,
    dataloader: dgl.dataloading.NodeDataLoader = None,
    eval_batch_size: int = None,
    eval_num_workers: int = None,
    mask: torch.Tensor = None,
) -> Tuple[float]:
    embedding_layer.eval()
    model.eval()

    start = default_timer()

    embedding_layer = embedding_layer.to(device)
    model = model.to(device)
    loss_function = loss_function.to(device)

    valid_labels = labels[mask].to(device)

    with torch.no_grad():
        if inference_mode == "neighbor_sampler":
            total_loss = 0
            total_accuracy = 0

            for step, (in_nodes, out_nodes, blocks) in enumerate(dataloader):
                if args.opt_mlp and epoch == 0 and step == 0:
                    cores = int(os.environ["OMP_NUM_THREADS"])
                    gnn_utils.affinitize_cores(cores, args.num_workers)
                in_nodes = {rel: nid.to(device) for rel, nid in in_nodes.items()}
                out_nodes = out_nodes[predict_category].to(device)
                blocks = [block.to(device) for block in blocks]

                batch_labels = labels[out_nodes].to(device)

                embedding = embedding_layer(in_nodes=in_nodes, device=device)
                logits = model(blocks, embedding)[predict_category]

                loss = loss_function(logits, batch_labels)

                indices = logits.argmax(dim=-1)
                correct = torch.sum(indices == batch_labels)
                accuracy = correct.item() / len(batch_labels)

                total_loss += loss.item()
                total_accuracy += accuracy

            total_loss /= step + 1
            total_accuracy /= step + 1
        elif inference_mode == "full_neighbor_sampler":
            logits = model.inference(
                hg,
                eval_batch_size,
                eval_num_workers,
                embedding_layer,
                device,
            )[predict_category][mask]

            total_loss = loss_function(logits, valid_labels)

            indices = logits.argmax(dim=-1)
            correct = torch.sum(indices == valid_labels)
            total_accuracy = correct.item() / len(valid_labels)

            total_loss = total_loss.item()
        else:
            embedding = embedding_layer(device=device)
            logits = model(hg, embedding)[predict_category][mask]

            total_loss = loss_function(logits, valid_labels)

            indices = logits.argmax(dim=-1)
            correct = torch.sum(indices == valid_labels)
            total_accuracy = correct.item() / len(valid_labels)

            total_loss = total_loss.item()

    stop = default_timer()
    time = stop - start

    return time, total_loss, total_accuracy


def run(args: argparse.ArgumentParser) -> None:
    torch.manual_seed(args.seed)

    dataset, hg, train_idx, valid_idx, test_idx = utils.process_dataset(
        args.dataset,
        root=args.dataset_root,
    )
    predict_category = dataset.predict_category
    labels = hg.nodes[predict_category].data["labels"]

    training_device = torch.device("cuda" if args.gpu_training else "cpu")
    inference_device = torch.device("cuda" if args.gpu_inference else "cpu")

    inferfence_mode = args.inference_mode

    fanouts = [int(fanout) for fanout in args.fanouts.split(",")]

    train_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        hg,
        {predict_category: train_idx},
        train_sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_cpu_worker_affinity=args.cpu_worker_aff,
        formats=["csc"],
        # persistent_workers=args.cpu_worker_aff,
    )

    if inferfence_mode == "neighbor_sampler":
        valid_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
        valid_dataloader = dgl.dataloading.NodeDataLoader(
            hg,
            {predict_category: valid_idx},
            valid_sampler,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.eval_num_workers,
            use_cpu_worker_affinity=args.cpu_worker_aff,
            formats=["csc"],
            # persistent_workers=args.cpu_worker_aff,
        )

        if args.test_validation:
            test_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
            test_dataloader = dgl.dataloading.NodeDataLoader(
                hg,
                {predict_category: test_idx},
                test_sampler,
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.eval_num_workers,
                use_cpu_worker_affinity=args.cpu_worker_aff,
                formats=["csc"],
                # persistent_workers=args.cpu_worker_aff,
            )
    else:
        valid_dataloader = None

        if args.test_validation:
            test_dataloader = None

    in_feats = hg.nodes[predict_category].data["feat"].shape[-1]
    out_feats = dataset.num_classes

    num_nodes = {}
    node_feats = {}

    for ntype in hg.ntypes:
        num_nodes[ntype] = hg.num_nodes(ntype)
        node_feats[ntype] = hg.nodes[ntype].data.get("feat")

    activations = {"leaky_relu": F.leaky_relu, "relu": F.relu}

    with opt_impl(args.opt_mlp, args.use_bf16):
        embedding_layer = base_model.RelGraphEmbedding(
            hg, in_feats, num_nodes, node_feats
        )
        model = base_model.EntityClassify(
            hg,
            in_feats,
            args.hidden_feats,
            out_feats,
            args.num_bases,
            args.num_layers,
            norm=args.norm,
            layer_norm=args.layer_norm,
            input_dropout=args.input_dropout,
            dropout=args.dropout,
            activation=activations[args.activation],
            self_loop=args.self_loop,
        )
    block(embedding_layer)
    block(model)

    loss_function = nn.CrossEntropyLoss()

    if args.opt_mlp:
        embedding_optimizer = ppx.optim.AdamW(
            embedding_layer.node_embeddings.parameters(), lr=args.embedding_lr
        )
    else:
        embedding_optimizer = torch.optim.SparseAdam(
            embedding_layer.node_embeddings.parameters(), lr=args.embedding_lr
        )

    if args.node_feats_projection:
        all_parameters = chain(
            model.parameters(), embedding_layer.embeddings.parameters()
        )
        model_optimizer = torch.optim.Adam(all_parameters, lr=args.model_lr)
    else:
        if args.opt_mlp:
            model_optimizer = ppx.optim.AdamW(model.parameters(), lr=args.model_lr)
        else:
            model_optimizer = torch.optim.AdamW(model.parameters(), lr=args.model_lr)

    checkpoint = utils.Callback(
        args.early_stopping_patience, args.early_stopping_monitor
    )

    print("## Training started ##")

    if args.opt_mlp:
        ppx.manual_seed(args.seed)

    avg_train_time = []
    avg_val_time = []

    for epoch in range(args.num_epochs):
        train_time, train_loss, train_accuracy = train(
            embedding_layer,
            model,
            training_device,
            embedding_optimizer,
            model_optimizer,
            loss_function,
            labels,
            predict_category,
            train_dataloader,
            epoch,
        )

        if not args.profile:
            valid_time, valid_loss, valid_accuracy = validate(
                embedding_layer,
                model,
                inference_device,
                inferfence_mode,
                loss_function,
                hg,
                labels,
                epoch,
                predict_category=predict_category,
                dataloader=valid_dataloader,
                eval_batch_size=args.eval_batch_size,
                eval_num_workers=args.eval_num_workers,
                mask=valid_idx,
            )

            checkpoint.create(
                epoch,
                train_time,
                valid_time,
                train_loss,
                valid_loss,
                train_accuracy,
                valid_accuracy,
                {"embedding_layer": embedding_layer, "model": model},
            )

            print(
                f"Epoch: {epoch + 1:03} "
                f"Train Loss: {train_loss:.2f} "
                f"Valid Loss: {valid_loss:.2f} "
                f"Train Accuracy: {train_accuracy:.4f} "
                f"Valid Accuracy: {valid_accuracy:.4f} "
                f"Train Epoch Time: {train_time:.2f} "
                f"Valid Epoch Time: {valid_time:.2f}"
            )
            avg_train_time.append(train_time)
            avg_val_time.append(valid_time)

        if checkpoint.should_stop:
            print("## Training finished: early stopping ##")
            print(f"Avg. Training Time: {np.mean(avg_train_time):.2f}")
            print(f"Avg. Validation Time: {np.mean(avg_val_time):.2f}")

            break
        elif epoch >= args.num_epochs - 1:
            print("## Training finished ##")

    if not args.profile:
        print(
            f"Best Epoch: {checkpoint.best_epoch} "
            f"Train Loss: {checkpoint.best_epoch_train_loss:.2f} "
            f"Valid Loss: {checkpoint.best_epoch_valid_loss:.2f} "
            f"Train Accuracy: {checkpoint.best_epoch_train_accuracy:.4f} "
            f"Valid Accuracy: {checkpoint.best_epoch_valid_accuracy:.4f}"
        )

    if args.test_validation:
        print("## Test data validation ##")

        embedding_layer.load_state_dict(
            checkpoint.best_epoch_model_parameters["embedding_layer"]
        )
        model.load_state_dict(checkpoint.best_epoch_model_parameters["model"])

        test_time, test_loss, test_accuracy = validate(
            embedding_layer,
            model,
            inference_device,
            inferfence_mode,
            loss_function,
            hg,
            labels,
            epoch,
            predict_category=predict_category,
            dataloader=test_dataloader,
            eval_batch_size=args.eval_batch_size,
            eval_num_workers=args.eval_num_workers,
            mask=test_idx,
        )

        print(
            f"Test Loss: {test_loss:.2f} "
            f"Test Accuracy: {test_accuracy:.4f} "
            f"Test Epoch Time: {test_time:.2f}"
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("RGCN")

    argparser.add_argument("--gpu-training", dest="gpu_training", action="store_true")
    argparser.add_argument(
        "--no-gpu-training", dest="gpu_training", action="store_false"
    )
    argparser.set_defaults(gpu_training=True)
    argparser.add_argument("--gpu-inference", dest="gpu_inference", action="store_true")
    argparser.add_argument(
        "--no-gpu-inference", dest="gpu_inference", action="store_false"
    )
    argparser.set_defaults(gpu_inference=True)
    argparser.add_argument(
        "--inference-mode",
        default="neighbor_sampler",
        type=str,
        choices=["neighbor_sampler", "full_neighbor_sampler", "full_graph"],
    )
    argparser.add_argument(
        "--dataset", default="ogbn-mag", type=str, choices=["ogbn-mag"]
    )
    argparser.add_argument("--dataset-root", default="dataset", type=str)
    argparser.add_argument("--num-epochs", default=500, type=int)
    argparser.add_argument("--embedding-lr", default=0.01, type=float)
    argparser.add_argument("--model-lr", default=0.01, type=float)
    argparser.add_argument(
        "--node-feats-projection", dest="node_feats_projection", action="store_true"
    )
    argparser.add_argument(
        "--no-node-feats-projection", dest="node_feats_projection", action="store_false"
    )
    argparser.set_defaults(node_feats_projection=False)
    argparser.add_argument("--hidden-feats", default=64, type=int)
    argparser.add_argument("--num-bases", default=2, type=int)
    argparser.add_argument("--num-layers", default=2, type=int)
    argparser.add_argument(
        "--norm", default="right", type=str, choices=["both", "none", "right"]
    )
    argparser.add_argument("--layer-norm", dest="layer_norm", action="store_true")
    argparser.add_argument("--no-layer-norm", dest="layer_norm", action="store_false")
    argparser.set_defaults(layer_norm=False)
    argparser.add_argument("--input-dropout", default=0.1, type=float)
    argparser.add_argument("--dropout", default=0.5, type=float)
    argparser.add_argument(
        "--activation", default="relu", type=str, choices=["leaky_relu", "relu"]
    )
    argparser.add_argument("--self-loop", dest="self_loop", action="store_true")
    argparser.add_argument("--no-self-loop", dest="self_loop", action="store_false")
    argparser.set_defaults(self_loop=True)
    argparser.add_argument("--fanouts", default="25,20", type=str)
    argparser.add_argument("--batch-size", default=1024, type=int)
    argparser.add_argument("--eval-batch-size", default=1024, type=int)
    argparser.add_argument("--num-workers", default=4, type=int)
    argparser.add_argument("--eval-num-workers", default=4, type=int)
    argparser.add_argument("--early-stopping-patience", default=10, type=int)
    argparser.add_argument(
        "--early-stopping-monitor",
        default="loss",
        type=str,
        choices=["accuracy", "loss"],
    )
    argparser.add_argument(
        "--test-validation", dest="test_validation", action="store_true"
    )
    argparser.add_argument(
        "--no-test-validation", dest="test_validation", action="store_false"
    )
    argparser.set_defaults(test_validation=True)
    argparser.add_argument("--seed", default=13, type=int)
    argparser.add_argument(
        "--opt_mlp",
        action="store_true",
        help="Whether to use PCL Fused impl when available",
    )
    argparser.add_argument(
        "--use_bf16", action="store_true", help="Whether to use BF16 datatype"
    )
    argparser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile or not",
    )
    argparser.add_argument(
        "--mlp-profile",
        action="store_true",
        help="Whether to profile MLP or not",
    )
    argparser.add_argument(
        "--cpu-worker-aff",
        action="store_true",
        help="Whether to affinitize DL workers or not",
    )

    args = argparser.parse_args()

    run(args)

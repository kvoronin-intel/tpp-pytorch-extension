"""RGCN layer implementation"""
from collections import defaultdict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from tpp_pytorch_extension.utils.blocked_layout import (
    BlockedParameter,
    BlockedModule,
    BlockedTensor,
)
from tpp_pytorch_extension.utils import blocked_layout, xsmm
from tpp_pytorch_extension._C import _fused_rgcn as fused_rgcn_cpp
from tpp_pytorch_extension._C import _xsmm as xsmm_cpp
from dgl.nn.pytorch.conv.graphconv import *

import time
import tqdm
import numpy as np

th.autograd.set_detect_anomaly(False)

USE_BF16_ACT_PARAMS = False


class RGCNNormFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, align, norm_type, *inputs):
        out, norm = fused_rgcn_cpp.fused_rgcn_norm_fwd(align, norm_type, inputs)

        ctx.save_for_backward(norm)
        ctx.align = align
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors

        grad_input = fused_rgcn_cpp.fused_rgcn_norm_bwd(ctx.align, inputs)
        return (None, None, None, grad_input)


class RGCNMLPFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, align, norm_type, *inputs):
        (_, inp, wt) = inputs
        out, norm = fused_rgcn_cpp.fused_rgcn_mlp_fwd(align, norm_type, inputs)

        ctx.save_for_backward(inp, wt, norm)
        ctx.align = align
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_input, grad_wt = fused_rgcn_cpp.fused_rgcn_mlp_bwd(ctx.align, inputs)
        return (None, None, None, grad_input, grad_wt)


class RGCNGEMMFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, align, *inputs):
        (inp, wt) = inputs
        out = fused_rgcn_cpp.fused_rgcn_gemm_fwd(align, inputs)

        ctx.save_for_backward(inp, wt)
        ctx.align = align
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors

        grad_input, grad_wt = fused_rgcn_cpp.fused_rgcn_gemm_bwd(ctx.align, inputs)
        return (None, grad_input, grad_wt)


class RGCNEltwiseFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, self_loop, align, bias, p, act, training, *inputs):

        if self_loop:
            (_, inp_dst, wt, _) = inputs

        out, act_mask, dp_mask = fused_rgcn_cpp.fused_rgcn_eltw_fwd(
            self_loop, align, bias, p, act, training, inputs
        )

        if act == "None":
            act_mask = th.tensor([], dtype=th.short)
        if p == 0.0:
            dp_mask = th.tensor([], dtype=th.short)

        if self_loop:
            ctx.save_for_backward(inp_dst, wt, act_mask, dp_mask)
        else:
            ctx.save_for_backward(act_mask, dp_mask)

        ctx.act = act
        ctx.p = p
        ctx.align = align
        ctx.bias = bias
        ctx.self_loop = self_loop
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors

        if ctx.self_loop:
            if ctx.bias:
                (
                    grad_input,
                    grad_inp_dst,
                    grad_wt,
                    grad_bias,
                ) = fused_rgcn_cpp.fused_rgcn_eltw_bwd(
                    ctx.self_loop, ctx.align, ctx.bias, ctx.p, ctx.act, inputs
                )
            else:
                grad_input, grad_inp_dst, grad_wt = fused_rgcn_cpp.fused_rgcn_eltw_bwd(
                    ctx.self_loop, ctx.align, ctx.bias, ctx.p, ctx.act, inputs
                )
        else:
            if ctx.bias:
                grad_input, grad_bias = fused_rgcn_cpp.fused_rgcn_eltw_bwd(
                    ctx.self_loop, ctx.align, ctx.bias, ctx.p, ctx.act, inputs
                )
            else:
                grad_input = fused_rgcn_cpp.fused_rgcn_eltw_bwd(
                    ctx.self_loop, ctx.align, ctx.p, ctx.bias, ctx.act, inputs
                )

        if ctx.self_loop:
            if ctx.bias:
                return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    grad_input,
                    grad_inp_dst,
                    grad_wt,
                    grad_bias,
                )
            else:
                return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    grad_input,
                    grad_inp_dst,
                    grad_wt,
                )
        else:
            if ctx.bias:
                return (None, None, None, None, None, None, grad_input, grad_bias)
            else:
                return (None, None, None, None, None, None, grad_input)


class DropoutFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, align, p, training, *inps):
        out, dp_mask = fused_rgcn_cpp.rgcn_dropout_fwd(align, p, training, inps)

        if p == 0.0:
            dp_mask = th.tensor([], dtype=th.short)

        ctx.save_for_backward(dp_mask)
        ctx.p = p
        ctx.align = align
        return out

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors
        grad_inp = fused_rgcn_cpp.rgcn_dropout_bwd(ctx.align, ctx.p, inputs)
        return (None, None, None, grad_inp)


class OptGraphConv(BlockedModule):
    def __init__(
        self,
        in_feat,
        out_feat,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(OptGraphConv, self).__init__()

        if norm not in ("none", "both", "right", "left"):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm)
            )
        self._in_feats = in_feat
        self._out_feats = out_feat
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.align = 32

        self.bc = self._in_feats
        self.bk = self._out_feats

        for cbf in [32, 16]:
            if self._in_feats % cbf == 0:
                self.bc = cbf
                break

        for kbf in [32, 16]:
            if self._out_feats % kbf == 0:
                self.bk = kbf
                break

        if weight:
            self.weight = BlockedParameter(th.Tensor(self._in_feats, self._out_feats))
            self.weight.set_blocking_param(
                (
                    [self.bc, self.bk],
                    [2, 0, 1, 3],
                )
            )
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = BlockedParameter(th.Tensor(self._out_feats))
        else:
            self.register_parameter("bias", None)

        if not USE_BF16_ACT_PARAMS:
            self.use_bf16 = False
        self.reset_parameters()

        self._activation = activation
        self.blocked_weight_signature = blocked_layout.get_blocking_signature(
            "CK", "KCCK"
        )

    def maybe_block_params(self):
        if self.weight is not None:
            self.weight.block()

    def reset_parameters(self):
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            aggregate_fn = fn.copy_src("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.

            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self.use_bf16:
                feat_src = feat_src.to(th.bfloat16)
                feat_dst = feat_dst.to(th.bfloat16)

            if self._norm in ["left", "both"]:
                N = feat_src.size(0)
                align = self.align if (N > self.align or N == 0) else N
                degs = graph.out_degrees().float().clamp(min=1)
                inputs = [feat_src, degs]
                feat_src = RGCNNormFunction.apply(align, self._norm, *inputs)

            self.maybe_block_params()

            if weight is None:
                weight = self.weight

            # aggregate first then mult W
            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]

            N = rst.size(0)
            align = self.align if (N > self.align or N == 0) else N

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().float().clamp(min=1)

            if weight is not None:
                wt = self.get_blocked_tensor(
                    weight, self.blocked_weight_signature, [self.bc, self.bk]
                )
                if self.use_bf16:
                    wt = wt.to(th.bfloat16)
                    vs = wt.shape
                    vs = [vs[0], vs[1], vs[2] // 2, 2, vs[3]]
                    wt = wt.view(vs).permute([0, 1, 2, 4, 3]).contiguous()

                    rst = rst.to(th.bfloat16)
                inputs = [degs, rst, wt]
                rst = RGCNMLPFunction.apply(align, self._norm, *inputs)
            else:
                if self.use_bf16:
                    rst = rst.to(bfloat16)
                inputs = [degs, rst]
                rst = RGCNNormFunction.apply(align, self._norm, *inputs)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class OptGraphConvBF16(OptGraphConv):
    def __init__(
        self,
        in_feat,
        out_feat,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):

        super(OptGraphConvBF16, self).__init__(
            in_feat, out_feat, norm, weight, bias, activation, allow_zero_in_degree
        )

        self.use_bf16 = True


class OptRelGraphEmbed(BlockedModule):
    def __init__(
        self, g, embed_size, num_nodes, node_feats, node_feats_projection=False
    ):

        super(OptRelGraphEmbed, self).__init__()
        self.g = g
        self.node_feats = node_feats
        self.node_feats_projection = node_feats_projection
        self.node_embeddings = nn.ModuleDict()

        if not USE_BF16_ACT_PARAMS:
            self.use_bf16 = False

        if node_feats_projection:
            self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            if node_feats[ntype] is None:
                node_embedding = nn.Embedding(num_nodes[ntype], embed_size, sparse=True)
                node_embedding.weight = BlockedParameter(
                    th.Tensor(
                        node_embedding.weight.shape[0], node_embedding.weight.shape[1]
                    )
                )
                node_embedding.weight.set_blocking_param((None, None, th.float32))
                nn.init.uniform_(node_embedding.weight, -1, 1)
                self.node_embeddings[ntype] = node_embedding
            elif node_feats[ntype] is not None and node_feats_projection:
                input_embedding_size = node_feats[ntype].shape[-1]
                embed = BlockedParameter(
                    th.Tensor(input_embedding_size, self.embed_size)
                )
                nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain("relu"))
                self.embeds[ntype] = embed

    def maybe_block_params(self):
        for ntype in self.g.ntypes:
            if self.node_feats[ntype] is None:
                self.node_embeddings[ntype].weight.block()
            if self.node_feats[ntype] is not None and self.node_feats_projection:
                self.embeds[ntype].weight.block()

    def forward(self, in_nodes, device):

        if in_nodes is not None:
            ntypes = [ntype for ntype in in_nodes.keys()]
            nids = [nid for nid in in_nodes.values()]
        else:
            ntypes = self._hg.ntypes
            nids = [self.g.nodes(ntype) for ntype in ntypes]

        x = {}

        for ntype, nid in zip(ntypes, nids):
            if self.node_feats[ntype] is None:
                x[ntype] = self.node_embeddings[ntype](nid)
            else:
                if self.node_feats_projection:
                    x[ntype] = self.node_feats[ntype][nid] @ self.embeddings[ntype]
                else:
                    x[ntype] = self.node_feats[ntype][nid]
        return x


class OptRelGraphConvLayer(BlockedModule):
    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        num_bases,
        norm="right",
        weight=True,
        bias=True,
        activation=None,
        dropout=0.0,
        self_loop=False,
    ):
        super(OptRelGraphConvLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        self.bias = bias
        self.activation = None
        if activation == F.relu:
            self.activation = "relu"
        elif activation == F.leaky_relu:
            self.activation = "leaky_relu"
        elif activation is None:
            self.activation = ""
        self.self_loop = self_loop

        if USE_BF16_ACT_PARAMS:
            if self.use_bf16:
                self.conv = dglnn.HeteroGraphConv(
                    {
                        rel: OptGraphConvBF16(
                            in_feat, out_feat, norm=norm, weight=False, bias=False
                        )
                        for rel in rel_names
                    }
                )
        else:
            self.conv = dglnn.HeteroGraphConv(
                {
                    rel: OptGraphConv(
                        in_feat, out_feat, norm=norm, weight=False, bias=False
                    )
                    for rel in rel_names
                }
            )

        self.align = 32
        self.bc = self.in_feat
        self.bk = self.out_feat

        for cbf in [32, 16]:
            if self.in_feat % cbf == 0:
                self.bc = cbf
                break

        for kbf in [32, 16]:
            if self.out_feat % kbf == 0:
                self.bk = kbf
                break

        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names)
                )
            else:
                self.weight = BlockedParameter(len(self.rel_names), in_feat, out_feat)
                outs = len(self.rel_names)
                for i in range(outs):
                    self.weight[i].set_blocking_param(
                        (
                            [self.bc, self.bk],
                            [2, 0, 1, 3],
                        )
                    )
                    nn.init.xavier_uniform_(
                        self.weight[i], gain=nn.init.calculate_gain("relu")
                    )
        # bias
        if bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = BlockedParameter(th.Tensor(in_feat, out_feat))
            self.loop_weight.set_blocking_param(
                (
                    [self.bc, self.bk],
                    [2, 0, 1, 3],
                )
            )
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = dropout
        self.nn_dropout = nn.Dropout(dropout)
        if not USE_BF16_ACT_PARAMS:
            self.use_bf16 = False

    def maybe_block_params(self):
        self.loop_weight.block()

    def forward(self, g, inputs):
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                self.rel_names[i]: {"weight": w.squeeze(dim=0)}
                for i, w in enumerate(th.split(weight, 1, dim=0))
            }
        else:
            wdict = {}

        if self.self_loop:
            self.maybe_block_params()
            if g.is_block:
                inputs_dst = {k: v[: g.num_dst_nodes(k)] for k, v in inputs.items()}
            else:
                inputs_dst = inputs
        else:
            inputs_dst = None

        # apply graphconv
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        rdict = {}

        for ntype, h in hs.items():
            N = h.size(0)
            rst_tpp = None
            if inputs_dst is not None:
                inps = [h, inputs_dst[ntype], self.loop_weight]
            else:
                inps = h
            if self.use_bf16:
                inps = [i.to(th.bfloat16) if i.is_floating_point() else i for i in inps]
            if self.bias:
                inps.append(self.h_bias)

            align = self.align if (N > self.align or N == 0) else N

            rst_tpp = RGCNEltwiseFunction.apply(
                self.self_loop,
                align,
                self.bias,
                self.dropout,
                self.activation,
                self.training,
                *inps
            )

            rdict[ntype] = rst_tpp

        return rdict


class OptRelGraphConvLayerBF16(OptRelGraphConvLayer):
    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        num_bases,
        norm="right",
        weight=True,
        bias=True,
        activation=None,
        dropout=0.0,
        self_loop=False,
    ):
        self.use_bf16 = True
        global USE_BF16_ACT_PARAMS
        USE_BF16_ACT_PARAMS = True

        super(OptRelGraphConvLayerBF16, self).__init__(
            in_feat,
            out_feat,
            rel_names,
            num_bases,
            norm,
            weight,
            bias,
            activation,
            dropout,
            self_loop,
        )

        # weight for self loop
        if self.self_loop:
            self.loop_weight.set_blocking_param(
                (
                    [[self.bc // 2, 2], self.bk],
                    [3, 0, 1, 4, 2],
                    th.bfloat16,
                )
            )
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )


class OptRelGraphEmbedBF16(OptRelGraphEmbed):
    def __init__(
        self, g, embed_size, num_nodes, node_feats, node_feats_projection=False
    ):

        self.use_bf16 = True
        global USE_BF16_ACT_PARAMS
        USE_BF16_ACT_PARAMS = True

        super(OptRelGraphEmbedBF16, self).__init__(g, embed_size, num_nodes, node_feats)

        for ntype in self.g.ntypes:
            if self.node_feats[ntype] is None:
                self.node_embeddings[ntype].weight.set_blocking_param(
                    (
                        None,
                        None,
                        th.bfloat16,
                    )
                )
            if self.node_feats[ntype] is None and node_feats_projection:
                self.embeds[ntype].weight.set_blocking_param(
                    (
                        None,
                        None,
                        th.bfloat16,
                    )
                )


class OptEntityClassify(BlockedModule):
    def __init__(
        self,
        g,
        in_dim,
        h_dim,
        out_dim,
        num_bases,
        num_layers,
        norm="right",
        layer_norm=False,
        input_dropout=0,
        dropout=0,
        activation=None,
        self_loop=False,
    ):
        super(OptEntityClassify, self).__init__()

        self.g = g
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = sorted(list(set(g.etypes)))
        self._input_dropout = nn.Dropout(input_dropout)
        if num_bases < 0 or num_bases > len(self.rel_names):
            self.num_bases = len(self.rel_names)
        else:
            self.num_bases = num_bases
        self.num_layers = num_layers
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.self_loop = self_loop
        self.align = 32

        self.layers = nn.ModuleList()
        if USE_BF16_ACT_PARAMS:
            assert self.use_bf16 == True
            # i2h
            self.layers.append(
                OptRelGraphConvLayerBF16(
                    self.in_dim,
                    self.h_dim,
                    self.rel_names,
                    self.num_bases,
                    norm=norm,
                    activation=activation,
                    self_loop=self.self_loop,
                    dropout=self.dropout,
                )
            )
            # h2h
            for _ in range(1, self.num_layers - 1):
                self.layers.append(
                    OptRelGraphConvLayerBF16(
                        self.h_dim,
                        self.h_dim,
                        self.rel_names,
                        self.num_bases,
                        norm=norm,
                        activation=activation,
                        self_loop=self.self_loop,
                        dropout=self.dropout,
                    )
                )
            # h2o
            self.layers.append(
                OptRelGraphConvLayerBF16(
                    self.h_dim,
                    self.out_dim,
                    self.rel_names,
                    self.num_bases,
                    norm=norm,
                    activation=None,
                    self_loop=self.self_loop,
                )
            )
        else:
            # i2h
            self.layers.append(
                OptRelGraphConvLayer(
                    self.in_dim,
                    self.h_dim,
                    self.rel_names,
                    self.num_bases,
                    norm=norm,
                    activation=activation,
                    self_loop=self.self_loop,
                    dropout=self.dropout,
                )
            )
            # h2h
            for _ in range(1, self.num_layers - 1):
                self.layers.append(
                    OptRelGraphConvLayer(
                        self.h_dim,
                        self.h_dim,
                        self.rel_names,
                        self.num_bases,
                        norm=norm,
                        activation=activation,
                        self_loop=self.self_loop,
                        dropout=self.dropout,
                    )
                )
            # h2o
            self.layers.append(
                OptRelGraphConvLayer(
                    self.h_dim,
                    self.out_dim,
                    self.rel_names,
                    self.num_bases,
                    norm=norm,
                    activation=None,
                    self_loop=self.self_loop,
                )
            )
        if not USE_BF16_ACT_PARAMS:
            self.use_bf16 = False

    def forward(self, g, inputs):
        if self.use_bf16:
            for nt, nh in inputs.items():
                nh = nh.to(th.bfloat16)
                inputs.update({nt: nh})

        # minibatch training
        # h = {ntype: self._input_dropout(h) for ntype, h in inputs.items()}
        hs = {}
        for ntype, h in inputs.items():
            N = h.size(0)
            align = self.align if (N > self.align or N == 0) else N
            inps = [h]
            hs[ntype] = DropoutFunction.apply(
                align, self.input_dropout, self.training, *inps
            )

        if isinstance(g, list):
            for _, (layer, block) in enumerate(zip(self.layers, g)):
                hs = layer(block, hs)
        else:
            for _, layer in enumerate(self.layers):
                hs = layer(g, hs)
        return hs

    def inference(self, g, batch_size, device, num_workers, embedding_layer):
        for l, layer in enumerate(self.layers):
            y = {
                k: th.Tensor(
                    g.number_of_nodes(k),
                    self.h_dim if l != len(self.layers) - 1 else self.out_dim,
                )
                for k in g.ntypes
            }

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                {k: g.number_of_nodes(k) for k in g.ntypes},
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
                use_cpu_worker_affinity=True,
            )

            if l < self._num_layers - 1:
                y = {
                    ntype: torch.zeros(g.num_nodes(ntype), self.h_dim)
                    for ntype in g.ntypes
                }
            else:
                y = {
                    ntype: torch.zeros(g.num_nodes(ntype), self.out_dim)
                    for ntype in g.ntypes
                }

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0]
                input_nodes = {rel: nid for rel, nid in input_nodes.items()}
                out_nodes = {rel: nid for rel, nid in output_nodes.items()}

                if l == 0:
                    h = embedding_layer(input_nodes=input_nodes)
                else:
                    h = {k: x[k][input_nodes[k]] for k in g.ntypes}

                h = layer(block, h)

                for k in h.keys():
                    y[k][output_nodes[k]] = h[k].cpu()

            x = y
        return y


class OptEntityClassifyBF16(OptEntityClassify):
    def __init__(
        self,
        g,
        in_dim,
        h_dim,
        out_dim,
        num_bases,
        num_layers,
        norm="right",
        layer_norm=False,
        input_dropout=0,
        dropout=0,
        activation=None,
        self_loop=False,
    ):
        self.use_bf16 = True
        global USE_BF16_ACT_PARAMS
        USE_BF16_ACT_PARAMS = True

        super(OptEntityClassifyBF16, self).__init__(
            g,
            in_dim,
            h_dim,
            out_dim,
            num_bases,
            num_layers,
            norm,
            layer_norm,
            input_dropout,
            dropout,
            activation,
            self_loop,
        )


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()

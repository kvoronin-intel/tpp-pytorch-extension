import torch
from pcl_pytorch_extension._C import _gnn_utils as gnn_utils_cpp


def affinitize_cores(nthreads, nworkers):
    gnn_utils_cpp.affinitize_cores(nthreads, nworkers)


def gather_features(nfeat, indices):
    N = nfeat.shape[0]
    align = 32 if N >= 32 or N==0 else N
    inputs = [nfeat, indices]

    out = gnn_utils_cpp.gather_features(align, inputs)

    return out

def scatter_features(feat_src, indices, feat_dst):
    N = feat_src.shape[0]
    align = 32 if N >= 32 or N==0 else N
    inputs = [feat_src, indices, feat_dst]

    gnn_utils_cpp.scatter_features(align, inputs)

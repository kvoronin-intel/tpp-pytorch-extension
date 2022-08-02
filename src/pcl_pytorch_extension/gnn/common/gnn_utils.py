import torch
from pcl_pytorch_extension._C import _gnn_utils as gnn_utils_cpp


def affinitize_cores(nthreads, nworkers):
    gnn_utils_cpp.affinitize_cores(nthreads, nworkers)


def gather_features(nfeat, indices):
    N = nfeat.shape[1]
    # align = 32 if N >= 32 or N==0 else N
    align = [50, 32]
    falign = N
    for a in align:
        if N >= a and N % a == 0 or N == 0:
            falign = a
            break

    inputs = [nfeat, indices]

    out = gnn_utils_cpp.gather_features(falign, inputs)

    return out

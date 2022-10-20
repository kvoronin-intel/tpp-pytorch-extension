import torch
from torch import nn
from torch.autograd import Function

from pcl_pytorch_extension._C import _embedding as embedding_cpp

# from embedding_cpp import bf16_update
torch_embedding = torch.embedding


def pcl_embedding(weight, input, padding_idx, scale_grad_by_freq, sparse=False):
    if (
        sparse
        and padding_idx == -1
        and scale_grad_by_freq == False
        and weight.device == torch.device("cpu")
    ):  # and weight.dtype == torch.float32:
        N = input.size(0)
        alignN = 32 if (N > 32 or N == 0) else N
        inputs = [weight, input.contiguous()]
        ret = EmbeddingFunction.apply(alignN, *inputs)
    else:
        ret = torch_embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)

    return ret


torch.embedding = pcl_embedding
print("Using PCL Embedding Implementation")


class EmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, alignN, *inputs):
        (weight, input) = inputs
        ctx.save_for_backward(weight, input)
        output = embedding_cpp.emb_fwd(alignN, inputs)
        # breakpoint()
        return output

    @staticmethod
    def backward(ctx, *grad_outs):
        inputs = list(grad_outs)
        inputs += ctx.saved_tensors

        grad_weight = embedding_cpp.emb_bwd(inputs)
        return (None, grad_weight, None)

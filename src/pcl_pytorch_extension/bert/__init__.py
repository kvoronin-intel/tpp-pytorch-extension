from . import fused_bert_unpad
from . import fused_bert
from pcl_pytorch_extension import manual_seed
from pcl_pytorch_extension import reset_debug_timers
from pcl_pytorch_extension import print_debug_timers
from pcl_pytorch_extension import print_debug_thread_imbalance
from pcl_pytorch_extension.optim import AdamW
from pcl_pytorch_extension.optim import Lamb
from pcl_pytorch_extension.optim import DistLamb
from pcl_pytorch_extension.optim import clip_grad_norm_
from pcl_pytorch_extension.utils.blocked_layout import block_model_params as block
from contextlib import contextmanager


@contextmanager
def pcl_impl(enable=True, use_low_prec=False, use_unpad=True, use_bf8=False):
    if use_unpad == True:
        with fused_bert_unpad.pcl_impl(enable, use_low_prec, use_bf8):
            yield
    else:
        if use_low_prec and use_bf8:
            raise NotImplementedError("BF8 is only supported with unpad")
        with fused_bert.pcl_impl(enable, use_low_prec):
            yield


def set_rnd_seed(seed):
    manual_seed(seed)

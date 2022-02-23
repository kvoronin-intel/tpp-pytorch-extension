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
def pcl_impl(enable=True, use_bf16=False, use_unpad=True):
    if use_unpad == True:
        with fused_bert_unpad.pcl_impl(enable, use_bf16):
            yield
    else:
        with fused_bert.pcl_impl(enable, use_bf16):
            yield


def set_rnd_seed(seed):
    manual_seed(seed)

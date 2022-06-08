from . import batchnorm
from . import conv
from . import bottleneck

from pcl_pytorch_extension.utils.blocked_layout import block_model_params as block
from contextlib import contextmanager

@contextmanager
def pcl_impl(enable=True, use_bf16=False):
    with batchnorm.pcl_impl(enable, use_bf16):
        yield

def set_rnd_seed(seed):
    manual_seed(seed)
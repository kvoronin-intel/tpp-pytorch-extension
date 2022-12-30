from . import batchnorm
from . import conv
from . import bottleneck

from tpp_pytorch_extension.utils.blocked_layout import block_model_params as block
from tpp_pytorch_extension.utils.xsmm import get_vnni_blocking

from contextlib import contextmanager

@contextmanager
def pcl_impl(enable=True, use_bf16=False):
    with batchnorm.pcl_impl(enable, use_bf16):
        yield

def set_rnd_seed(seed):
    manual_seed(seed)

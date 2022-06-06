import torch

from . import _C

from .resnet import bottleneck, batchnorm, conv

try:
    from .utils import extend_profiler
except:
    extend_profiler = None

from .utils.xsmm import manual_seed
from .utils import blocked_layout
from .utils import bfloat8
from . import optim
from . import embedding

# from . import fused_bert


def reset_debug_timers():
    _C.reset_debug_timers()


def print_debug_timers(tid=0):
    _C.print_debug_timers(tid)


def print_debug_thread_imbalance():
    _C.print_debug_thread_imbalance()


reset_debug_timers()

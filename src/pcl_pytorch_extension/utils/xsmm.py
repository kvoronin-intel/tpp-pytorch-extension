import time
from pcl_pytorch_extension._C import _xsmm as xsmm_cpp

def manual_seed(seed):
    xsmm_cpp.manual_seed(seed)

def set_rng_state(new_state):
    raise NotImplemented
   
def get_rng_state():
    raise NotImplemented

# initialize libxsmm library and random number generator
xsmm_cpp.init_libxsmm()


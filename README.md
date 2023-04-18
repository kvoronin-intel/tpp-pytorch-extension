Intel(R) Tensor Processing Primitives extension for PyTorch\*
=============================================================
*Copyright (c) Intel corp.*

# Pre-requisite
gcc v8.3.0 or higher

# Installation
Setup conda environment using `utils/setup_conda.sh`

```bash
# Create new conda env 
# It creates an env.sh script for activating conda env
$bash utils/setup_conda.sh [-p <conda_install_path>]
```

Install the extension:
```
# Source the env.sh and install the extension
$source env.sh
$git submodule update --init
# Specifically for Resnet-50 training:
$export LIBXSMM_ROOT=<path_to_libxsmm>
$export LIBXSMMROOT=<path_to_libxsmm> # required in this spelling for building LIBXSMM-DNN
$export LIBXSMM_DNN_ROOT=<path_to_libxsmm_dnn>
$python setup.py install
# Note: if for Resnet-50 training build fails with a complaint about #include <pybind11/pybind11.h>
# This is likely because the installed torch does not have pybind11 as a third-party dependence installed
# In this case one needs to add into setup.py a path to pybind11.h (likely, $CONDA_INSTALL_DIR/envs/<env_name>/lib/python3.8/site-packages/pybind11/include/)
```

# For multi-node runs:
(Optional) install torch_ccl module:
```bash
$bash utils/install_torch_ccl.sh
```

(Optional) install torch_mpi module (requires MPI compiler in PATH):
```bash
$bash utils/install_torch_mpi.sh
```

# Examples
- [BERT SQuAD Fine-tuning](examples/bert/squad/README.txt)
- [BERT MLPerf pre-training](examples/bert/pretrain_mlperf/README.txt)
- [ResNet-50 v1.5 training](examples/resnet/README.txt)


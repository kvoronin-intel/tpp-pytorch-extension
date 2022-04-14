PCL PyTorch Extension for blocked layout and TPP based optimized custom Ops
=================================================================================
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
$python setup.py install
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

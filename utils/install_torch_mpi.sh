#!/bin/bash

if ! command -v 'mpicxx' >& /dev/null ; then
  echo "MPI compiler (mpicxx) needs to be in PATH for torch-mpi to install"
  exit 1
fi
pt_version=$(python -c "import torch; print(torch.__version__)" 2> /dev/null)
if [ "x$pt_version" == "x" ] ; then
  echo "Can't find pytorch version, need PyTorch 1.9 or higher..."
  exit 1
fi
branch=$(echo $pt_version | tr "." " " | awk '{print "v" $1 "." $2}')

unset CC
unset CXX
pip install git+https://github.com/intel-sandbox/torch-mpi.git@$branch



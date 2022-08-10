#!/bin/bash

export LIBXSMM_ROOT=/nfs_home/ramanara/libxsmm_avx3/libxsmm_03_06_22/libxsmm/


pip uninstall pcl-pytorch-extension -y

cd ../../../

python setup.py install

cd examples/gnn/gat

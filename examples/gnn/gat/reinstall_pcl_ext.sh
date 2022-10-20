#!/bin/bash


pip uninstall pcl-pytorch-extension -y
pip uninstall pcl-pytorch-extension -y

cd ../../../

python setup.py install

cd examples/gnn/gat

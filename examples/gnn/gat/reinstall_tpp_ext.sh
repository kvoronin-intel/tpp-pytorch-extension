#!/bin/bash


pip uninstall tpp-pytorch-extension -y
pip uninstall tpp-pytorch-extension -y

cd ../../../

python setup.py install

cd examples/gnn/gat

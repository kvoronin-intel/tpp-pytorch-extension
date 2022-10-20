
This example trains GraphSAGE model on OGBN-Products and OGBN-Papers100M on CPUs. It uses the optimizations in DGL as well as those in this extension for the MLP part of GNN training. 

Install conda env and activate it as described in (TODO link)

To recompile the extension:

$bash ./recompile_ext.sh

Training the model with OGBN-Products
=====================================

For FP32 training

To run baseline

$bash ./run.sh ogbn-products

To run optimized version

$bash ./run.sh ogbn-products --opt_mlp

For BF16 training (works only with optimized version)

$bash ./run.sh ogbn-products --opt_mlp --use_bf16

Training the model with OGBN-Papers100M
=======================================

For FP32 training

To run baseline

$bash ./run.sh ogbn-papers100M

To run optimized version

$bash ./run.sh ogbn-papers100M --opt_mlp

For BF16 training (works only with optimized version)

$bash ./run.sh ogbn-papers100M --opt_mlp --use_bf16




# Optimized GAT


## Command to run GAT

Follow this command to run with cpu affinity
 - To get the benefit of cpu affinitization in the code and load the data fast use `--cpu-worker-aff` flag in the command and assign dl = number_of_threads (e.g. dl=4) otherwise use dl=0 and `don't use the --cpu-worker-aff` flag.
 
```
    dl=4
    mdir_ss="/path/to/checkpoint/directory/"
    mkdir $mdir_ss
    ./run_gat_all.sh  $dl $mdir_ss
``` 

- To activate the optmized fp32 MLP use `--opt_mlp` flag  
- To use the Bfloat16 MLP use `--use_bf16` flag
- To profile code use `--profile` flag



## Performance Numbers on Intel ICX-8380 CPU (40 core):

 The fp32 training time for optimized GAT on `OGBN-Papers100M` dataset is `avg. 450 sec/epoch` and `OGBN-Products` dataset it is `avg. 85 sec/epoch` 

 The command to run the `OGBN-Papers100M` is:

 ```
     ./run_gat_all.sh 2 "ogbn_papers"
 ```
  Other parameters used --dataset "ogbn-papers100M", --num-epochs 30, --batch-size 1024, --lr 0.006, --fan-out 15,10,5

The command to run the `OGBN-Products` is:

 ```
    ./run_gat_all.sh 2 "ogbn_products"
 ``` 
  Other parameters used --dataset "ogbn-products", --num-epochs 30, --lr 0.006 

## Recompile the package:

```
   ./reinstall_pcl_ext.sh
```

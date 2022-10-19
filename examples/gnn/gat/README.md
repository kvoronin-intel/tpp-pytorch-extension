
# Run optimized GAT

### Basic requirements to execute optimized GAT:

- DGL version 0.8+
- GCC version 10.X



## Command to run GAT

Follow this command to run with cpu affinity
 - To get the benefit of cpu affinitization in the code use `--cpu-worker-aff` flag in the command and assign dl > 0 otherwise use dl = 0 and `don't use the --cpu-worker-aff` flag.
 - To get the profiler of the GAT code use `--profile` flag

```
    dl=number_of_dataloader
    mdir_ss="\path\to\checkpoint\directory\"
    mkdir $mdir_ss
    ./run_gat_all.sh  $dl $mdir_ss
``` 

- To activate the optmized Float32 MLP use `--opt_mlp` flag  
- To use the Bfloat16 MLP use `--use_bf16` flag



## Performance Numbers on Intel ICX-8380 CPU (40 core):

 The float32 training time for optimized GAT on `OGBN-Papers100M` dataset is `~ 450 sec/epoch` and `OGBN-Products` dataset it is `~ 85 sec/epoch` 

 The command to run the `OGBN-Papers100M` is:

 ```
     numactl -C 38 -m 0  python  main.py  --num-epochs 30  --num-workers 2  --batch-size 1024 --dataset "ogbn-papers100M"  --lr 0.006 --fan-out 15,10,5 --checkpoint  --checkpoint-dir "ogbn_papers" --cpu-worker-aff  --opt_mlp
 ```

The command to run the `OGBN-Products` is:

 ```
     numactl -C 38 -m 0 python  main.py  --num-epochs 30  --num-workers 2  --dataset "ogbn-products" --cpu-worker-aff  --opt_mlp
 ``` 




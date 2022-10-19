
# Run optimized GAT

### Basic requirements to execute optimized GAT:

- DGL version 0.8+
- GCC version 10.X



## Command to run GAT

Follow this command to run with cpu affinity
 - To execute the code with cpu affnity use `--cpu-worker-aff` flag in the command.

```
    dl=number_of_dataloader
    mdir_ss="\path\to\checkpoint\directory\"
    mkdir $mdir_ss
    ./run_gat_all.sh  $dl $mdir_ss
``` 

- To activate the optmized Float32 MLP use `--opt_mlp` flag 
- To use the Bfloat16 MLP use `--use_bf16` flag in the command




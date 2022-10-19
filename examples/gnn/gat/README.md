
# Run optimized GAT

### Basic requirements to execute optimized GAT:

- DGL version 0.8+
- GCC version 10.X



## Command to run GAT

Follow this command to run 

```
   numactl -C $2-$up -m 0  python  main_gat.py  --num-epochs 30  --num-workers $2  --batch-size $4 --dataset "ogbn-papers100M"  --lr 0.006 --fan-out 15,10,5 --checkpoint  --checkpoint-dir $3 --cpu-worker-aff  --opt_mlp
``` 

- To activate the optmized Float32 MLP use `--opt_mlp` flag in the command.
- To use the Bfloat16 MLP use `--use_bf16` flag in the command




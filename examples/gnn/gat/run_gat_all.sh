
#!/bin/bash


export OMP_NUM_THREADS=$1

up=$(($1+$2-1))
echo $up


numactl -C $2-$up -m 0  python  main.py  --num-workers $2 --dataset "ogbn-papers100M"  --checkpoint  --checkpoint-dir $3 --cpu-worker-aff  --opt_mlp



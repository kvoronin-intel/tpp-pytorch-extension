
#!/bin/bash


export OMP_NUM_THREADS=$1

up=$(($1+$2-1))
echo $up

#numactl -C $2-$up -m 0  python main_gat.py --gpu -1 --num-epochs 2  --num-workers $2  --cpu-worker-aff --batch-size 1024 --dataset "ogbn-products" --opt_mlp
#numactl -C $2-$up -m 0  python main_gat.py --gpu -1 --num-epochs 2  --num-workers $2  --cpu-worker-aff --batch-size 1024 --dataset "ogbn-papers100M" --opt_mlp

numactl -C $2-$up -m 0  python  main_gat.py  --num-epochs 30  --num-workers $2  --batch-size $4 --dataset "ogbn-papers100M"  --lr 0.006 --fan-out 15,10,5 --checkpoint  --checkpoint-dir $3 --cpu-worker-aff  --opt_mlp  #--profile


#numactl -C $2-$up -m 0  python -u -W  main.py --gpu -1 --num-epochs 50  --num-workers $2  --batch-size 10240 --dataset "ogbn-papers100M"  --lr 0.006 --fan-out 15,10,5 --checkpoint  --checkpoint-dir $3 --cpu-worker-aff --opt_mlp  #--profile






















#numactl -C $2-$up  python main.py --gpu -1 --num-epochs $3  --num-workers $2  --cpu-worker-aff --use_pcl --batch-size 1024 --dataset "ogbn-papers100M" # --profile #--use_bf16 #--batch-size 1024

#numactl -C $2-$up  python main.py --gpu -1 --num-epochs $3  --num-workers $2  --cpu-worker-aff --batch-size 512 --dataset "ogbn-papers100M"

#numactl -C $2-$up  python main.py --gpu -1 --num-epochs $3  --num-workers $2  --cpu-worker-aff --use_pcl --batch-size 10000 --dataset "ogbn-papers100M"

#gdb --args numactl -C $2-$up  python main.py --gpu -1 --num-epochs $3  --num-workers $2  --use_pcl --cpu-worker-aff


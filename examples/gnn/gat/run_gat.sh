
#!/bin/bash


export OMP_NUM_THREADS=$1

up=$(($1+$2-1))
echo $up

numactl -C $2-$up  python main.py --gpu -1 --num-epochs $3  --num-workers $2  --use_pcl --cpu-worker-aff


#gdb --args numactl -C $2-$up  python main.py --gpu -1 --num-epochs $3  --num-workers $2  --use_pcl --cpu-worker-aff


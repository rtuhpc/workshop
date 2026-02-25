#!/bin/sh
#PBS -N cuda_job
#PBS -q batch
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -l feature=l40s
#PBS -l walltime=00:30:00

module load cuda/cuda-12.4

cd $PBS_O_WORKDIR

# compile 
nvcc hello-world.cu -o out
# execute
./out

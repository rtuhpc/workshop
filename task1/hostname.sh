#!/bin/bash
#PBS -N simple_test
#PBS -l nodes=1:ppn=1,mem=1g
#PBS -l walltime=00:30:00
#PBS -l feature=epyc
#PBS -q batch
##PBS -o test_$PBS_JOBID.out
#PBS -j oe

echo "Hello world from node $HOSTNAME"
echo "Sveiciens no nodes $HOSTNAME" 

#!/bin/bash
#PBS -N heat
#PBS -q rudens
#PBS -l walltime=00:01:00
#PBS -j oe

# For multi-node
# echo module load mpi/openmpi-4.1.1 >> ~/.bashrc

module load mpi/openmpi-4.1.1

echo "Datums:`date`"
echo "MPI procesu skaits:`cat $PBS_NODEFILE | wc -l`"

cd $PBS_O_WORKDIR


mpirun -N $PBS_NUM_PPN -hostfile $PBS_NODEFILE ./heat bottle.dat 10000
#mpirun -hostfile $PBS_NODEFILE ./heat 1152 1152 10000

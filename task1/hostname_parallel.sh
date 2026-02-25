#!/bin/bash
module load openmpi
mpirun -hostfile $PBS_NODEFILE /bin/hostname

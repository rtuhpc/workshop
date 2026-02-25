#!/bin/sh
#PBS -N blender
#PBS -q batch
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=2
#PBS -j oe
#PBS -t 1-100%50

#PARAMETRS=`cat param.txt | head -n $PBS_ARRAYID | tail -n 1`

module load blender/blender-2.70

cd $PBS_O_WORKDIR

blender -noaudio -b test.blend -o rend###.jpg --threads 2 -f $PBS_ARRAYID

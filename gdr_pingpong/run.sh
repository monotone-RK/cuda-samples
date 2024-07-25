#!/bin/bash
#------- qsub option -----------
#PBS -A CCUSC
#PBS -q gpu
#PBS -l elapstim_req=00:10:00
#PBS -T openmpi
#PBS -b 2
#PBS -v NQSV_MPI_VER=4.1.6/nvhpc23.1-cuda12.0

#------- Program execution ----------
cd $PBS_O_WORKDIR
module load openmpi/"$NQSV_MPI_VER"
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 2 -npernode 1 ./gdr_pingpong -max_elements 1048576 -num_iterations 100
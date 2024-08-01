#!/bin/bash
#------- qsub option -----------
#PBS -A CCUSC
#PBS -q gpu
#PBS -l elapstim_req=00:15:00
#PBS -T openmpi
#PBS -b 2
#PBS -v NQSV_MPI_VER=4.1.6/nvhpc23.1-cuda12.0

#------- Program execution ----------
cd $PBS_O_WORKDIR
module load openmpi/"$NQSV_MPI_VER"
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 2 -npernode 1 -x UCX_ZCOPY_THRESH=1 nsys profile -o 2_proc_gdr_allreduce_%h_%p --gpu-metrics-device=all --force-overwrite=true ./gdr_allreduce -max_elements 268435456 -num_iterations 1 -output_file 2_proc.csv

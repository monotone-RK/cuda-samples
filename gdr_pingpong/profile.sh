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
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 2 -npernode 1 nsys profile -o gdr_%h_%p --gpu-metrics-device=all --force-overwrite=true ./gdr_pingpong -max_elements 268435456 -num_iterations 1 -output_file gdr.csv
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 2 -npernode 1 -x UCX_TLS=sm,rc,ud,dc,ud_x,tcp,rdmacm,sockcm,self,cuda_ipc nsys profile -o no_gdr_1_%h_%p --gpu-metrics-device=all --force-overwrite=true ./gdr_pingpong -max_elements 268435456 -num_iterations 1 -output_file no_gdr_1.csv
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 2 -npernode 1 -x UCX_TLS=sm,rc,ud,dc,ud_x,tcp,rdmacm,sockcm,self,cuda_ipc,cuda_copy nsys profile -o no_gdr_2_%h_%p --gpu-metrics-device=all --force-overwrite=true ./gdr_pingpong -max_elements 268435456 -num_iterations 1 -output_file no_gdr_2.csv

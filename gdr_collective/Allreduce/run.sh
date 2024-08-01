#!/bin/bash
#------- qsub option -----------
#PBS -A CCUSC
#PBS -q gpu
#PBS -l elapstim_req=01:00:00
#PBS -T openmpi
#PBS -b 16
#PBS -v NQSV_MPI_VER=4.1.6/nvhpc23.1-cuda12.0

#------- Program execution ----------
cd $PBS_O_WORKDIR
module load openmpi/"$NQSV_MPI_VER"
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./gdr_allreduce -max_elements 268435456 -num_iterations 100 -output_file 1_proc.csv
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 2 -npernode 1 ./gdr_allreduce -max_elements 268435456 -num_iterations 100 -output_file 2_proc.csv
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 4 -npernode 1 ./gdr_allreduce -max_elements 268435456 -num_iterations 100 -output_file 4_proc.csv
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 8 -npernode 1 ./gdr_allreduce -max_elements 268435456 -num_iterations 100 -output_file 8_proc.csv
mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 16 -npernode 1 ./gdr_allreduce -max_elements 268435456 -num_iterations 100 -output_file 16_proc.csv

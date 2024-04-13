#!/bin/bash
#------- qsub option -----------
#PBS -A CCUSC
#PBS -q gpu
#PBS -l elapstim_req=01:00:00
#PBS -T openmpi
#PBS -b 1
#PBS -v NQSV_MPI_VER=4.1.5/gcc9.4.0-cuda11.8.0

#------- Program execution ----------
#--- move working dir
cd $PBS_O_WORKDIR
module load openmpi/"$NQSV_MPI_VER"
size=1024
while [ $size -le 32768 ]; do
  profile_name="naive_${size}"
  nsys profile -o ${profile_name} --gpu-metrics-device=all --force-overwrite=true mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./matmul ${size} naive
  ncu --target-processes all -o ${profile_name} --set full -f mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./matmul ${size} naive
  profile_name="shared_${size}"
  nsys profile -o ${profile_name} --gpu-metrics-device=all --force-overwrite=true mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./matmul ${size} shared
  ncu --target-processes all -o ${profile_name} --set full -f mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./matmul ${size} shared
  echo "Finished profiling for matrix size ${size}"
  size=$((size * 2))
done

echo "All profiling completed."

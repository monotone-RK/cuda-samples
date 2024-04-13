#!/bin/bash
#------- qsub option -----------
#PBS -A IDHUB
#PBS -q gen_S
#PBS -l elapstim_req=00:15:00
#PBS -T openmpi
#PBS -b 1
#PBS -v NQSV_MPI_VER=4.1.6/gcc11.4.0-cuda11.8.0

#------- Program execution ----------
#--- move working dir
cd $PBS_O_WORKDIR
module load openmpi/"$NQSV_MPI_VER"
size=65536
while [ $size -le 4294967296 ]; do
  profile_name="naive_${size}"
  nsys profile -o ${profile_name} --gpu-metrics-device=all --force-overwrite=true mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./daxpy ${size} 2
  # ncu --target-processes all -o ${profile_name} --set full -f mpirun ${NQSV_MPIOPTS} -report-bindings -bind-to none -np 1 -npernode 1 ./daxpy ${size} 2
  echo "Finished profiling for matrix size ${size}"
  size=$((size * 2))
done

echo "All profiling completed."

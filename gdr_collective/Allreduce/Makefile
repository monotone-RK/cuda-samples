FC=mpifort
FFLAGS=-O3 -cuda -gpu=cuda12.0,cc90 -tp=native
TARGET=gdr_allreduce

compile:
	$(FC) $(FFLAGS) nvtx_mod.cuf -c
	$(FC) $(FFLAGS) gdr_allreduce.cuf -c
	nvcc -c -arch sm_90 nvtx.cu
	$(FC) $(FFLAGS) -c++libs -cudalib=nvtx3 gdr_allreduce.o nvtx_mod.o nvtx.o -o $(TARGET)

run:
	qsub run.sh

profile:
	qsub profile.sh

result:
	python viz_gdr_effect.py --input *.csv --output result.svg

.PHONY: clean
clean:
	rm -rf $(TARGET) *.mod *.o *.sh.* *.svg *.csv *.nsys-rep

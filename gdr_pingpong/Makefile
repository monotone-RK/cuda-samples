FC=mpifort
FFLAGS=-O3 -cuda -gpu=cuda12.0,cc90 -tp=native
SRCS=gdr_pingpong.cuf
TARGET=gdr_pingpong

compile:
	$(FC) $(FFLAGS) $(SRCS) -o $(TARGET)

run:
	qsub run.sh

.PHONY: clean
clean:
	rm -rf $(TARGET)

FC=mpifort
FFLAGS=-O3 -cuda -gpu=cuda12.0,cc90 -tp=native
SRCS=gdr_pingpong.cuf
TARGET=gdr_pingpong

compile:
	$(FC) $(FFLAGS) $(SRCS) -o $(TARGET)

run:
	qsub run.sh

profile:
	qsub profile.sh

result:
	python viz_gdr_effect.py --input *.csv --output result.svg

.PHONY: clean
clean:
	rm -rf $(TARGET) run.sh.* *.svg

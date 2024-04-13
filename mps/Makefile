################################################################################
# Makefile for CUDA Sample Program                 Ryohei Kobayashi 2024.03.10 #
################################################################################

CC=nvcc
SRCS=main.cu
TARGET=matmul

compile:
	$(CC) -O3 -arch sm_90 $(SRCS) -o $(TARGET)

run:
	qsub eval_mps_effect.sh

check:
	qsub chk_mpsd_exist.sh

clean:
	rm -rf $(TARGET) *.sh.* *.nsys-rep *.csv .mps.*

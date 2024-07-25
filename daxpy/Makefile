################################################################################
# Makefile for CUDA Sample Program                 Ryohei Kobayashi 2024.03.10 #
################################################################################

CC=nvcc
SRCS=main.cu
TARGET=daxpy

compile:
	$(CC) -O3 -arch sm_90 $(SRCS) -o $(TARGET)

run:
	qsub chk_core_usage.sh

clean:
	rm -rf $(TARGET) *.sh.* *.nsys-rep *.csv .mps.*

################################################################################
# Makefile for CUDA Sample Program                      monotone-RK 2014.11.23 #
################################################################################

CC=nvcc
SRCS=main.cu
TARGET=matmul

compile:
	$(CC) -O3 $(SRCS) -o $(TARGET)

run:
	./$(TARGET) naive
	./$(TARGET) shared

clean:
	rm -f $(TARGET)

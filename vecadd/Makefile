################################################################################
# Makefile for CUDA Sample Program                      monotone-RK 2014.08.21 #
################################################################################

CC=nvcc
SRCS=main.cu
TARGET=vecadd

compile:
	$(CC) $(SRCS) -o $(TARGET)

run:
	./$(TARGET)

clean:
	rm -f $(TARGET)

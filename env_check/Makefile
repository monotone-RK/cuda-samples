################################################################################
# Makefile for CUDA Sample Program                      monotone-RK 2018.06.10 #
################################################################################

CC=nvcc
SRCS=main.cu
TARGET=env_check

compile:
	$(CC) -O3 $(SRCS) -o $(TARGET)

run:
	./$(TARGET) 

clean:
	rm -f $(TARGET)

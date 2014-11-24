/******************************************************************************/
/* CUDA Sample Program (Matrix Multiplication)         monotone-RK 2014.11.23 */
/******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define L_NAME "Matrix Multiplication with CUDA Ver 1.0"
#define MATRIX_SIZE 2048
#define BLOCK_SIZE  32
#define SHOW_SIZE   8

/** function to return the current time                                      **/
/******************************************************************************/
__host__
long long get_time() {
  struct timeval  tp;
  struct timezone tz;
  gettimeofday(&tp, &tz);
  return tp.tv_sec * 1000000ull + tp.tv_usec;
}

/******************************************************************************/
__global__
void matmul_naive(float *a, float *b, float *c, int matrix_size) {
  const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0;
  for (int i = 0; i<matrix_size; ++i) {
    sum += a[yidx*matrix_size+i] * b[i*matrix_size+xidx];
  }
  c[yidx*matrix_size+xidx] = sum;
}
  
/******************************************************************************/
__global__
void matmul_shared(float *a, float *b, float *c, int matrix_size) {
  const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;
  
  float sum = 0;
  for (int i = 0; i<matrix_size; i+=BLOCK_SIZE) {
    __shared__ float sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sub_b[BLOCK_SIZE][BLOCK_SIZE];

    sub_a[threadIdx.y][threadIdx.x] = a[yidx*matrix_size+(i+threadIdx.x)];
    sub_b[threadIdx.y][threadIdx.x] = b[(i+threadIdx.y)*matrix_size+xidx];
    __syncthreads();
    
    for (int j = 0; j<BLOCK_SIZE; ++j) {
      sum += sub_a[threadIdx.y][j] * sub_b[j][threadIdx.x];
    }
    __syncthreads();
    
  }
  c[yidx*matrix_size+xidx] = sum;
}
  
/******************************************************************************/
int main(int argc, char *argv[]) {

  if (argc == 1) {
    printf("%s\n", L_NAME);
    printf("usage: ./merge [execution type]\n");
    printf(" execution type : naive, shared\n");
    exit(1);
  }

  if (argc != 2) {
    printf("Error! The number of argument is wrong.\n");
    exit(1);
  }

  float *a;
  float *b;
  float *c;

  long long start;
  long long end;

  cudaMallocHost(&a, MATRIX_SIZE*MATRIX_SIZE*sizeof(float));
  cudaMallocHost(&b, MATRIX_SIZE*MATRIX_SIZE*sizeof(float));  
  cudaMallocHost(&c, MATRIX_SIZE*MATRIX_SIZE*sizeof(float));

  for (int y=0; y<MATRIX_SIZE; ++y) {
    for (int x=0; x<MATRIX_SIZE; ++x) {
      a[y*MATRIX_SIZE+x] = 1.0;
      b[y*MATRIX_SIZE+x] = 2.0;
      c[y*MATRIX_SIZE+x] = 0.0;
    }
  }

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);                         // determine the number of threads
  dim3 grid(MATRIX_SIZE/BLOCK_SIZE, MATRIX_SIZE/BLOCK_SIZE);  // determine the number of blocks
  
  if (!strcmp(argv[1], "naive")) {
    start = get_time();
    matmul_naive<<<grid, block>>>(a, b, c, MATRIX_SIZE);
  } else if (!strcmp(argv[1], "shared")) {
    start = get_time();
    matmul_shared<<<grid, block>>>(a, b, c, MATRIX_SIZE);
  } else {
    printf("execution type is wrong.\n");
    exit(1);
  }
  
  cudaThreadSynchronize();
  end = get_time();
  
  for (int y=0; y<SHOW_SIZE; ++y) {
    for (int x=0; x<SHOW_SIZE; ++x) {
      printf("%f ", c[y*MATRIX_SIZE+x]);
    }
    printf("\n");
  }
  printf("# elasped time:%9.3f sec\n", (end - start)/1000000.0);
  
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);

  return 0;
}

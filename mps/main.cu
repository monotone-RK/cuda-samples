#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK_SIZE  32
#define SHOW_SIZE   8

#define CUCHECK(call)                                                     \
  do {                                                                    \
    cudaError_t error = call;                                             \
    if (error != cudaSuccess) {                                           \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      return EXIT_FAILURE;                                                \
    }                                                                     \
  } while (0)

/** Stop watch                                                               **/
/******************************************************************************/
typedef struct timer {
  double seconds;
  double ref;
  void (*reset)(struct timer *this_timer);
  void (*start)(struct timer *this_timer);
  void (*stop)(struct timer *this_timer);
  void (*display)(struct timer *this_timer);
  double (*result)(struct timer *this_timer);
} Timer;

void timer_reset(Timer *this_timer) {
  this_timer->seconds = 0.0;
  this_timer->ref = 0.0;
  printf("Timer reset\n");
  struct timespec ts;
  clock_getres(CLOCK_MONOTONIC, &ts);
  printf("Time precision:\t%ld.%09ld sec\n", (long)ts.tv_sec, ts.tv_nsec);
}

void timer_start(Timer *this_timer) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  this_timer->ref = (double)(ts.tv_sec) + (double)ts.tv_nsec * 1e-9;
}

void timer_stop(Timer *this_timer) {
  this_timer->seconds -= this_timer->ref;
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  this_timer->ref = (double)(ts.tv_sec) + (double)ts.tv_nsec * 1e-9;
  this_timer->seconds += this_timer->ref;
}

void timer_display(Timer *this_timer) {
  printf("Elapsed time: \t%lf sec\n", this_timer->seconds);
}

double timer_result(Timer *this_timer) {
  return this_timer->seconds;
}

/******************************************************************************/
__global__
void matmul_naive(double *a, double *b, double *c, int matrix_size) {
  const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;

  double sum = 0;
  for (int i = 0; i<matrix_size; ++i) {
    sum += a[yidx*matrix_size+i] * b[i*matrix_size+xidx];
  }
  c[yidx*matrix_size+xidx] = sum;
}

/******************************************************************************/
__global__
void matmul_shared(double *a, double *b, double *c, int matrix_size) {
  const unsigned int xidx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int yidx = blockIdx.y * blockDim.y + threadIdx.y;

  double sum = 0;
  for (int i = 0; i<matrix_size; i+=BLOCK_SIZE) {
    __shared__ double sub_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double sub_b[BLOCK_SIZE][BLOCK_SIZE];

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
  // Show how to use this program
  if (argc != 3) {
    printf("Usage: ./matmul [matrix size] [execution type]\n");
    return EXIT_FAILURE;
  }

  // Declare stop watch
  Timer stop_watch = {
                      0.0,
                      0.0,
                      timer_reset,
                      timer_start,
                      timer_stop,
                      timer_display,
                      timer_result
                      };

  int matrix_size = atoi(argv[1]);
  double *a, *b, *c;
  double *d_a, *d_b, *d_c;

  // Allocate host memory
  CUCHECK(cudaMallocHost((void **)&a, matrix_size * matrix_size * sizeof(double)));
  CUCHECK(cudaMallocHost((void **)&b, matrix_size * matrix_size * sizeof(double)));
  CUCHECK(cudaMallocHost((void **)&c, matrix_size * matrix_size * sizeof(double)));

  // Allocate device memory
  CUCHECK(cudaMalloc((void **)&d_a, matrix_size * matrix_size * sizeof(double)));
  CUCHECK(cudaMalloc((void **)&d_b, matrix_size * matrix_size * sizeof(double)));
  CUCHECK(cudaMalloc((void **)&d_c, matrix_size * matrix_size * sizeof(double)));

  // Initialize matrices on the host
  for (int y = 0; y < matrix_size; ++y) {
    for (int x = 0; x < matrix_size; ++x) {
      a[y * matrix_size + x] = 1.0;
      b[y * matrix_size + x] = 2.0;
      c[y * matrix_size + x] = 0.0;
    }
  }

  // Copy matrices from the host to the device
  CUCHECK(cudaMemcpy(d_a, a, matrix_size * matrix_size * sizeof(double), cudaMemcpyHostToDevice));
  CUCHECK(cudaMemcpy(d_b, b, matrix_size * matrix_size * sizeof(double), cudaMemcpyHostToDevice));

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (matrix_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

  stop_watch.reset(&stop_watch);
  stop_watch.start(&stop_watch);
  if (!strcmp(argv[2], "naive")) {
    matmul_naive<<<grid, block>>>(d_a, d_b, d_c, matrix_size);
  } else if (!strcmp(argv[2], "shared")) {
    matmul_shared<<<grid, block>>>(d_a, d_b, d_c, matrix_size);
  } else {
    printf("Execution type is wrong.\n");
    return EXIT_FAILURE;
  }
  cudaDeviceSynchronize();
  stop_watch.stop(&stop_watch);

  // Copy result from device to host
  CUCHECK(cudaMemcpy(c, d_c, matrix_size * matrix_size * sizeof(double), cudaMemcpyDeviceToHost));

  // Display some of the result
  for (int y = 0; y < SHOW_SIZE && y < matrix_size; ++y) {
    for (int x = 0; x < SHOW_SIZE && x < matrix_size; ++x) {
      printf("%f ", c[y * matrix_size + x]);
    }
    printf("\n");
  }
  stop_watch.display(&stop_watch);

  // Free memory
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return EXIT_SUCCESS;
}

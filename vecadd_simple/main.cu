/******************************************************************************/
/* CUDA Sample Program (Vector add)                    monotone-RK 2014.08.21 */
/******************************************************************************/

#include <stdio.h>

void wait() {
  volatile int sum = 0;
  for (int i=0; i<10000; ++i) sum += i;
}

__global__
void vecadd(float *a, float *b, float *c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(int argc, char *argv[]) {
  const int num = 16;

  float *a;
  float *b;
  float *c;

  cudaMallocHost(&a, num*sizeof(float));
  cudaMallocHost(&b, num*sizeof(float));  
  cudaMallocHost(&c, num*sizeof(float));

  for (int i=0; i<num; i++) {
    a[i] = 1.0;
    b[i] = 2.0;
    c[i] = 0.0;
  }
    
  dim3 grid_size  = dim3(1, 1, 1);    // determine the number of blocks
  dim3 block_size = dim3(num, 1, 1);  // determine the number of threads
  
  vecadd<<<grid_size, block_size>>>(a, b, c);
  
  wait();
  
  for (int i=0; i<num; ++i) printf("c[%2d]: %f\n", i, c[i]);
  
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  
  return 0;
}
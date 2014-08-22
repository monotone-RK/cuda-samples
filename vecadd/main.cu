/******************************************************************************/
/* CUDA Sample Program (Vector add)                    monotone-RK 2014.08.21 */
/******************************************************************************/

#include <stdio.h>
#include <vector>

__global__
void vecadd(float *a, float *b, float *c) {
  c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main(int argc, char *argv[]) {
  const int num = 16;
  std::vector<float> a(num, 1.0);
  std::vector<float> b(num, 2.0);
  std::vector<float> c(num, 0.0);

  float *d_a;
  float *d_b;
  float *d_c;

  cudaMalloc(&d_a, num * sizeof(float));
  cudaMalloc(&d_b, num * sizeof(float));
  cudaMalloc(&d_c, num * sizeof(float));

  cudaMemcpy(d_a, &a[0], num*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &b[0], num*sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid_size  = dim3(1, 1, 1);    // determine the number of blocks
  dim3 block_size = dim3(num, 1, 1);  // determine the number of threads
  
  vecadd<<<grid_size, block_size>>>(d_a, d_b, d_c);
  
  cudaMemcpy(&c[0], d_c, num*sizeof(float), cudaMemcpyDeviceToHost);
  
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  for (int i=0; i<num; i++) printf("c[%2d]: %f\n", i, c[i]);
  
  return 0;
}
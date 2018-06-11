/******************************************************************************/
/* CUDA Sample Program (Vector add with stream)   Ryohei Kobayashi 2018.06.11 */
/******************************************************************************/
#include <iostream>
#include <cstdlib>
#include <vector>
#include <omp.h>

static const int numthread = 256;

__global__
void vecadd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char *argv[]) {
  // check command line arguments
  ///////////////////////////////////////////
  if (argc == 1) { std::cout << "usage: ./vecadd <numdata> <numstream> <numtry>"   << std::endl; exit(0); }
  if (argc != 4) { std::cerr << "Error! The number of argument is wrong." << std::endl; exit(1); }

  const int numdata   = std::stoull(std::string(argv[1]));
  const int numstream = std::stoull(std::string(argv[2]));
  const int numtry    = std::stoull(std::string(argv[3]));
  const int numbyte   = numdata * sizeof(float); // this sample uses "float"

  // host memory settings
  ///////////////////////////////////////////
  float *h_a, *h_b, *h_c;

  cudaMallocHost(&h_a, numbyte);
  cudaMallocHost(&h_b, numbyte);
  cudaMallocHost(&h_c, numbyte);
  
  #pragma omp parallel for
  for (int i = 0; i < numdata; ++i) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
    h_c[i] = 0.0f;
  }
  
  // device memory settings
  ///////////////////////////////////////////
  float *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, numbyte);
  cudaMalloc(&d_b, numbyte);
  cudaMalloc(&d_c, numbyte);

  // stream settings
  ///////////////////////////////////////////
  cudaStream_t stream[numstream];
  for (int stm = 0; stm < numstream; ++stm) {
    cudaStreamCreate(&stream[stm]);
  }

  // main routine
  ///////////////////////////////////////////
  const int numblock = (numdata % numthread) ? (numdata/numthread) + 1 : (numdata/numthread);
  
  double start = omp_get_wtime();
  for (int t = 0; t < numtry; ++t) {
    for (int stm = 0; stm < numstream; ++stm) {
      int idx = (numdata*stm)/numstream;
      
      cudaMemcpyAsync(&d_a[idx], &h_a[idx], numbyte/numstream, cudaMemcpyHostToDevice, stream[stm]);
      cudaMemcpyAsync(&d_b[idx], &h_b[idx], numbyte/numstream, cudaMemcpyHostToDevice, stream[stm]);
      
      vecadd<<<(numblock/numstream), numthread, 0, stream[stm]>>>(&d_a[idx], &d_b[idx], &d_c[idx], (numdata/numstream));
      
      cudaMemcpyAsync(&h_c[idx], &d_c[idx], numbyte/numstream, cudaMemcpyDeviceToHost, stream[stm]);
    }
    cudaDeviceSynchronize();
  }
  double end = omp_get_wtime();
  
  // verification
  ///////////////////////////////////////////
  bool error = false;
  std::cout << std::endl;
  
  #pragma omp parallel for
  for (int i = 0; i < numdata; ++i) {
    if (h_c[i] != (h_a[i]+h_b[i])) error = true;
  }
  
  if (!error) {
    std::cout << std::string(30, '-') << std::endl;
    std::cout << "Verification: PASS" << std::endl;
    std::cout << "elapsed time: " << std::fixed << (end - start)/static_cast<double>(numtry) << " sec" << std::endl;
  } else {
    std::cout << "Error! Verification failed..." << std::endl;
    for (int i = 0; i < numdata; ++i) {
      if (h_c[i] != (h_a[i]+h_b[i])) {
        std::cout << "h_c[" << i << "]: " << std::fixed << h_c[i] << ", expected: " << (h_a[i]+h_b[i]) << std::endl;
        break;
      }
    }
  }

  // cleanup
  ///////////////////////////////////////////
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  for (int stm = 0; stm < numstream; ++stm) {
    cudaStreamDestroy(stream[stm]);
  }
  
  return 0;
}

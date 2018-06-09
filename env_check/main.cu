#include <iostream>

int main(void) {
  int n;
  cudaGetDeviceCount(&n);
  std::cout << n << " CUDA devices found." << std::endl;
  return 0;
}

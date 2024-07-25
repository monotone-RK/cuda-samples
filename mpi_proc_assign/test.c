#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  int rank, size;
  int deviceId;
  struct cudaDeviceProp prop;
  char hostname[64];

  // MPIの初期化
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // ホスト名を取得
  gethostname(hostname, sizeof(hostname));
  // 現在のCUDAデバイスIDを取得
  cudaGetDevice(&deviceId);
  // デバイスプロパティの取得
  cudaGetDeviceProperties(&prop, deviceId);

  // 結果を出力
  printf("MPI Rank: %02d, Using %s (deviceId: %d, Bus-Id: %x) on %s\n", rank,
         prop.name, deviceId, prop.pciBusID, hostname);

  // MPIの終了
  MPI_Finalize();
  return 0;
}

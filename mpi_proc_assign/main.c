#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  int rank, size;
  int deviceCount, deviceId;
  struct cudaDeviceProp prop;
  char hostname[64];

  int procs_per_gpu = 4;

  // MPIの初期化
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // ホスト名を取得
  gethostname(hostname, sizeof(hostname));
  // CUDAデバイスの数を取得
  cudaGetDeviceCount(&deviceCount);
  // CUDAデバイスを設定
  //   cudaSetDevice(rank);  // deviceCountを超えるランクはデフォルトのデバイス（通常はデバイスID 0）を触るようになる
  //   cudaSetDevice(rank % deviceCount);  // GPU 0: Rank 0, GPU 1: 1,...
  cudaSetDevice((rank / procs_per_gpu) %
                deviceCount);  // GPU 0: Rank 0 ~ procs_per_gpu-1, GPU 1:
                               // procs_per_gpu ~...
  // 現在のCUDAデバイスIDを取得
  cudaGetDevice(&deviceId);
  // デバイスプロパティの取得
  cudaGetDeviceProperties(&prop, deviceId);

  // 結果を出力
  printf("MPI Rank: %02d, Using %s: %d - Bus-Id: %x - %d Devices on %s\n",
         rank, prop.name, deviceId, prop.pciBusID, deviceCount, hostname);

  // MPIの終了
  MPI_Finalize();
  return 0;
}

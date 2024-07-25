# NVIDIA GPUへのMPIプロセス割り当て方
## プログラム外で指定する方法
環境変数OMPI_COMM_WORLD_RANKの値を用いて計算した結果に応じて，CUDA_VISIBLE_DEVICESを設定するシェルスクリプトを用意する．
例えば，以下のシェルスクリプト (set_cuvisible_dev.sh) ではOMPI_COMM_WORLD_RANKが0から15の場合，プロセスは4プロセスごとに異なるGPUにアクセスする．
```bash
#!/bin/bash

# GPUあたりに割り当てるプロセス数とGPUの総数
procs_per_gpu=4
ngpus=4

# 環境変数OMPI_COMM_WORLD_RANKに基づいて、割り当てるGPUを計算
gpu_id=$(($OMPI_COMM_WORLD_RANK / $procs_per_gpu % $ngpus))

# 計算されたGPU IDに基づいてCUDA_VISIBLE_DEVICESを設定し、MPIプログラムを実行
exec env CUDA_VISIBLE_DEVICES=$gpu_id "$@"
```
set_cuvisible_dev.shに実行権限を付与する．
```
chmod +x set_cuvisible_dev.sh
```
以下のテストコード (test.c) で意図通りの挙動になるかを確認する．
```c
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
```
コンパイルし，実行してみる．
```bash
[rkobayashi@cygnus01] $ mpicc -I$CUDA_HOME/include test.c -L$CUDA_HOME/lib64 -lcudart
[rkobayashi@cygnus01] $ qlogin -A CCUSC -q debug -T openmpi -b 2 -v NQSV_MPI_VER=gdr/4.1.5/nvhpc22.11_gcc8.3.1-cuda11.8 -V
Request 390879.nqsv submitted to queue: debug.
Waiting for 390879.nqsv to start.
[rkobayashi@gnode47] $ cd $PBS_O_WORKDIR
[rkobayashi@gnode47] $ module load openmpi/gdr/4.1.5/nvhpc22.11_gcc8.3.1-cuda11.8
[rkobayashi@gnode47] $ mpirun ${NQSV_MPIOPTS} -np 16 -npernode 16 ./a.out
MPI Rank: 05, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 01, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 11, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 02, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 04, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 07, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 10, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 12, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 15, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 08, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 14, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 09, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 13, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 00, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 03, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 06, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
[rkobayashi@gnode47] $ mpirun ${NQSV_MPIOPTS} -np 16 -npernode 16 ./set_cuvisible_dev.sh ./a.out
MPI Rank: 15, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 40) on gnode47
MPI Rank: 08, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 3f) on gnode47
MPI Rank: 11, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 3f) on gnode47
MPI Rank: 09, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 3f) on gnode47
MPI Rank: 13, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 40) on gnode47
MPI Rank: 12, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 40) on gnode47
MPI Rank: 01, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 05, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1d) on gnode47
MPI Rank: 14, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 40) on gnode47
MPI Rank: 02, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 10, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 3f) on gnode47
MPI Rank: 04, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1d) on gnode47
MPI Rank: 07, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1d) on gnode47
MPI Rank: 00, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
MPI Rank: 06, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1d) on gnode47
MPI Rank: 03, Using Tesla V100-PCIE-32GB (deviceId: 0, Bus-Id: 1c) on gnode47
[rkobayashi@gnode47] $ nvidia-smi
Thu Mar  7 00:21:43 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.06              Driver Version: 545.23.06    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla V100-PCIE-32GB           On  | 00000000:1C:00.0 Off |                    0 |
| N/A   36C    P0              26W / 250W |      0MiB / 32768MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla V100-PCIE-32GB           On  | 00000000:1D:00.0 Off |                    0 |
| N/A   39C    P0              28W / 250W |      0MiB / 32768MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  Tesla V100-PCIE-32GB           On  | 00000000:3F:00.0 Off |                    0 |
| N/A   36C    P0              28W / 250W |      0MiB / 32768MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  Tesla V100-PCIE-32GB           On  | 00000000:40:00.0 Off |                    0 |
| N/A   35C    P0              27W / 250W |      0MiB / 32768MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```
set_cuvisible_dev.shを付ければ4プロセスごとに異なるGPUにアクセスするようになるが，付けないと全てのプロセスはBus-Id: 1cのGPUにアクセスする．
各ノードに搭載されているGPUのBus-Idは```nvidia-smi```で確認出来る．

## プログラム外で指定する方法
cudaSetDevice()関数を使ってGPUに割り当てるプロセスを指定する．
### main.c
```c
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
```
#### コンパイル
```bash
mpicc -I$CUDA_HOME/include main.c -L$CUDA_HOME/lib64 -lcudart
```
#### 実行 (Cygnus上)
debugノード x 2 に搭載されている各 GPU に MPI プロセス を 4 つ割り当ててみる．
```bash
[rkobayashi@cygnus01] $ qlogin -A CCUSC -q debug -T openmpi -b 2 -v NQSV_MPI_VER=gdr/4.1.5/nvhpc22.11_gcc8.3.1-cuda11.8 -V
Request 389830.nqsv submitted to queue: debug.
Waiting for 389830.nqsv to start.
[rkobayashi@gnode47] $ cd $PBS_O_WORKDIR
[rkobayashi@gnode47] $ module load openmpi/gdr/4.1.5/nvhpc22.11_gcc8.3.1-cuda11.8
[rkobayashi@gnode47] $ mpirun ${NQSV_MPIOPTS} -np 32 -npernode 16 ./a.out
MPI Rank: 30, Using Tesla V100-PCIE-32GB: 3 - Bus-Id: 40 - 4 Devices on gnode48
MPI Rank: 07, Using Tesla V100-PCIE-32GB: 1 - Bus-Id: 1d - 4 Devices on gnode47
MPI Rank: 16, Using Tesla V100-PCIE-32GB: 0 - Bus-Id: 1c - 4 Devices on gnode48
MPI Rank: 31, Using Tesla V100-PCIE-32GB: 3 - Bus-Id: 40 - 4 Devices on gnode48
MPI Rank: 11, Using Tesla V100-PCIE-32GB: 2 - Bus-Id: 3f - 4 Devices on gnode47
MPI Rank: 09, Using Tesla V100-PCIE-32GB: 2 - Bus-Id: 3f - 4 Devices on gnode47
MPI Rank: 18, Using Tesla V100-PCIE-32GB: 0 - Bus-Id: 1c - 4 Devices on gnode48
MPI Rank: 14, Using Tesla V100-PCIE-32GB: 3 - Bus-Id: 40 - 4 Devices on gnode47
MPI Rank: 19, Using Tesla V100-PCIE-32GB: 0 - Bus-Id: 1c - 4 Devices on gnode48
MPI Rank: 10, Using Tesla V100-PCIE-32GB: 2 - Bus-Id: 3f - 4 Devices on gnode47
MPI Rank: 21, Using Tesla V100-PCIE-32GB: 1 - Bus-Id: 1d - 4 Devices on gnode48
MPI Rank: 05, Using Tesla V100-PCIE-32GB: 1 - Bus-Id: 1d - 4 Devices on gnode47
MPI Rank: 08, Using Tesla V100-PCIE-32GB: 2 - Bus-Id: 3f - 4 Devices on gnode47
MPI Rank: 26, Using Tesla V100-PCIE-32GB: 2 - Bus-Id: 3f - 4 Devices on gnode48
MPI Rank: 04, Using Tesla V100-PCIE-32GB: 1 - Bus-Id: 1d - 4 Devices on gnode47
MPI Rank: 28, Using Tesla V100-PCIE-32GB: 3 - Bus-Id: 40 - 4 Devices on gnode48
MPI Rank: 13, Using Tesla V100-PCIE-32GB: 3 - Bus-Id: 40 - 4 Devices on gnode47
MPI Rank: 22, Using Tesla V100-PCIE-32GB: 1 - Bus-Id: 1d - 4 Devices on gnode48
MPI Rank: 02, Using Tesla V100-PCIE-32GB: 0 - Bus-Id: 1c - 4 Devices on gnode47
MPI Rank: 12, Using Tesla V100-PCIE-32GB: 3 - Bus-Id: 40 - 4 Devices on gnode47
MPI Rank: 27, Using Tesla V100-PCIE-32GB: 2 - Bus-Id: 3f - 4 Devices on gnode48
MPI Rank: 20, Using Tesla V100-PCIE-32GB: 1 - Bus-Id: 1d - 4 Devices on gnode48
MPI Rank: 03, Using Tesla V100-PCIE-32GB: 0 - Bus-Id: 1c - 4 Devices on gnode47
MPI Rank: 06, Using Tesla V100-PCIE-32GB: 1 - Bus-Id: 1d - 4 Devices on gnode47
MPI Rank: 23, Using Tesla V100-PCIE-32GB: 1 - Bus-Id: 1d - 4 Devices on gnode48
MPI Rank: 01, Using Tesla V100-PCIE-32GB: 0 - Bus-Id: 1c - 4 Devices on gnode47
MPI Rank: 15, Using Tesla V100-PCIE-32GB: 3 - Bus-Id: 40 - 4 Devices on gnode47
MPI Rank: 00, Using Tesla V100-PCIE-32GB: 0 - Bus-Id: 1c - 4 Devices on gnode47
MPI Rank: 24, Using Tesla V100-PCIE-32GB: 2 - Bus-Id: 3f - 4 Devices on gnode48
MPI Rank: 25, Using Tesla V100-PCIE-32GB: 2 - Bus-Id: 3f - 4 Devices on gnode48
MPI Rank: 17, Using Tesla V100-PCIE-32GB: 0 - Bus-Id: 1c - 4 Devices on gnode48
MPI Rank: 29, Using Tesla V100-PCIE-32GB: 3 - Bus-Id: 40 - 4 Devices on gnode48
```
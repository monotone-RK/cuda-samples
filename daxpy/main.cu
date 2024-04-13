#include <cuda_runtime.h>
#include <stdint.h>  // uint64_t を使用するために追加
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BLOCK_SIZE 256  // スレッドブロックサイズを適切に設定
#define CUCHECK(call)                                                     \
  do {                                                                    \
    cudaError_t error = call;                                             \
    if (error != cudaSuccess) {                                           \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE);                                                 \
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

double timer_result(Timer *this_timer) { return this_timer->seconds; }

// DAXPYカーネル関数の定義
__global__ void daxpy(size_t n, double a, double *x, double *y) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = a * x[i] + y[i];
  }
}

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <vector size> <scalar a>\n", argv[0]);
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

  uint64_t n = strtoull(argv[1], NULL, 10);  // ベクトルのサイズ
  double a = atof(argv[2]);                  // スカラーa
  double *x, *y, *d_x, *d_y;

  // ホストメモリの割り当てと初期化
  x = (double *)malloc(n * sizeof(double));
  if (x == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for x\n");
    exit(EXIT_FAILURE);
  }
  y = (double *)malloc(n * sizeof(double));
  if (y == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for y\n");
    free(x);  // xはすでに割り当てられているので、解放する
    exit(EXIT_FAILURE);
  }

  // ベクトルの初期化に使用するループカウンタも uint64_t に修正
  for (uint64_t i = 0; i < n; i++) {
    x[i] = 1.0;  // ここでは例として1.0で初期化
    y[i] = 2.0;  // ここでは例として2.0で初期化
  }

  // デバイスメモリの割り当て
  CUCHECK(cudaMalloc(&d_x, n * sizeof(double)));
  CUCHECK(cudaMalloc(&d_y, n * sizeof(double)));

  // ホストからデバイスへのデータ転送
  CUCHECK(cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice));
  CUCHECK(cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

  // カーネルの実行
  int blocksPerGrid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  stop_watch.start(&stop_watch);
  daxpy<<<blocksPerGrid, BLOCK_SIZE>>>(n, a, d_x, d_y);

  // カーネルの実行が完了するのを待つ
  CUCHECK(cudaDeviceSynchronize());
  stop_watch.stop(&stop_watch);

  // デバイスからホストへのデータ転送
  CUCHECK(cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost));

  // 結果の表示（オプションで結果を表示したい場合はここで行う）
  for (int i = 0; i < 4; i++) {
    printf("%f\n", y[i]);
  }
  stop_watch.display(&stop_watch);
  printf("%f TFLOPS\n", ((2.0f * n) / (stop_watch.result(&stop_watch)) / 1e12));
  // メモリの解放
  free(x);
  free(y);
  CUCHECK(cudaFree(d_x));
  CUCHECK(cudaFree(d_y));

  return EXIT_SUCCESS;
}

#!/bin/bash
# 実行するmatmulプロセスの数を設定
NUM_PROCESSES=4

# matmulの実行と時間計測
run_matmul() {
    echo "Running $1..."
    local start=$(date +%s.%N)
    local pids=() # プロセスIDを格納する配列

    # 指定された数のmatmulプロセスを起動し、プロセスIDを配列に保存
    for ((i=0; i<$NUM_PROCESSES; i++)); do
        ./matmul $1 &
        pids+=($!) # 最後にバックグラウンドで実行されたプロセスのIDを配列に追加
    done
    # sleep 2
    # nvidia-smi

    # 各matmulプロセスの終了を個別に待機
    for pid in "${pids[@]}"; do
        wait $pid
    done

    local end=$(date +%s.%N)
    local elapsed=$(echo "$end - $start" | bc)
    echo "Total time elapsed: $elapsed seconds"
}

run_matmul naive

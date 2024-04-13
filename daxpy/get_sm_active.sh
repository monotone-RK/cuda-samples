#!/bin/bash

# 現在のディレクトリの .sqlite ファイルのリストを取得
for sqlite_file in *.sqlite; do
  # SQLite データベースから GPU メトリクスをエクスポート
  sqlite3 "$sqlite_file" < export_gpu_metrics.sql

  # CSV ファイルから GPU のコアの稼働率を計算し、結果を表示
  avg=$(awk -F, '$2 != 0 {sum+=$2; count++} END {if (count > 0) print sum/count; else print "No non-zero entries found"}' SM-Active.csv)

  # 結果の表示
  echo "$sqlite_file: $avg"

  # 生成された CSV ファイルを削除（必要に応じて）
  # rm "SM-Active.csv"
done

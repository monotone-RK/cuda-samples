import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def format_data_size(size):
    if size >= 1073741824:
        return f'{size // 1073741824}G'
    elif size >= 1048576:
        return f'{size // 1048576}M'
    elif size >= 1024:
        return f'{size // 1024}K'
    else:
        return str(size)

def plot_data(csv_files, output_file):
    plt.figure(figsize=(18, 10))

    max_bandwidth = 0

    for csv_file in csv_files:
        # CSVファイルを読み込む
        df = pd.read_csv(csv_file)

        # 横軸のラベルをフォーマット
        formatted_data_size = [format_data_size(size) for size in df['Data size']]

        # CSVファイル名を拡張子を除いて取得
        legend_name = os.path.splitext(os.path.basename(csv_file))[0]

        # データをプロット
        plt.plot(formatted_data_size, df['Bandwidth'], marker='o', linestyle='-', label=legend_name)

        # 最大帯域幅を更新
        max_bandwidth = max(max_bandwidth, max(df['Bandwidth']))

    # フォントサイズを設定
    plt.xlabel('Data size [Bytes]', fontsize=20)
    plt.ylabel('Bandwidth [GB/s]', fontsize=20)
    plt.xticks(ticks=range(len(formatted_data_size)), labels=formatted_data_size, rotation='vertical', fontsize=20, ha='center')
    plt.yticks(fontsize=20)

    # 凡例を表示（グラフのボックスの外、中央の上側、横並び）
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=20, ncol=len(csv_files))

    # 縦軸を0から最大帯域幅の1.1倍までに設定
    plt.ylim(0, max_bandwidth * 1.1)

    # グリッド線を消す
    plt.grid(False)

    # レイアウトを調整
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # SVGファイルとして保存
    plt.savefig(output_file, format='svg', bbox_inches='tight')

    # グラフを表示
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from CSV files.')
    parser.add_argument('-i', '--input', nargs='+', required=True, help='Input CSV file(s)')
    parser.add_argument('-o', '--output', required=True, help='Output SVG file')
    args = parser.parse_args()

    plot_data(args.input, args.output)

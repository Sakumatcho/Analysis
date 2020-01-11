"""
改良予定
現状は各変数の分布のみだが、相関分析できるようにする
"""

from configparser import ConfigParser
from pathlib import Path 
import sys 
from typing import List 
from math import log2 

import pandas as pd 
from matplotlib import pyplot as plt

# グローバル
args = list()
path_input_dir = Path()
path_output_dir_root = Path()
input_charset = ''
threshould_hist = 0

def get_args():
    """
    コマンドライン引数取得
    """
    global args 
    args = sys.argv
    if len(args) >= 3:
        print('引数が多すぎます。')
        sys.exit()


def get_conf():
    """
    設定ファイル読込
    読み込んだ値を全てグローバル変数に代入
    """
    conf = ConfigParser()
    conf.read('02_conf_analysis_correlation.ini')
    global path_input_dir 
    global path_output_dir_root 
    global input_charset
    global threshould_hist
    if len(args) == 2:
        path_input_dir = Path(args[1])

    else: 
        # get_args()で引数チェックしているため、引数なしの場合のみ
        path_input_dir = Path(conf['file']['input_dir'])

    path_output_dir_root = Path(conf['file']['output_dir'])
    input_charset = conf['file']['input_charset']
    threshould_hist = int(conf['graph']['threshould_hist'])

    print(f'入力フォルダ: {path_input_dir}')
    print(f'出力フォルダ: {path_output_dir_root}')
    print(f'入力ファイルの文字コード: {input_charset}')


def get_inputfilepaths():
    """
    入力ファイルリスト取得
    
    Returns:
        List[pathlib.Path]: 入力ファイルパスのリスト
    """
    list_path_inputfiles = list(path_input_dir.glob('*.csv'))

    return list_path_inputfiles


def make_output_dir(list_path_inputfiles):
    """
    出力先フォルダ作成
    入力CSVファイルと同名のフォルダを作成

    Args:
        list_path_inputfiles (List[pathlib.Path]): 入力ファイルパスのリスト
    
    Returns:
        List[pathlib.Path]: 出力先フォルダパスのリスト
    """
    list_path_outputdirs = list()
    for inputfilename in list_path_inputfiles:
        path_output_dir = path_output_dir_root/inputfilename.name.replace('.csv', '')
        path_output_dir.mkdir(parents=True, exist_ok=True)
        list_path_outputdirs.append(path_output_dir)

    return list_path_outputdirs


def calc_starges(num_samples):
    """
    スタージェスの公式により、サンプル数からヒストグラムのビンの数を算出
    
    Args:
        num_samples (int): データのサンプル数
    
    Returns:
        int: ヒストグラムのビンの数
    """
    return int(round(log2(num_samples) + 1, 0))


def make_hist(df_data, path_outputdir, num_bin, num_sample):
    """
    ヒストグラム作成
    全レコード欠損の場合はヒストグラム作成せず
    ヒストグラムのビンの数について：
        数値データはスタージェスの公式数
        カテゴリデータはカテゴリ数
    Args:
        df_data (pd.DataFrame): 入力データCSV
        path_outputdir (pathlib.Path): 出力先フォルダパス
        num_bin (int): 数値データヒストグラムビン数
        num_sample (int): DataFrameレコード数
    """
    num_cols = len(df_data.columns)
    num_fig_rows = round(num_cols / 4, 0)
    fig = plt.figure(figsize=(40,7 * num_fig_rows))
    for idx, col in enumerate(df_data.columns):
        if df_data[col].isna().sum() == num_sample: 
            print(f'{col}: ヒストグラム化不可')
            continue 

        print(f'{col}: ヒストグラム化可')
        ax = fig.add_subplot(num_fig_rows,4,idx+1)
        if (df_data[col].dtype == 'object') or (len(df_data[col].unique()) <= threshould_hist): 
            sr_counts = df_data[col].value_counts().sort_index()
            ax.bar(sr_counts.index.astype(str).fillna('欠損'), sr_counts)

        else: 
            ax.hist(df_data[col].dropna(), bins=num_bin)

        ax.set_title(col, fontsize=19)
        ax.set_xlabel('value', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        plt.yscale('log')

    plt.tight_layout()
    outputfilepath = path_outputdir/(path_outputdir.name + '_histgram.png')
    plt.savefig(outputfilepath, dpi=100)
    plt.close()

    print(f'ヒストグラムファイルパス: {outputfilepath}')


def analysis_distribution(df_data, path_outputdir):

        num_sample = len(df_data)
        num_bin = calc_starges(num_sample)
        print(f'ビンの数： {num_bin}')
        # ヒストグラム生成
        make_hist(df_data, path_outputdir, num_bin, num_sample)
        

def exec_analysis(list_path_inputfiles, list_path_outputdirs):

    for inputfilepath, outputdirpath in zip(list_path_inputfiles, list_path_outputdirs):
        print(f'入力ファイル名: {inputfilepath}')
        df_data = pd.read_csv(inputfilepath, encoding=input_charset, engine='python')
        print(f'カラム名: {df_data.columns}')
        print(f'レコード数: {len(df_data)}')
        print('データ例: ')
        print(df_data.head(5))
        analysis_distribution(df_data, outputdirpath)
        # analysis_correlation(list_path_inputfiles, list_path_outputdirs)

if __name__ =='__main__':
    get_args()

    get_conf()
    if not(path_input_dir.exists()):
        print(f'次のフォルダは存在しません。終了します。: {path_input_dir.name}')
        sys.exit()

    list_path_inputfiles = get_inputfilepaths()
    if len(list_path_inputfiles) == 0:
        print('入力フォルダにCSVファイルが存在しません。終了します。')
        sys.exit()
    
    list_path_outputdirs = make_output_dir(list_path_inputfiles)
    print(f'出力先フォルダパス: {list_path_outputdirs}')

    exec_analysis(list_path_inputfiles, list_path_outputdirs)

    # 終了
    print('全ての処理が終了しました。')
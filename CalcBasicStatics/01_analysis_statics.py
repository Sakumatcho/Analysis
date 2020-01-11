"""
改良予定：
    出力先をexcel
    フィルター
    条件付き書式で欠損率50%以上のカラムをグレーアウトとか
"""

from configparser import ConfigParser
from pathlib import Path 
import sys 
from typing import List 

import pandas as pd 
import pandas_profiling as pdp  

# グローバル
args = list()
path_input_dir = Path()
path_output_dir_root = Path()
input_charset = ''

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
    conf.read('01_conf_analysis_statics.ini')
    global path_input_dir 
    global path_output_dir_root 
    global input_charset
    if len(args) == 2:
        path_input_dir = Path(args[1])

    else: 
        # get_args()で引数チェックしているため、引数なしの場合のみ
        path_input_dir = Path(conf['file']['input_dir'])

    path_output_dir_root = Path(conf['file']['output_dir'])
    input_charset = conf['file']['input_charset']

    print(f'入力フォルダ: {path_input_dir}')
    print(f'出力フォルダ: {path_output_dir_root}')


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


def aggrigate_statics(df_data, path_outputdir):
    """
    基本統計量算出
    
    Args:
        df_data (pd.DataFrame): 算出対象CSVデータ
        path_outputdir (pathlib.Path): 出力フォルダパス
    """
    list_cols = ['全レコード数','count','欠損数','欠損率','unique','top','freq',
                'mean','std','min','25%','50%','75%','max','75% - 25%']
    num_record = len(df_data)
    df_statics = df_data.describe(include='all').T
    df_statics['全レコード数'] = num_record
    df_statics['欠損数'] = num_record - df_statics['count']
    df_statics['欠損率'] = df_statics['欠損数'] / num_record
    df_statics['75% - 25%'] = df_statics['75%'] - df_statics['25%']
    df_statics = df_statics.loc[:,list_cols].copy()
    df_statics = df_statics.rename(columns={'count': '有効レコード数','unique': '値種類数','top': '最頻値',
                                            'freq': '頻度','mean': '平均値','std': '標準偏差',
                                            'min': '最小値','25%': '25%-tile値','50%': '中央値',
                                            '75%': '75%-tile値','max': '最大値', '75% - 25%': '四分位範囲'})

    print(df_statics)
    df_statics.to_csv(path_outputdir/(path_outputdir.name + '_statics.csv'), header=True, index=True)


def aggrigate_uniques(df_data, path_outputdir):
    """
    値種類カウント
    
    Args:
        df_data (pd.DataFrame): 値種類カウント対象CSVデータ
        path_outputdir (pathlib.Path): 出力フォルダパス
    """
    list_cols = df_data.columns.tolist()
    list_df_unique_count = list()
    for col in list_cols:
        sr_unique_counts = df_data[col].value_counts()
        df_unique_counts = pd.DataFrame()
        df_unique_counts[col] = sr_unique_counts.index
        df_unique_counts['レコード数'] = sr_unique_counts.values
        list_df_unique_count.append(df_unique_counts)

    df_unique_counts = pd.concat(list_df_unique_count, axis=1)
    df_unique_counts.to_csv(path_outputdir/(path_outputdir.name + '_unique_counts.csv'), header=True, index=False)


def report_profile(df_data, path_outputdir):
    """
    pandas-profileレポート出力
    
    Args:
        df_data (pd.DataFrame): レポート対象CSVデータ
        path_outputdir (pathlib.Path): 出力フォルダパス
    """
    profile = pdp.ProfileReport(df_data)
    profile.to_file(path_outputdir/(path_outputdir.name + '_profile.html'))


def exec_report(list_path_inputfiles, list_path_outputdirs):
    """
    集計
    
    Args:
        list_path_inputfiles (List[pathlib.Path]): 入力ファイルパスリスト
        list_path_outputdirs (List[pathlib.Path]): 出力フォルダパスリスト
    """
    for idx, inputfilepath in enumerate(list_path_inputfiles):
        print(f'入力ファイル[{idx + 1}]: {inputfilepath.name}')
        df_data = pd.read_csv(inputfilepath, encoding=input_charset, engine='python')
        print(f'行数: {len(df_data)}')
        print(f'列数: {len(df_data.columns)}')
        print(f'列名: {df_data.columns}')
        print('データ概要')
        print(df_data.head(5))

        path_outputdir = list_path_outputdirs[idx]
        aggrigate_statics(df_data, path_outputdir)
        aggrigate_uniques(df_data, path_outputdir)
        report_profile(df_data, path_outputdir)


if __name__ == '__main__':
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

    exec_report(list_path_inputfiles, list_path_outputdirs)

    # 終了
    print('全ての処理が終了しました。')
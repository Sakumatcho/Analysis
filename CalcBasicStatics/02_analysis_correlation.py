"""
改良予定
現状は各変数の分布のみだが、相関分析できるようにする
"""

from configparser import ConfigParser
from pathlib import Path 
import sys 
from typing import List 
from math import log2, ceil
import itertools 

import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np 

# グローバル
args = list()
path_input_dir = Path()
path_output_dir_root = Path()
input_charset = ''
threshould_hist = 0
corr_mode = 'all'

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
    corr_mode = conf['graph']['corr_mode']

    print(f'入力フォルダ: {path_input_dir}')
    print(f'出力フォルダ: {path_output_dir_root}')
    print(f'入力ファイルの文字コード: {input_charset}')
    print(f'相関係数算出モード: {corr_mode}')


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


def set_log_scale(ax, sr_counts):
    if sr_counts.max() // sr_counts.min() >= 1000: 
        ax.set_yscale('log')
        ax.set_ylim([sr_counts.min() - (sr_counts.min() / 2), sr_counts.max() + (sr_counts.max() / 2)])


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
    num_fig_rows = ceil(num_cols / 4)
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
            set_log_scale(ax, sr_counts)

        else: 
            sr_feature = df_data[col].dropna()
            ax.hist(sr_feature, bins=num_bin)

            sr_feature_bins = sr_feature // ((sr_feature.max() - sr_feature.min()) / num_bin)
            sr_feature_counts = sr_feature_bins.value_counts()
            set_log_scale(ax, sr_feature_counts)

        ax.set_title(col, fontsize=19)
        ax.set_xlabel('value', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    plt.tight_layout()
    outputfilepath = path_outputdir/(path_outputdir.name + '_histgram.png')
    plt.savefig(outputfilepath, dpi=100)
    plt.close()

    print(f'ヒストグラムファイルパス: {outputfilepath}')


def analysis_distribution(df_data, path_outputdir):
    """
    ヒストグラムによるデータ分布作成
    
    Args:
        df_data (pd.DataFrame): 入力データCSV
        path_outputdir (pathlib.Path): 出力先フォルダパス
    """
    num_sample = len(df_data)
    num_bin = calc_starges(num_sample)
    print(f'ビンの数： {num_bin}')
    # ヒストグラム生成
    make_hist(df_data, path_outputdir, num_bin, num_sample)
        
        
def calc_pearson_corr(sr_subject, df_object_num): 
    """
    Pearsonの相関係数算出
    数値型×数値型の相関係数をPearsonの相関係数で算出
    
    Args:
        sr_subject (pd.Series): 主目的特徴量
        df_object_num (pd.DataFrame): 相手となるその他の数値型の特徴量
    
    Returns:
        pd.DataFrame: Pearsonの相関係数および相関係数タイプ(相関係数)
    """
    df_pearson_corr = pd.DataFrame(index=df_object_num.columns, columns=['相関係数', '相関係数タイプ'])
    for object_feature in df_object_num.columns: 
        sr_object_feature = df_object_num[object_feature].dropna()
        list_match_index = list(set(sr_subject.index) & set(sr_object_feature.index))

        df_pearson_corr.loc[object_feature, '相関係数'] = sr_subject[list_match_index].corr(sr_object_feature[list_match_index])

    df_pearson_corr['相関係数タイプ'] = '相関係数'
    return df_pearson_corr 


def rate_corr(sr_numeric_feature, sr_categorical_feature): 
    """
    相関比算出
    
    Args:
        sr_numeric_feature (pd.Series): 数値型変数配列
        sr_categorical_feature (pd.Series): カテゴリ型変数配列
    
    Returns:
        float: 相関比
    """
    numeric = sr_numeric_feature.name 
    category = sr_categorical_feature.name
    df_calc = pd.concat([sr_numeric_feature, sr_categorical_feature], axis=1)

    # クラス毎の平均値
    groupby_calc = df_calc.groupby(category)
    sr_group_ave = groupby_calc.mean()[numeric]

    # 偏差平方和
    list_dev = list()
    for label in df_calc[category].unique().tolist(): 
        query = df_calc[category] == label 
        sr_numeric_by_label = df_calc[query][numeric]
        list_dev.append(pow((sr_numeric_by_label - sr_group_ave[label]), 2).sum())

    # 級内変動
    dev_inner_class = sum(list_dev) 

    # 級間変動
    ave_all = sr_numeric_feature.mean()
    sr_deviation_inter_class = pow((sr_group_ave - ave_all), 2)
    sr_group_count = groupby_calc.count()
    dev_inter_class = np.dot(sr_group_count.values.reshape(-1), sr_deviation_inter_class.values.reshape(-1))

    return dev_inter_class / (dev_inner_class + dev_inter_class)


def calc_rate_corr(sr_subject_feature, df_object, category_col=None): 
    """
    相関比の算出
    SeriesとDataFrameとどちらがカテゴリ型変数化に応じて適切な引数設定をし、相関比算出処理を実行
    Args:
        sr_subject_feature (pd.Series): 主目的特徴量
        df_object (pd.DataFrame): 相手となる特徴量全て
        category_col (str, optional): カテゴリ型変数がsr('left')かdf('right)のどちらか. Defaults to None.
    
    Raises:
        ValueError: カテゴリ変数の指定方法誤り
    
    Returns:
        pd.DataFrame: 相関比リスト
    """
    df_rate_corr = pd.DataFrame(index=df_object.columns, columns=['相関係数', '相関係数タイプ'])
    if category_col == 'left': 
        for object_feature in df_object.columns: 
            sr_object_feature = df_object[object_feature].dropna()
            list_match_index = list(set(sr_subject_feature.index) & set(sr_object_feature.index))
            df_rate_corr.loc[object_feature, '相関係数'] = rate_corr(sr_object_feature[list_match_index], sr_subject_feature[list_match_index])

    elif category_col == 'right':
        for object_feature in df_object.columns: 
            sr_object_feature = df_object[object_feature].dropna()
            list_match_index = list(set(sr_subject_feature.index) & set(sr_object_feature.index))
            df_rate_corr.loc[object_feature, '相関係数'] = rate_corr(sr_subject_feature[list_match_index], sr_object_feature[list_match_index])

    else: 
        raise ValueError(f'カテゴリ変数の指定が誤りがあります: category_col={category_col}')

    df_rate_corr['相関係数タイプ'] = '相関比' 
    return df_rate_corr


def cramersV(x, y):
    """
    Cramerの連関係数算出
    
    Args:
        x (np.ndarray, pd.Series): カテゴリ型の特徴量配列1
        y (np.ndarray, pd.Series): カテゴリ型の特徴量配列2
    
    Returns:
        float: Cramerの連関係数
    """
    table = np.array(pd.crosstab(x, y)).astype(np.float32)
    n = table.sum()
    colsum = table.sum(axis=0)
    rowsum = table.sum(axis=1)
    expect = np.outer(rowsum, colsum) / n
    chisq = np.sum((table - expect) ** 2 / expect)

    return np.sqrt(chisq / (n * (np.min(table.shape) - 1)))


def calc_cramers_corr(sr_subject_feature, df_object_cate): 
    """
    Cramerの連関係数算出
    カテゴリ型×カテゴリ型の特徴量の相関係数をCramerの連関係数で算出
    
    Args:
        sr_subject_feature (pd.Series): 主目的特徴量
        df_object_cate (pd.DataFrame): 相手となるその他のカテゴリ型の特徴量
    
    Returns:
        pd.DataFrame: Cramerの連関係数および相関係数タイプ(連関係数)
    """
    df_cramers_corr = pd.DataFrame(index=df_object_cate.columns, columns=['相関係数', '相関係数タイプ'])
    for object_feature in df_object_cate.columns: 
        sr_object_feature = df_object_cate[object_feature].dropna()
        list_match_index = list(set(sr_subject_feature.index) & set(sr_object_feature.index))
        df_cramers_corr.loc[object_feature, '相関係数'] = cramersV(sr_subject_feature[list_match_index], sr_object_feature[list_match_index])

    df_cramers_corr['相関係数タイプ'] = '連関係数'
    return df_cramers_corr 


def drop_all_null_cols(df_data, num_sample): 
    """
    計算可能カラム抽出
    全サンプル欠損の特徴量以外の特徴量(列名)を抽出
    
    Args:
        df_data (pd.DataFrame): CSVファイルデータ
        num_sample (float): 全サンプル数
    
    Returns:
        pd.Index: 計算可能特徴量リスト
    """
    sr_num_null = df_data.isna().sum()
    query_not_all_null = sr_num_null != num_sample 
    effective_features = sr_num_null[query_not_all_null].index

    return effective_features


def create_label_encoder(sr_feature): 
    """
    ラベルエンコーダ作成
    
    Args:
        sr_feature (pd.Series): 特徴量データ
    
    Returns:
        sklearn.preprocessing.LabelEncoder: 当該特徴量のラベルエンコーダ
    """
    le = LabelEncoder().fit(sr_feature)

    return le

    
def make_correlation(df_data, path_outputdir, num_sample): 
    """
    総当たりで相関係数算出・散布図描画
    
    Args:
        df_data (pd.DataFrame): CSVファイルのデータ
        path_outputdir (pathlib.Path): 出力先フォルダパス
        num_sample (float): 総サンプル数
    """
    num_cols = len(df_data.columns)
    num_fig_rows = ceil(num_cols / 4)
    effective_features = drop_all_null_cols(df_data, num_sample)

    # Xに割り当てる特徴量を総当たり
    list_sr_correlation = list() 
    for subject_feature in effective_features: 
        sr_subject_feature = df_data[subject_feature].dropna()
        dtype_subject = sr_subject_feature.dtype
        object_features = effective_features.drop(subject_feature)

        print('【X軸】')
        print(subject_feature, dtype_subject)

        # 相関係数算出
        # Xに対して、Yの組み合わせを総当たり
        sr_dtype_subject = pd.Series(dtype_subject, index=object_features, name='subject')
        sr_dtype_object = df_data.drop(subject_feature, axis=1).dtypes
        sr_dtype_object.name = 'object'
        df_dtype_combination = pd.concat([sr_dtype_subject, sr_dtype_object], axis=1)

        # 数値・カテゴリの組み合わせで変数名を相関係数の算出方法でグループ分け
        is_numeric_subject = dtype_subject in [int, float]
        list_df_corr_subject = list()
        if is_numeric_subject: 
            # 数値×数値
            query_num_num = (df_dtype_combination['subject'].apply(lambda x: x in [int, float])) & (df_dtype_combination['object'].apply(lambda x: x in [int, float])) 
            list_num_num = df_dtype_combination.iloc[query_num_num.values.tolist(), :].index.tolist()

            # 数値×カテゴリ
            query_num_cate = (df_dtype_combination['subject'].apply(lambda x: x in [int, float])) & ~(df_dtype_combination['object'].apply(lambda x: x in [int, float])) 
            list_num_cate = df_dtype_combination.iloc[query_num_cate.values.tolist(), :].index.tolist()

            df_corr_num_num = calc_pearson_corr(sr_subject_feature, df_data[list_num_num])
            df_corr_num_cate = calc_rate_corr(sr_subject_feature, df_data[list_num_cate], category_col='right')

            list_df_corr_subject.append(df_corr_num_num)
            list_df_corr_subject.append(df_corr_num_cate)

        else: 
            # カテゴリ×数値
            query_cate_num = ~(df_dtype_combination['subject'].apply(lambda x: x in [int, float])) & (df_dtype_combination['object'].apply(lambda x: x in [int, float])) 
            list_cate_num = df_dtype_combination.iloc[query_cate_num.values.tolist(), :].index.tolist()

            # カテゴリ×カテゴリ
            query_cate_cate = ~(df_dtype_combination['subject'].apply(lambda x: x in [int, float])) & ~(df_dtype_combination['object'].apply(lambda x: x in [int, float])) 
            list_cate_cate = df_dtype_combination.iloc[query_cate_cate.values.tolist(), :].index.tolist()

            df_corr_cate_num = calc_rate_corr(sr_subject_feature, df_data[list_cate_num], category_col='left')
            df_corr_cate_cate = calc_cramers_corr(sr_subject_feature, df_data[list_cate_cate])

            list_df_corr_subject.append(df_corr_cate_num)
            list_df_corr_subject.append(df_corr_cate_cate)

        df_corr_subject = pd.concat(list_df_corr_subject, axis=0)
        sr_corr_subject = df_corr_subject['相関係数']
        sr_corr_subject.name = subject_feature
        list_sr_correlation.append(sr_corr_subject)


        # 散布図描画
        fig = plt.figure(figsize=(40,7 * num_fig_rows))

        # 散布図で表示できるようにラベルエンコーディング
        if dtype_subject == 'object': 
            le_subject = create_label_encoder(sr_subject_feature)
            sr_subject_feature = pd.Series(le_subject.transform(sr_subject_feature), index=sr_subject_feature.index)

        print('【Y軸】')
        # Yに割り当てる特徴量をXを除いて総当たり
        for idx, object_feature in enumerate(object_features): 
            sr_subject_feature_tmp = sr_subject_feature.copy()
            sr_object_feature = df_data[object_feature].dropna()
            dtype_object = sr_object_feature.dtype
            print(object_feature, dtype_object)

            # どちらの特徴量にも存在するサンプルのみを抽出
            list_match_index = list(set(sr_subject_feature_tmp.index) & set(sr_object_feature.index))
            if len(list_match_index) == 0:
                continue 

            # 散布図で表示できるようにラベルエンコーディング
            if dtype_object == 'object': 
                le_object = create_label_encoder(sr_object_feature)
                sr_object_feature = pd.Series(le_object.transform(sr_object_feature), index=sr_object_feature.index)

            # 有効なデータで散布図描画
            ax = fig.add_subplot(num_fig_rows,4,idx+1)
            ax.scatter(sr_subject_feature_tmp[list_match_index], sr_object_feature[list_match_index])
            ax.set_title(f'{subject_feature} vs {object_feature}', fontsize=19)
            ax.set_xlabel(subject_feature, fontsize=15)
            ax.set_ylabel(object_feature, fontsize=15)

            # 軸を文字列に復元(LabelEncoderにもともと存在しないラベル値は削除)
            if dtype_subject == 'object': 
                ar_xticks = np.unique(np.asarray(ax.get_xticks(), dtype=np.int32))
                while True: 
                    try: 
                        ar_xticks = ar_xticks[ar_xticks >= 0]
                        ax.set_xticks(ar_xticks)
                        ax.set_xticklabels(le_subject.inverse_transform(ar_xticks))
                        break 

                    except ValueError: 
                        ar_xticks[-1] = ar_xticks[-1] - 1 

            if dtype_object == 'object': 
                ar_yticks = np.unique(np.asarray(ax.get_yticks(), dtype=np.int32))
                while True: 
                    try: 
                        ar_yticks = ar_yticks[ar_yticks >= 0]
                        ax.set_yticks(ar_yticks)
                        ax.set_yticklabels(le_object.inverse_transform(ar_yticks))
                        break 

                    except ValueError: 
                        ar_yticks[-1] = ar_yticks[-1] - 1 

            # 3桁以上変化がある場合にlogスケール変換(数値のみ)
            if dtype_subject != 'object': 
                if sr_subject_feature_tmp[list_match_index].max() // sr_subject_feature_tmp[list_match_index].min() >= 1000: 
                    ax.set_xscale('log')
                    ax.set_xlim([sr_subject_feature_tmp[list_match_index].min() - (sr_subject_feature_tmp[list_match_index].min() / 2), sr_subject_feature_tmp[list_match_index].max() + (sr_subject_feature_tmp[list_match_index].max() / 2)])

            if dtype_object != 'object': 
                if sr_object_feature[list_match_index].max() // sr_object_feature[list_match_index].min() >= 1000: 
                    ax.set_yscale('log')
                    ax.set_ylim([sr_object_feature[list_match_index].min() - (sr_object_feature[list_match_index].min() / 2), sr_object_feature[list_match_index].max() + (sr_object_feature[list_match_index].max() / 2)])

            # 相関係数表示
            str_corr_type = df_corr_subject.loc[object_feature, '相関係数タイプ']
            str_corr_val = str(df_corr_subject.loc[object_feature, '相関係数'])
            str_corr = f'{str_corr_type}: {str_corr_val}'
            ax.text(x=ax.get_xlim()[0], y=ax.get_ylim()[0], s=str_corr, fontsize=12)


        # 全特徴量について描画終了後
        plt.tight_layout()
        subject_feature = subject_feature.replace('/', '／')
        outputfilepath = path_outputdir/(f'{subject_feature}_scatters_correlations.png')
        plt.savefig(outputfilepath, dpi=100)
        plt.close()

    # 相関係数行列出力
    df_correlation = pd.concat(list_sr_correlation, axis=1, sort=False)
    path_corr_output = path_outputdir/'correlation.csv'
    df_correlation.to_csv(path_corr_output, encoding=input_charset)


def analysis_correlation(df_data, path_outputdir):
    num_sample = len(df_data)
    # 散布図描画・相関係数算出
    make_correlation(df_data, path_outputdir, num_sample)


def exec_analysis(list_path_inputfiles, list_path_outputdirs):
    """
    グラフ解析処理
    
    Args:
        list_path_inputfiles (List[pathlib.Path]): 入力ファイルパスリスト
        list_path_outputdirs (List[pathlib.Path]): 出力フォルダパスリスト
    """

    for inputfilepath, outputdirpath in zip(list_path_inputfiles, list_path_outputdirs):
        print(f'入力ファイル名: {inputfilepath}')
        df_data = pd.read_csv(inputfilepath, encoding=input_charset)
        print(f'カラム名: {df_data.columns}')
        print(f'レコード数: {len(df_data)}')
        print('データ例: ')
        print(df_data.head(5))
        analysis_distribution(df_data, outputdirpath)
        analysis_correlation(df_data, outputdirpath)


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

from argparse import ArgumentParser
from logging import getLogger, StreamHandler, DEBUG, INFO, Formatter
from pathlib import Path 

import pandas as pd 

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
formatter = Formatter('%(asctime)s - FILE: %(filename)s - Func: %(funcName)s - L: %(lineno)d - %(levelname)s | %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


parser = ArgumentParser()
parser.add_argument('input_dir', help='input directory path')
parser.add_argument('output_dir', help='output directory path')
args = parser.parse_args()


def get_filepathlist(input_dir):
    logger.info('Start get_filepathlist')
    input_dir_path = Path(input_dir)
    logger.info(f'input_dir: {input_dir_path}')
    list_filepath = list(input_dir_path.glob('*/*_statics.csv'))
    logger.info(f'filepath list: {list_filepath}')
    return list_filepath


def read(filepath):
    logger.info(f'Start read {filepath}')
    df = pd.read_csv(filepath, index_col=0)
    logger.info(f'Success shape: {df.shape}\n{df[:3]}')
    logger.info(f'Colmuns: {df.columns}')
    return df


def conv_columns(filepath, columns):
    logger.info(f'Start conv_columns')
    new_columns = [f'{colname}({filepath.stem.replace("_statics", "")})' for colname in columns]
    logger.info(f'new_columns: {new_columns}')
    return new_columns


def aggregate(list_df, output_dir):
    logger.info('Start aggregate')
    output_dirpath = Path(output_dir)
    output_dirpath.mkdir(exist_ok=True)
    output_filepath = output_dirpath/'statics_aggregated.csv'

    df_aggregated = pd.concat(list_df, axis=1)
    df_aggregated.to_csv(output_filepath, index=True)
    logger.info(f'Success convert. output_filepath: {output_filepath}')


if __name__ == '__main__':
    list_filepath = get_filepathlist(args.input_dir)
    list_df = list()
    for filepath in list_filepath:
        df_statics = read(filepath)
        df_statics.columns = conv_columns(filepath, df_statics.columns)
        list_df.append(df_statics)

    aggregate(list_df, args.output_dir)
    logger.info('All files completed!')

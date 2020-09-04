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
    list_filepath = list(input_dir_path.glob('*.xlsx'))
    logger.info(f'filepath list: {list_filepath}')
    return list_filepath


def read(filepath):
    logger.info(f'Start read {filepath}')
    df = pd.read_excel(filepath)
    logger.info(f'Success shape: {df.shape}\n{df[:3]}')
    return df


def conv_xlsx2csv(df_excel, filepath, output_dir):
    logger.info('Start conv_xlsx2csv')
    output_dirpath = Path(output_dir)
    output_dirpath.mkdir(exist_ok=True)
    output_filepath = output_dirpath/f'{filepath.stem}.csv'
    df_excel.to_csv(output_filepath, index=False)
    logger.info(f'Success convert. output_filepath: {output_filepath}')


if __name__ == '__main__':
    list_filepath = get_filepathlist(args.input_dir)
    for filepath in list_filepath:
        df_excel = read(filepath)
        conv_xlsx2csv(df_excel, filepath, args.output_dir)

    logger.info('All files completed!')

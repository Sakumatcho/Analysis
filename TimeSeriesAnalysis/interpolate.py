from math import floor 

import numpy as np 
import pandas as pd 


class InterPolator():

    def __init__(self, df_data, col_time, unit_time):
        if unit_time not in ['sec', 'min', 'hour']: 
            raise ValueError(f'"unit_time" should be selected from ["sec", "min", "hour"] but now "unit_time": {unit_time}')
        self.col_sampling_time = f'sampling time[{unit_time}]'
        self.df_data = df_data 
        self.col_time = col_time 
        self.min_time = df_data[self.col_time].min() 
        self.max_time = df_data[self.col_time].max() 

    def create_sampling_time_series(self, delta_time): 
        num_sampling = floor((self.max_time - self.min_time) / delta_time) 
        print(f'最小値: {self.min_time}, 最大値: {self.max_time}')
        print(f'サンプリング数: {num_sampling + 1}')

        self.sr_sampling = pd.Series(
            np.arange(self.min_time, (num_sampling + 1) * delta_time, dtype=int),
            name=self.col_sampling_time, 
            dtype=np.float64
        )

        set_all_time = set(self.sr_sampling) | set(self.df_data[self.col_time])
        self.sr_sampling_all = pd.Series(list(set_all_time), name=self.col_sampling_time, dtype=np.float64).sort_values(ascending=True).reset_index(drop=True)

    def interpolate(self, features, max_interpolate=1): 
        df_data_copy = self.df_data.copy() 
        df_sampling = pd.DataFrame(self.sr_sampling_all)
        df_data_dummy = pd.merge(df_sampling, df_data_copy, how='left', left_on=self.col_sampling_time, right_on=self.col_time)
        for feature in features: 
            df_data_dummy[feature] = df_data_dummy[feature].interpolate(
                method='linear', limit_area='inside', limit=max_interpolate
            )
        output_cols = [self.col_sampling_time] + features 
        df_data_interpolated = df_data_dummy[output_cols]

        return df_data_interpolated[df_data_interpolated[self.col_sampling_time].isin(self.sr_sampling)].copy()


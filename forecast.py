#!/usr/bin/env python3

import numpy as np
from utils.utils_vision_data import * ; from utils.data_processing import * ;

y, _= prepare_tabular_data_vision(min_wind=34, min_steps=20, max_steps=120, path = 'since1980.csv')

np.save('./data/y_1980_34_20_120_forecast_test.npy', y, allow_pickle=True)

from src.helpers.config import train_data_path, test_data_path, output_base_path, item_metadata_path, outlet_metadata_path
from src.helpers.DataReader import filereader

import numpy as np


df_train = filereader(train_data_path)

vis = df_train['Item_Visibility'].replace(0, np.nan)

sample = df_train.assign(Item_Visibility=vis).groupby('Item_Identifier')["Item_Visibility"].mean()
sample

vis.median()

df_train['Item_category'] = df_train['Item_Identifier'].str[:2]
df_train.info()
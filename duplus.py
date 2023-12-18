import pandas as pd
import os
import glob
import numpy as np

base_path = "d:\\AI\\bossbase" 
new_data_path = glob.glob(os.path.join(base_path, 'boin_*.csv'))
latest = max(new_data_path, key=os.path.getctime)

data = pd.read_csv('d:/AI/bossbase/boin.csv')
new_data = pd.read_csv(latest)

result = pd.concat([data, new_data])

columns_to_process = ['岗位名称', '区位', '薪资',"工作经验","学历","企业名称","关键词1","更新时间"]
result = result.drop_duplicates(subset=columns_to_process)

data = data.replace(to_replace='None', value=np.nan)
data.dropna(subset=['岗位要求'], inplace=True)

result = result.reset_index(drop=True)
result.to_csv('d:/AI/bossbase/boin.csv', index=False)
# os.remove('d:/AI/boss.csv')
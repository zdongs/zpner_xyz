import pandas as pd
import numpy as np
import datetime
import os

def save_df_to_csv(df, filename):
    # 获取当前时间戳
    now = datetime.datetime.now()
    timestamp = now.strftime("%m%d%H%M")

    # 将文件名加上时间戳
    filename = f"d:/AI/bossbase/{filename}_{timestamp}.csv"

    # 保存 df 到 CSV 文件
    df.to_csv(filename, index=False)

data = pd.read_csv('d:/AI/boss.csv')

columns_to_process = ['岗位名称', '区位', '薪资',"工作经验","学历","企业名称","关键词1","链接"]
data = data.drop_duplicates(subset=columns_to_process)

data = data.replace(to_replace='None', value=np.nan)
data.dropna(subset=['岗位名称', '区位', '薪资'], inplace=True)

data = data.reset_index(drop=True)

save_df_to_csv(data, "boss")
os.remove('d:/AI/boss.csv')

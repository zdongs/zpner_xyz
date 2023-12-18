import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

class DataAnalyzer:
    def __init__(self, data_file_path):
        # custom_header = ['岗位名称', '区位', '薪资', '工作经验', '学历', '关键词1', '关键词2', '关键词3', '企业名称',
    #    '企业类别', '企业规模', '岗位要求', '更新时间']
        # self.alldata = pd.read_csv(data_file_path,encoding='GBK',header=None,names=custom_header)
        self.alldata = pd.read_csv(data_file_path)
        self.data = self.alldata

    def filter_data(self, filter_name, filter_way=False):
        condition = self.alldata['岗位名称'].str.contains(filter_name)
        if filter_way:
            self.data = self.alldata[condition]
        else:
            self.data = self.alldata[~condition]

    def stand_salary(self):
        def convert_salary(sal):
            match = re.match(r'^(\d{4,})-(\d{4,})元/月$', str(sal))
            if match:
                start, end = match.groups()
                return f'{int(int(start) / 1000)}-{int(int(end) / 1000)}K'
            return sal

        self.data['薪资'] = self.data['薪资'].apply(convert_salary)

    def quant_salary(self):
        def extract_numbers(salary_range):
            match = re.match(r'^(\d{1,3})-(\d{1,3})K(?:·(\d+)薪)?$', str(salary_range))
            if match:
                start, end, bonus = match.groups()
                return int(start), int(end), int(bonus) if bonus else 0
            return 0, 0, 0

        sal2low = {}
        for i in self.data['薪资']:
            j, _, _ = extract_numbers(i)
            sal2low[i] = int(j)
        
        self.data["薪资"] = self.data["薪资"].map(sal2low)

    def quant_work_exp(self):
        exp2inx = {'经验不限':0, '应届生':1, '1年以内':2, '1-3年':3, '3-5年':4, '5-10年':5, '10年以上':6}
        self.data['工作经验'] = self.data['工作经验'].map(exp2inx)

    def quant_update_time(self):
        time_set = set(self.data['更新时间'])

        def extract_time(time_range):
            match = re.match(r'.+(\d{2})-(\d{2})$', str(time_range))
            if match:
                start, end = match.groups()
                return int(start), int(end)
            return 0, 0

        sorted_time = sorted(time_set, key=lambda x: (extract_time(x)))

        time2inx = {k:v for v,k in enumerate(sorted_time)}
        self.data['更新时间'] = self.data['更新时间'].map(time2inx)

    def quant_scale(self):
        scale_set = set(self.data['企业规模'])

        def extract_scale(scale):
            match = re.match(r'^(\d+)',str(scale))
            if match:
                return int(match.groups()[0])
            return -1

        sorted_scale = sorted(scale_set, key=lambda x: extract_scale(x))
        sca2inx = {k:v for v,k in enumerate(sorted_scale)}
        self.data['企业规模'] = self.data['企业规模'].map(sca2inx)

    def quant_edu(self):
        edu2inx = {'学历不限':0, '大专':1, '本科':2, '硕士':3, '博士':4}
        self.data['学历'] = self.data['学历'].map(edu2inx)
    
    def quant_all(self):
        self.quant_scale()
        self.quant_edu()
        self.quant_salary()
        self.quant_update_time()
        self.quant_work_exp()

    def sand_salary(self, b_name, c_name):
        salaries = self.data['薪资'].values.reshape(-1, 1)
        b = self.data[b_name].values.reshape(-1, 1)
        c = self.data[c_name].values.reshape(-1, 1)

        data_3d = np.hstack(( b, c,salaries,))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=data_3d[:, 2], cmap='viridis')
        ax.set_xlabel(b_name)
        ax.set_ylabel(c_name)
        ax.set_zlabel('起薪')

        plt.title('起薪分布三维点图')

        plt.show()

if __name__ == "__main__":
    # 使用示例
    data_file_path = 'd:/AI/bossbase/boin.csv'
    # data_file_path = "d:/AI/bossbase/猎聘_招聘信息.csv"
    analyzer = DataAnalyzer(data_file_path)
    analyzer.filter_data("数据挖掘")
    analyzer.stand_salary()
    analyzer.quant_salary()
    analyzer.quant_edu()
    analyzer.quant_scale()
    analyzer.quant_work_exp()
    # 更多的量化方法可以根据需要调用
    # print(analyzer.data)
    analyzer.sand_salary('企业规模', '工作经验') # 热力图
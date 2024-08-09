import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, argrelmin
from sklearn.preprocessing import StandardScaler






def generatenew(df):
    def count_peaks(data):
        """计算每个样本的极大值个数"""
        peaks_count = data.apply(lambda x: len(find_peaks(x)[0]), axis=1)
        return peaks_count

    def max_peak_position(data):
        """找到每个样本中最大值的位置"""
        max_position = data.idxmax(axis=1)
        return max_position

    def std_deviation(data):
        """计算每个样本的标准差"""
        std_dev = data.std(axis=1)
        return std_dev

    def total_integral(data):
        """计算每个样本的总积分"""
        total_int = data.sum(axis=1)
        return total_int

    def count_valleys(data):
        """计算每个样本的极小值个数"""
        valleys_count = data.apply(lambda x: len(find_peaks(-x)[0]), axis=1)
        return valleys_count

    def min_peak_position(data):
        """找到每个样本中最小值的位置"""
        min_position = data.idxmin(axis=1)
        return min_position

    def calculate_skewness(data):
        """计算每个样本的偏度"""
        skewness = data.apply(skew, axis=1)
        return skewness

    def calculate_kurtosis(data):
        """计算每个样本的峰度"""
        kurtosis_ = data.apply(kurtosis, axis=1)
        return kurtosis_

    def calculate_entropy(data):
        """计算每个样本的信息熵"""
        # 需要先归一化数据到 [0, 1] 区间内
        normalized_data = (data - data.min()) / (data.max() - data.min())
        # 使用 scipy 的 entropy 函数计算信息熵
        entropy_values = normalized_data.apply(lambda x: entropy(x), axis=1)
        return entropy_values

    def calculate_gini_coefficient(data):
        """计算每个样本的 Gini 系数"""
        gini_values = data.apply(lambda x: gini(x), axis=1)
        return gini_values

    # 定义 Gini 系数计算函数
    def gini(x):
        """计算单个样本的 Gini 系数"""
        unique_values, counts = np.unique(x, return_counts=True)
        cumulative_sum = np.cumsum(np.sort(counts)) - counts
        gini_value = 1 - sum((counts / len(x)) ** 2) + (cumulative_sum * (counts / len(x))).sum() / (
                    len(x) * (len(x) - 1))
        return gini_value

    data = df.copy()
    data = data.drop(labels=["Label", "ID", "Color_1", "Color_2", "Color_3", "Color_4"], axis=1)

    data1 = data.copy()
    data2 = data.copy()
    data3 = data.copy()
    data4 = data.copy()
    data5 = data.copy()
    data6 = data.copy()
    data7 = data.copy()
    data8 = data.copy()
    data9 = data.copy()
    data10 = data.copy()


    # 计算特征并添加到 DataFrame
    data['peaks_count'] = count_peaks(data1)
    data['max_peak_position'] = max_peak_position(data2)
    data['std_deviation'] = std_deviation(data3)
    data['total_integral'] = total_integral(data4)
    data['valleys_count'] = count_valleys(data5)
    data['min_peak_position'] = min_peak_position(data6)
    data['skewness'] = calculate_skewness(data7)
    data['kurtosis'] = calculate_kurtosis(data8)
    data['entropy'] = calculate_entropy(data9)
    data['gini_coefficient'] = calculate_gini_coefficient(data10)

    # 添加原始列
    data.insert(0, 'ID', df['ID'])
    data.insert(1, 'Label', df['Label'])
    data.insert(2, 'Color_1', df['Color_1'])
    data.insert(3, 'Color_2', df['Color_2'])
    data.insert(4, 'Color_3', df['Color_3'])
    data.insert(5, 'Color_4', df['Color_4'])
    data['max_peak_position'] = pd.to_numeric(data['max_peak_position'], errors='coerce')
    data['min_peak_position'] = pd.to_numeric(data['min_peak_position'], errors='coerce')

    # 标准化特征
    columns_to_standardize = ['peaks_count', 'max_peak_position', 'std_deviation',
                              'total_integral', 'valleys_count', 'min_peak_position',
                              'skewness', 'kurtosis', 'entropy', 'gini_coefficient']
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(data[columns_to_standardize])
    df_standardized = pd.DataFrame(df_standardized, columns=columns_to_standardize, index=data.index)
    data[columns_to_standardize] = df_standardized

    print(data)

    return data





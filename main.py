import pandas as pd
from sklearn.model_selection import train_test_split

import FeatrueSelect
import Model_Classify
import Read_Raman_spectrum
import processing

RamanFilename_list = ['Raman_spectrum_BK.npy', 'Raman_spectrum_C(1).npy', 'Raman_spectrum_M(1).npy',
                      'Raman_spectrum_Y(1).npy']
# RamanFilename_list = ['Raman_spectrum_BK.npy']
# RamanFilename_list = ['Raman_spectrum_C(1).npy', 'Raman_spectrum_M(1).npy',
#                       'Raman_spectrum_Y(1).npy']
LabelFilename_list = ['FN_BK(1).npy', 'FN_C(1).npy', 'FN_M(1).npy', 'FN_Y(1).npy']
# LabelFilename_list = ['FN_BK(1).npy']
# LabelFilename_list = ['FN_C(1).npy', 'FN_M(1).npy', 'FN_Y(1).npy']

df_encoded = Read_Raman_spectrum.readtodf( RamanFilename_list, LabelFilename_list)

# 先滤波后生成新的指标/先生成指标后滤波
# df_encoded = FeatrueSelect.flitering_beforege(df_encoded)
df_encoded = FeatrueSelect.generatenew(df_encoded)

# 滤波平滑
df_encoded = processing.flitering(df_encoded)

# 基线调整
df_encoded = processing.baseline(df_encoded)


# pca/tsne降维
# df_encoded = perform_pca(df_encoded)
# df_encoded = perform_tsne(df_encoded)


pd.set_option('display.max_columns', None)
print(df_encoded)

unique_labels = df_encoded['Label'].unique()
# 打印这些标签集合
print(len(unique_labels))

df_encoded.columns = df_encoded.columns.astype(str)



# 划分特征数据和标签
y = df_encoded["Label"]
X = df_encoded.drop(labels=["Label", "ID"], axis=1)


print(X.columns)


# 定义区间的索引，可以自由选用全数据集或者是部分颜色的数据集
intervals = [(0, 559), (559, 1092), (1092, 1625), (1625, 2158)]
# intervals = [(559, 1092), (1092, 1625), (1625, 2158)]
# intervals = [(0, 559)]

# 初始化空列表来存储分割后的数据集
X_train_splits = []
X_val_splits = []
y_train_splits = []
y_val_splits = []

# 对每个区间进行训练集和验证集的分割
for start, end in intervals:
    X_interval = X.iloc[start:end]
    y_interval = y.iloc[start:end]

    # 分割当前区间的数据
    X_train, X_val, y_train, y_val = train_test_split(
        X_interval, y_interval, test_size=0.25, random_state=42, stratify=y_interval
    ) # 这里固定random_state的值可以保证每次划分的数据集都是一样的

    # 将当前区间的分割结果添加到列表中
    X_train_splits.append(X_train)
    X_val_splits.append(X_val)
    y_train_splits.append(y_train)
    y_val_splits.append(y_val)

# 合并所有区间的分割结果，形成最终的训练集和验证集
X_train_split = pd.concat(X_train_splits)
X_val = pd.concat(X_val_splits)
y_train_split = pd.concat(y_train_splits)
y_val = pd.concat(y_val_splits)

# 现在 X_train_split, X_val, y_train_split, y_val 已经准备好，且保证了每个区间的分割比例一致

# 选用分类模型
# Model_Classify.xgboost(X_train_split, X_val, y_train_split, y_val)
bestmodel = Model_Classify.lightgbm2(X_train_split, X_val, y_train_split, y_val)
# Model_Classify.catboost(X, y)
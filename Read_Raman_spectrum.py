import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder

import FeatrueSelect
import Model_Classify


def perform_tsne(df):
    copy = df.copy()
    copy = copy.drop(columns=['ID', 'Color', 'Label'])
    X = copy.values

    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_standardized)
    df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])

    # 将 ID 列添加回 PCA 结果 DataFrame
    df_tsne.insert(0, 'ID', df['ID'])
    df_tsne.insert(1, 'Color', df['Color'])
    df_tsne.insert(2, 'Label', df['Label'])
    print(df_tsne)
    return df_tsne


def perform_pca(df):  ## 输入df，返回一个pca梳理后的df

    X = df.drop(
        labels=["Label", "ID", "Color_1", "Color_2", "Color_3", "Color_4", 'peaks_count', 'max_peak_position',
                'std_deviation', 'total_integral'], axis=1)
    # 标准化数据（对 PCA 很重要）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 创建 PCA 实例，不预先设置 n_components，让 PCA 根据方差解释率来确定维数
    pca = PCA(n_components=0.95)  # 保留至少解释 95% 的总方差

    # 对数据进行 PCA 处理
    X_pca = pca.fit_transform(X_scaled)

    # 查看保留了多少主成分
    print(f"保留的主成分数: {pca.n_components_}")

    # 将 PCA 结果转换为 DataFrame
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i + 1}' for i in range(pca.n_components_)])

    # 将 ID 列添加回 PCA 结果 DataFrame
    pca_df.insert(0, 'ID', df['ID'])
    pca_df.insert(1, 'Label', df['Label'])
    pca_df.insert(2, 'Color_1', df['Color_1'])
    pca_df.insert(3, 'Color_2', df['Color_2'])
    pca_df.insert(4, 'Color_3', df['Color_3'])
    pca_df.insert(5, 'Color_4', df['Color_4'])

    pca_df.insert(6, 'peaks_count', df['peaks_count'])
    pca_df.insert(7, 'max_peak_position', df['max_peak_position'])
    pca_df.insert(8, 'std_deviation', df['std_deviation'])
    pca_df.insert(9, 'total_integral', df['total_integral'])
    # 显示结果
    return pca_df

def normalize_dataframe(df):
    # 移除 'ID' 和 'Color' 列
    copy = df.copy()
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    if 'Color' in df.columns:
        df = df.drop(columns=['Color'])

    # 实例化 MinMaxScaler
    scaler = MinMaxScaler()

    # 拟合并转换数据
    scaled_data = scaler.fit_transform(df)

     # 将标准化后的数据转换回DataFrame
    normalized_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    normalized_df.insert(0, 'ID', copy['ID'])
    normalized_df.insert(1, 'Color', copy['Color'])

    return normalized_df


def readtodf(RamanFilename_list, LabelFilename_list):
    OriginalData_list = []
    # 处理特征数据
    index = 1
    FileIndex = 1

    array = np.load('Raman_spectrum_BK(1).npy', allow_pickle=True)
    spectral_container = array.item()
    spectra_list = spectral_container.tolist()
    data_column = spectra_list[0].spectral_axis.tolist()
    # print(data_column)
    data_column.insert(0, "ID")  # 在开头加入序号
    data_column.insert(1, "Color")  # 在开头加入序号

    for f in RamanFilename_list:
        array = np.load(f, allow_pickle=True)
        spectral_container = array.item()
        spectra_list = spectral_container.tolist()
        print("len")
        print(len(spectra_list))

        # print(len(data_column))
        for spectrum in spectra_list:
            # print(help(spectrum))
            mylist = spectrum.spectral_data.tolist()  # 变成list

            mylist.insert(0, index)  # 在开头加入序号
            mylist.insert(1, FileIndex)  # 第2列添加墨水颜色
            index += 1
            OriginalData_list.append(mylist)  # 将其添加至整体数据的list

        FileIndex += 1

    df = pd.DataFrame(OriginalData_list, columns=data_column)
    print(df.shape[0])

    df = normalize_dataframe(df)

    # 处理标签数据
    label_list = []
    for f in LabelFilename_list:
        array = np.load(f, allow_pickle=True)
        label = array.tolist()

        label_list.extend(label)

    print(len(label_list))
    df.insert(loc=2, column='Label', value=label_list)

    # df = perform_pca(df)

    # df = perform_tsne(df)

    # 执行独热编码

    df_encoded = pd.get_dummies(df, columns=['Color'])

    for column in df_encoded.columns:
        if df_encoded[column].dtype == 'bool':
            df_encoded[column] = df_encoded[column].astype(int)

    # 创建 LabelEncoder 对象
    label_encoder = LabelEncoder()

    # 应用 LabelEncoder
    df_encoded['Label'] = label_encoder.fit_transform(df_encoded['Label'])

    return df_encoded




def randomseq(df):
    if len(df) >= 2158:
        # 对不同的行段进行随机重排
        segment1 = df.iloc[:559].sample(frac=1).reset_index(drop=True)
        segment2 = df.iloc[559:1092].sample(frac=1).reset_index(drop=True)
        segment3 = df.iloc[1092:1625].sample(frac=1).reset_index(drop=True)
        segment4 = df.iloc[1625:2158].sample(frac=1).reset_index(drop=True)

        # 将所有随机重排的段合并回一个新的 DataFrame
        new_df = pd.concat([segment1, segment2, segment3, segment4], ignore_index=True)
        # 为 'ID' 列赋新值，从 1 开始，直到 new_df 的行数
        new_df['ID'] = range(1, len(new_df) + 1)

    else:
        new_df = df
        print("DataFrame 的行数少于 2158 行，请添加更多数据。")

    return new_df




def compare_lists(list1, list2):
    if len(list1) != len(list2):
        print("列表长度不同，无法比较。")
        return

    differences = []
    for index, (item1, item2) in enumerate(zip(list1, list2)):
        if item1 != item2:
            differences.append((index, item1, item2))

    if differences:
        print("发现不同的元素：")
        for diff in differences:
            print(f"位置 {diff[0]}: list1 中是 {diff[1]}, list2 中是 {diff[2]}")
    else:
        print("两个列表完全相同。")



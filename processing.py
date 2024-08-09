import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

def rfe(df):
    train_df = df.copy()
    Y = train_df["Label"]
    X = train_df.drop(labels=["Label","ID"], axis=1)

    X_train_split, X_val, y_train_split, y_val = train_test_split(X, Y, test_size=0.25)  # 数据切分

    # print(X_train_split)
    # print(y_train_split)

    train_matrix = xgb.DMatrix(X_train_split, label=y_train_split)  # 测试集

    valid_matrix = xgb.DMatrix(X_val, label=y_val)  # 验证集

    # # 假设 X_train 和 y_train 是你的训练数据
    # log_reg = LogisticRegression(max_iter=2000)  # 使用简单的线性模型

    # 使用 RFE 进行特征选择
    xgb_model = XGBClassifier(objective='multi:softmax', num_class=48, eval_metric='mlogloss')
    print("yes")
    rfe = RFE(estimator=xgb_model, n_features_to_select=100, step=100)
    rfe = rfe.fit(X_train_split, y_train_split)
    print("yes")

    # 获取选中的特征
    selected_features = df.columns[rfe.support_]
    print("Selected features:", selected_features)






def flitering(df):
    train_df = df.copy()
    # X = train_df.drop(labels=["Label", "ID", "Color_1", "Color_2", "Color_3", "Color_4"], axis=1)
    X = train_df.drop(labels=["Label", "ID", "Color_1", "Color_2", "Color_3", "Color_4", 'peaks_count', 'max_peak_position', 'std_deviation','total_integral', 'valleys_count', 'min_peak_position',
                'skewness', 'kurtosis', 'entropy', 'gini_coefficient'], axis=1)



    # X = train_df.drop(labels=["Label", "ID", "Color_1"], axis=1)
    # # 初始化 StandardScaler
    # scaler = StandardScaler()

    # # 拟合并转换数据
    # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


    # 定义高斯滤波函数
    def apply_gaussian_filter(row, sigma=10):
        """对DataFrame的每行应用高斯滤波"""
        return gaussian_filter1d(row, sigma=sigma)

    def apply_savgol_filter(df, window_length, polyorder):
        """
        对DataFrame中的每一列应用Savitzky-Golay滤波。
        参数:
        df (pd.DataFrame): 输入的DataFrame，每一列是一个需要滤波的数据系列。
        window_length (int): 滤波器的窗口长度，必须是正奇数。
        polyorder (int): 拟合多项式的阶数，必须小于window_length。

        返回:
        pd.DataFrame: 滤波后的DataFrame。
        """
        # 检查窗口长度是否为正奇数
        if window_length % 2 == 0:
            raise ValueError("Window length must be a positive odd number")
        # 检查多项式阶数是否小于窗口长度
        if polyorder >= window_length:
            raise ValueError("Polyorder must be less than window length")

        # 对DataFrame的每一列应用Savitzky-Golay滤波
        filtered_df = df.apply(lambda x: savgol_filter(x, window_length, polyorder), axis=0)

        return filtered_df

    # 应用Savitzky-Golay滤波
    # filtered_df = df.apply(lambda row: apply_savgol_filter(row), axis=1)
    filtered_df = apply_savgol_filter(X, window_length=35, polyorder=2)

    # # 应用高斯滤波
    # filtered_df = X.apply(lambda row: apply_gaussian_filter(row), axis=1)

    # 取出第一个样本的原始数据和滤波后的数据
    original_signal = X.iloc[1]
    filtered_signal = filtered_df.iloc[1]

    #filtered_df = pd.DataFrame(filtered_df.tolist(), index=df.index)

    print(filtered_df.dtypes)
    filtered_df.insert(0, 'ID', df['ID'])
    filtered_df.insert(1, 'Label', df['Label'])
    filtered_df.insert(2, 'Color_1', df['Color_1'])
    filtered_df.insert(3, 'Color_2', df['Color_2'])
    filtered_df.insert(4, 'Color_3', df['Color_3'])
    filtered_df.insert(5, 'Color_4', df['Color_4'])

    filtered_df.insert(6, 'peaks_count', df['peaks_count'])
    filtered_df.insert(7, 'max_peak_position', df['max_peak_position'])
    filtered_df.insert(8, 'std_deviation', df['std_deviation'])
    filtered_df.insert(9, 'total_integral', df['total_integral'])

    filtered_df.insert(10, 'valleys_count', df['valleys_count'])
    filtered_df.insert(11, 'min_peak_position', df['min_peak_position'])
    filtered_df.insert(12, 'skewness', df['skewness'])
    filtered_df.insert(13, 'kurtosis', df['kurtosis'])
    filtered_df.insert(14, 'entropy', df['entropy'])
    filtered_df.insert(15, 'gini_coefficient', df['gini_coefficient'])


    # print(filtered_df)
    return filtered_df

    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(original_signal, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.subplot(1, 2, 2)

    plt.plot(filtered_signal, label='Filtered Signal', color='r')
    plt.title('Filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.suptitle('Signal Filtering Comparison')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the subplots to provide space for the suptitle
    plt.show()





def flitering_beforege(df):
    train_df = df.copy()
    # print(train_df)
    Y = train_df["Label"]
    # X = train_df.drop(labels=["Label", "ID", "Color_1", "Color_2", "Color_3", "Color_4"], axis=1)
    X = train_df.drop(labels=["Label", "ID", "Color_1", "Color_2", "Color_3", "Color_4"], axis=1)



    # X = train_df.drop(labels=["Label", "ID", "Color_1"], axis=1)
    # # 初始化 StandardScaler
    # scaler = StandardScaler()

    # # 拟合并转换数据
    # X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


    # 定义高斯滤波函数
    def apply_gaussian_filter(row, sigma=10):
        """对DataFrame的每行应用高斯滤波"""
        return gaussian_filter1d(row, sigma=sigma)


    # 应用高斯滤波
    filtered_df = X.apply(lambda row: apply_gaussian_filter(row), axis=1)

    # 取出第一个样本的原始数据和滤波后的数据
    original_signal = X.iloc[1]
    filtered_signal = filtered_df.iloc[1]

    filtered_df = pd.DataFrame(filtered_df.tolist(), index=df.index)

    print(filtered_df.dtypes)
    filtered_df.insert(0, 'ID', df['ID'])
    filtered_df.insert(1, 'Label', df['Label'])
    filtered_df.insert(2, 'Color_1', df['Color_1'])
    filtered_df.insert(3, 'Color_2', df['Color_2'])
    filtered_df.insert(4, 'Color_3', df['Color_3'])
    filtered_df.insert(5, 'Color_4', df['Color_4'])



    # print(filtered_df)
    return filtered_df

    # 绘制对比图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(original_signal, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.subplot(1, 2, 2)

    plt.plot(filtered_signal, label='Filtered Signal', color='r')
    plt.title('Filtered Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.suptitle('Signal Filtering Comparison')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the subplots to provide space for the suptitle
    plt.show()


def aspls(y, lam=1e5, ratio=0.05, max_iter=10):
    L = len(y)
    D = np.diff(np.eye(L), 2)  # D 的维度是 (L-2, L)
    W = np.ones(L)
    for i in range(max_iter):
        W = np.diag(W)
        Z = W + lam * (np.pad(D.T @ D, ((1, 1), (1, 1)), 'constant', constant_values=0))  # 填充 D.T @ D
        z = np.linalg.solve(Z, W @ y)
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        W = 1 / (1 + np.exp(2 * (d - (2 * s - m)) / s))
    return z

# 进行基线调整
def baseline(df):
    spectra_data = df.copy()
    # X = train_df.drop(labels=["Label", "ID", "Color_1", "Color_2", "Color_3", "Color_4"], axis=1)
    spectra_data = spectra_data.drop(labels=["Label", "ID", "Color_1", "Color_2", "Color_3", "Color_4", 'peaks_count', 'max_peak_position', 'std_deviation','total_integral', 'valleys_count', 'min_peak_position',
                'skewness', 'kurtosis', 'entropy', 'gini_coefficient'], axis=1).values


    # 初始化处理后的数据
    baseline_corrected_data = np.zeros_like(spectra_data)
    print(spectra_data)

    # 对每一条光谱数据进行ASPLS基线校正
    for i, spectrum in enumerate(spectra_data):
        print(i)
        print(spectrum)
        baseline = aspls(spectrum)
        baseline_corrected = spectrum - baseline
        baseline_corrected_data[i] = baseline_corrected

    processed_df = pd.DataFrame(baseline_corrected_data, columns=df.columns[1:])

    processed_df.insert(0, 'ID', df['ID'])
    processed_df.insert(1, 'Label', df['Label'])
    processed_df.insert(2, 'Color_1', df['Color_1'])
    processed_df.insert(3, 'Color_2', df['Color_2'])
    processed_df.insert(4, 'Color_3', df['Color_3'])
    processed_df.insert(5, 'Color_4', df['Color_4'])

    processed_df.insert(6, 'peaks_count', df['peaks_count'])
    processed_df.insert(7, 'max_peak_position', df['max_peak_position'])
    processed_df.insert(8, 'std_deviation', df['std_deviation'])
    processed_df.insert(9, 'total_integral', df['total_integral'])

    processed_df.insert(10, 'valleys_count', df['valleys_count'])
    processed_df.insert(11, 'min_peak_position', df['min_peak_position'])
    processed_df.insert(12, 'skewness', df['skewness'])
    processed_df.insert(13, 'kurtosis', df['kurtosis'])
    processed_df.insert(14, 'entropy', df['entropy'])
    processed_df.insert(15, 'gini_coefficient', df['gini_coefficient'])

    return processed_df



import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

def load_data(file_path, data_type):
    if data_type == 0:
        # カンマ区切り（CSV）の場合
        df = pd.read_csv(file_path)
    elif data_type == 1:
        # タブ区切りの場合
        df = pd.read_csv(file_path, sep="\t")
    else:
        # スペース区切りの場合
        df = pd.read_csv(file_path, delim_whitespace=True)
    return df

if __name__ == '__main__':

    file_path = "prostate.data"
    data_type = 1 # 0 comma, 1 tab, 2 space
    df = load_data(file_path, data_type)
    # 1列目を除く
    df = df.iloc[:, 1:]
    # データの確認
    if df is not None:
        print(df.head())
    df = df.drop(columns=["train"])
    data = df

    # 相関係数行列の計算
    corr_matrix = data.corr()

    # プロット
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of prostate.data')
    plt.tight_layout()
    plt.show()
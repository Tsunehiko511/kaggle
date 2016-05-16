# coding:utf-8

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

# csvファイルの読み込み
df = pd.read_csv('./data/train.csv', header=0)
# 平均
#df["Age"].mean()
# 中央値
#df["Age"].median()
#複数の列にまとめてアクセスする方法
#df[["Sex", "Pclass", "Age"]]
# フィルターされたデータ内の特定の列だけみたい場合
#df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]

# 階級ごとの男性の数を出す
'''
for i in range(1,4):
	print i, len(df[(df['Sex'] == 'male') & (df['Pclass'] == i)])
'''

#df["Age"].hist()
#df["Age"].dropna().hist(bins=16, range=(0, 80), alpha=0.5)
#plt.show()

# pandasでは新しいカラムを追加するのは新しいカラム名に値を代入するだけ
df["Gender"] = 4  # 元はSex
df.head(10)  # Genderカラムが増えている
# coding:utf-8
import csv as csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings;
with warnings.catch_warnings():
    warnings.simplefilter("ignore"); 
    import matplotlib.pyplot as plt


# 下ごしらえ
def df_cleaner(df):
    # 足りない部分は補おう
    # 年齢
    median_age = np.median(df[(df['Age'].notnull())]['Age'])
    for passenger in df[(df['Age'].isnull())].index: #.index = 配列内のnullの場所
    	df.loc[passenger, 'Age'] = median_age
    # fare
    median_fare = np.median(df[(df['Fare'].notnull())]['Fare'])
    for passenger in df[(df['Fare'].isnull())].index:
        df.loc[passenger, 'Fare'] = median_fare

    # 文字列データを数値データへ
    df.loc[(df['Sex'] == 'male'),'Sex'] = 0
    df.loc[(df['Sex'] == 'female'),'Sex'] = 1
    df.loc[(df['Sex'].isnull()),'Sex'] = 2
    df.loc[(df['Embarked'] == 'S'),'Embarked'] = 0
    df.loc[(df['Embarked'] == 'C'),'Embarked'] = 1
    df.loc[(df['Embarked'] == 'Q'),'Embarked'] = 2
    df.loc[(df['Embarked'].isnull()),'Embarked'] = 3

    return df

# 提出用csvを作りましょ
def make_csv(file_path, passengerId, predicts):
    f = open(file_path, "wb")
    writer = csv.writer(f)
    writer.writerow(["PassengerId", "Survived"])
    for row, survived in zip(passengerId, predicts):
        writer.writerow([row, survived])

# 作ったモデルの性能をしれべましょ
def getScore(answer, predicts):
    sum_p = 0.0
    total = 0.0
    for (row, predict) in zip(answer,predicts):
        if row == predict:
            sum_p += 1.0
        total += 1.0
    return sum_p/total

def main():
    # Read in the training data.
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv("./data/test.csv")
    # いらないデータ(予想)は消しましょう
    train.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # 下ごしらえをする
    train = df_cleaner(train)
    test = df_cleaner(test)
    x_train = train[:][['Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y_train = train[:][['Survived']]
    # ランダムフォレストでモデルを作ろう
    scores =[]
    for trees in range(1,100):
        model = RandomForestClassifier(n_estimators=trees)
        model.fit(x_train, np.ravel(y_train))
        # 一致率を見よう
        pre = model.predict(x_train)
        scores.append(getScore(y_train['Survived'],pre))
    plt.plot(scores,'-r')
    plt.show()
    
    # 本番のテストデータをお化粧直し
    x_test = test[:][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    label = test[:][['PassengerId']]
    # modelを使って予測しよう
    output = model.predict(x_test)
    # 提出用csvを作ろう
    make_csv("./output/random_forest.csv", label['PassengerId'], output.astype(int))

if __name__ == '__main__':
    main()
# coding:utf-8
import csv as csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 下ごしらえ
def df_cleaner(df):
    """
    Clean up a few variables in the training/test sets.
    """
    # Clean up ages. 
    median_age = np.median(df[(df['Age'].notnull())]['Age'])

    for passenger in df[(df['Age'].isnull())].index: #.index = 配列内のnullの場所
    	df.loc[passenger, 'Age'] = median_age

    # Clean up fares.
    median_fare = np.median(df[(df['Fare'].notnull())]['Fare'])
    for passenger in df[(df['Fare'].isnull())].index:
        df.loc[passenger, 'Fare'] = median_fare

    # 文字列データを数値データへ
    # Manually convert values to numeric columns for clarity.
    # Change the sex to a binary column.
    df['Sex'][(df['Sex'] == 'male')] = 0
    df['Sex'][(df['Sex'] == 'female')] = 1
    df['Sex'][(df['Sex'].isnull())] = 2

    # Transform to categorical data.
    df['Embarked'][(df['Embarked'] == 'S')] = 0
    df['Embarked'][(df['Embarked'] == 'C')] = 1
    df['Embarked'][(df['Embarked'] == 'Q')] = 2
    df['Embarked'][(df['Embarked'].isnull())] = 3

    return df


def main():
    """
    Visualization of random forest accuracy as function of
    the number of tress available in the ensemble.
    """

    # Read in the training data.
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv("./data/test.csv")

    # Set sampling.(75%をサンプルデータに)
    sampling = 0.75

    # Clean it up.
    # Remove unused columns, clean age, and convert gender to binary column.
    # いらないデータは消す（名前，Id，チケット代，Cabin?） 一次元で適切に
    train.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
    test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    # 下ごしらえをする
    train = df_cleaner(train)
    test = df_cleaner(test)
    # Split it into coordinates.
    # 75%までの配列と残り25%の配列のtrainとtestを用意
    x_train = train[:int(len(train) * sampling)][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y_train = train[:int(len(train) * sampling)][['Survived']]
    x1_train = train[int(len(train) * sampling):][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y1_train = train[int(len(train) * sampling):][['Survived']]
    
    x_test = test[:int(len(test))][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    label = test[:int(len(test))][['PassengerId']]
    model = RandomForestClassifier(n_estimators=200) 		# 決定木=treesの数作る
    model.fit(x_train, np.ravel(y_train))
    pre = model.predict(x1_train)
    # print y1_train['Survived']
    sum_p = 0.0
    total = 0.0
    for (row, predict) in zip(y1_train['Survived'],pre):
    	if row == predict:
	    	sum_p += 1.0
    	total += 1.0
    print sum_p/total
    output = model.predict(x_test)


    f = open("random_forest.csv", "wb")
    writer = csv.writer(f)
    writer.writerow(["PassengerId", "Survived"])
    for row, survived in zip(label['PassengerId'], output.astype(int)):
    	writer.writerow([row, survived])


if __name__ == '__main__':
    main()
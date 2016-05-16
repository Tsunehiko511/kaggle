#coding:utf-8
import csv as csv
import numpy as np

#Open up the csv file
csv_file_object = csv.reader(open('./data/train.csv', 'rb'))
header = csv_file_object.next() # 先頭のデータを抜く

data = []
for row in csv_file_object:
	data.append(row)
data = np.array(data)
print header
print data
print data.shape  # 行列の形

number_passengers = np.size(data[0::,1].astype(np.float)) # 人の総数
number_survived = np.sum(data[0::,1].astype(np.float)) 		# 生き残った人の数
proportion_survivors = number_survived/number_passengers 	# 生存率

women_only_stats = data[0::,4] == "female" # データが女性かの真偽値
men_only_stats = data[0::,4] != "female" 	 # データが男性かの真偽値

# 女性と男性の生存率を比較する(?try)
# 生存を0/1で表現
women_onboard = data[women_only_stats,1].astype(np.float) # 女性のみのの配列
men_onboard = data[men_only_stats,1].astype(np.float)

proportion_women_survived = np.sum(women_onboard)/np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard)/np.size(men_onboard)

print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived
# train.csvでは女性の方が生存率が高いことがわかった

# クラス別の生存率
# クラス1の生存率 = クラス1の生存者数/クラス1の総数
pclass1_only_stats = data[0::,2] == "1"
pclass2_only_stats = data[0::,2] == "2"
pclass3_only_stats = data[0::,2] == "3"
# print pclass1_only_stats, pclass2_only_stats, pclass2_only_stats
pclass1_onboard = data[pclass1_only_stats,1].astype(np.float)
pclass2_onboard = data[pclass2_only_stats,1].astype(np.float)
pclass3_onboard = data[pclass3_only_stats,1].astype(np.float)
proportion_pclass1_survived = np.sum(pclass1_onboard)/np.size(pclass1_onboard)
proportion_pclass2_survived = np.sum(pclass2_onboard)/np.size(pclass2_onboard)
proportion_pclass3_survived = np.sum(pclass3_onboard)/np.size(pclass3_onboard)

print proportion_pclass1_survived,
print proportion_pclass2_survived,
print proportion_pclass3_survived

# testファイルの取得
test_file = open('./data/test.csv', 'rb') 	# testファイルを開く
test_file_object = csv.reader(test_file) 		# 取得する
header = test_file.next() 									# ヘッダーを取得する
# 提出用ファイル(predict=予測)を作成
prediction_file = open("./data/genderbasedmodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

prediction_file_object.writerow(["PassengerId", "Survived"]) 	# ヘッダに"PassengerId", "Survived"を書く
for row in test_file_object:       # For each row in test.csv
    if row[3] == 'female':         # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])    # predict 1
    else:                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])    # predict 0
test_file.close() 				# testファイルを閉じる
prediction_file.close() 	# 予測ファイルを閉じる







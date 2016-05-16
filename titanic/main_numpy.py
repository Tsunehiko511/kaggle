# coding:utf-8
import csv as csv
import numpy as np


csv_file_object = csv.reader(open('./data/train.csv', 'rb'))
header = csv_file_object.next()
data = []

for row in csv_file_object:
	data.append(row)
data = np.array(data)

print header
print data
print data.shape

# Ageの列データを15個読む
print "Ageの列データを15個読む"
print header[5]
print data[0:15, 5]

print "stringをfloatに変換する"
print data[0:5, 5].astype(np.float)

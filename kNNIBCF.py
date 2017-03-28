#! python3
# -*- coding: utf-8 -*-
import datetime
import math
from math import sqrt
from threading import Lock
from threading import Thread

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from DataHelper import *


class CollaborativeFilter(object):
    def __init__(self, MyDataFrame=None, test_size=0.25):
        self.lock = Lock()
        self.SimilityMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data, self.test_data = train_test_split(MyDataFrame, test_size=test_size)
        self.train_data_matrix = None
        self.testDataFrame = None
        self.RMSE = dict()
        self.MAE = dict()

    ### 平均加权策略，预测userId对itemId的评分
    def getRating(self, Train_data_matrix, userId, itemId, simility_matrix, knumber=20):
        SIM = simility_matrix.copy()
        zeroset = numpy.where(Train_data_matrix == 0)
        SIM[zeroset] = 0
        test3 = sparse_argsort(-SIM)[0:knumber]
        simSums = numpy.sum(simility_matrix[test3])  # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        averageOfUser = MyCF.ItemMeanMatrix[itemId - 1]  # 获取userId 的平均值
        jiaquanAverage = (Train_data_matrix[test3] - MyCF.ItemMeanMatrix[test3]).dot(
            simility_matrix[test3])  # 计算每个用户的加权，预测
        if simSums == 0:
            if math.isnan(averageOfUser):
                return 0
            else:
                return averageOfUser
        else:
            return averageOfUser + jiaquanAverage / simSums


    def doEvaluate(self, testDataFrame, K):
        print(testDataFrame.head())
        for row in testDataFrame.itertuples():
            prerating = self.getRating(self.train_data_matrix[row[1]-1], row[1], row[2], self.SimilityMatrix[row[2]-1], K)  # 基于训练集预测用户评分(用户数目<=K)
            self.lock.acquire()
            self.truerating.append(row[3])
            self.predictions.append(prerating)
            self.lock.release()
            #print(len(self.predictions))


if __name__ == '__main__':

    startTime = datetime.datetime.now()
    #MyData = LoadMovieLens100k('Datas/ml-100k/u.data')
    #MyData = LoadMovieLens1M()
    MyData = LoadMovieLens10M()
    MyCF = CollaborativeFilter(MyData, test_size=0.1)
    print(type(MyCF.train_data))
    print(MyData.head())
    n_users = MyData.user_id.max()
    n_items = MyData.item_id.max()
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    # Create two user-item matrices, one for training and another for testing
    MyCF.train_data_matrix = numpy.zeros((n_users, n_items))
    for line in MyCF.train_data.itertuples():
        MyCF.train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    MyCF.SimilityMatrix = cosine_similarity(MyCF.train_data_matrix.T)  # ItemSimility
    MyCF.ItemMeanMatrix = numpy.true_divide(MyCF.train_data_matrix.sum(0),(MyCF.train_data_matrix != 0).sum(0))  # 按X轴方向获取非0元素均值，如果某行所有元素为0返回nan
    KList = [25, 50, 75, 100, 125, 150]

    for i in range(len(KList)):
        MyCF.predictions.clear()
        MyCF.truerating.clear()
        medTime = datetime.datetime.now()
        part1testData, part2testData = train_test_split(MyCF.test_data, test_size=0.5)
        print((medTime - startTime).seconds)
        t1 = Thread(target=MyCF.doEvaluate, args=(part1testData, KList[i]))
        t2 = Thread(target=MyCF.doEvaluate, args=(part2testData, KList[i]))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # MyCF.doEvaluate(MyCF.test_data, K)
        endTime = datetime.datetime.now()
        print((endTime - startTime).seconds)
        MyCF.RMSE[KList[i]] = sqrt(mean_squared_error(MyCF.truerating, MyCF.predictions))
        MyCF.MAE[KList[i]] = mean_absolute_error(MyCF.truerating, MyCF.predictions)
        print("K=%d,RMSE:%f,MAE:%f" % (KList[i], MyCF.RMSE[KList[i]], MyCF.MAE[KList[i]]))
    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of IBCF in MovieLens 10M')
    plt.xlabel('K')
    plt.ylabel('value')
    plt.legend()
    plt.grid()
    plt.savefig('IBCF ml-10M.png')
    plt.show()

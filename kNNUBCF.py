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
        self.UserMeanMatrix = None

    ### 平均加权策略，预测userId对itemId的评分
    def getRating(self, Train_data_matrix, userId, itemId, simility_matrix, knumber=20):
        jiaquanAverage = 0.0
        # 获取K近邻用户(评过分的用户集)
        userset = Train_data_matrix[:, itemId - 1].nonzero()
        # zeroset = numpy.where(Train_data_matrix[:, itemId - 1] == 0)
        averageOfUser = MyCF.UserMeanMatrix[userId - 1]  # 获取userId 的平均值
        # SIM = simility_matrix.copy()
        # SIM[userId - 1][zeroset] = numpy.nan
        # test2 = numpy.argsort(-SIM[userId - 1])[0:knumber]
        Neighborusers = self.get_K_Neighbors(userId, userset, simility_matrix, knumber)
        # simSums = numpy.sum(SIM[userId - 1][test2])
        simSums = 0.0
        # 计算每个用户的加权，预测
        for other in Neighborusers:
            sim = Neighborusers[other]
            averageOther = MyCF.UserMeanMatrix[other - 1]  # 该用户的平均分
            # 累加
            simSums += abs(sim)  # 取绝对值
            jiaquanAverage += (Train_data_matrix[other - 1][itemId - 1] - averageOther) * sim  # 累加，一些值为负
        # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        if simSums == 0:
            if math.isnan(averageOfUser):
                return 0
            else:
                return averageOfUser
        else:
            return averageOfUser + jiaquanAverage / simSums

    # 给定用户实例编号，和相似度矩阵，得到最相似的K个用户
    def get_K_Neighbors(self, userinstance, neighborlist, SimNArray, k=10):
        rank = dict()
        for i in neighborlist[0]:
            rank.setdefault(i + 1, 0)  # 设置初始值，以便做下面的累加运算
            rank[i + 1] += SimNArray[userinstance - 1][i]
        # test=
        myresult = dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[
                        0:k])  # 用sorted方法对推荐的物品进行排序，预计评分高的排在前面，再取其中nitem个，nitem为每个用户推荐的物品数量
        return myresult

    def doEvaluate(self, testDataFrame, K):
        print(testDataFrame.head())
        for row in testDataFrame.itertuples():
            prerating = self.getRating(self.train_data_matrix, row[1], row[2], self.SimilityMatrix,
                                       K)  # 基于训练集预测用户评分(用户数目<=K)
            self.lock.acquire()
            self.truerating.append(row[3])
            self.predictions.append(prerating)
            self.lock.release()
            print(len(self.predictions))


if __name__ == '__main__':
    startTime = datetime.datetime.now()
    MyData = LoadMovieLens1M()
    # MyData = LoadMovieLens100k('Datas/ml-100k/u.data')
    # MyData = LoadMovieLens10M()
    MyCF = CollaborativeFilter(MyData, test_size=0.2)
    print(type(MyCF.train_data))
    print(MyData.head())
    n_users = MyData.user_id.max()
    n_items = MyData.item_id.max()
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    # Create two user-item matrices, one for training and another for testing
    MyCF.train_data_matrix = numpy.zeros((n_users, n_items))
    for line in MyCF.train_data.itertuples():
        MyCF.train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    MyCF.SimilityMatrix = cosine_similarity(MyCF.train_data_matrix)
    MyCF.UserMeanMatrix = numpy.true_divide(MyCF.train_data_matrix.sum(1),
                                            (MyCF.train_data_matrix != 0).sum(1))  # 按X轴方向获取非0元素均值，如果某行所有元素为0返回nan
    KList = [25, 50, 75, 100, 125, 150]
    for i in range(len(KList)):
        MyCF.predictions.clear()
        MyCF.truerating.clear()
        part1testData, part2testData = train_test_split(MyCF.test_data, test_size=0.5)
        '''
        TestData1, TestData2 = train_test_split(part1testData, test_size=0.5)
        TestData3, TestData4 = train_test_split(part2testData, test_size=0.5)
        testData1, testData2 = train_test_split(TestData1, test_size=0.5)
        testData3, testData4 = train_test_split(TestData2, test_size=0.5)
        testData5, testData6 = train_test_split(TestData3, test_size=0.5)
        testData7, testData8 = train_test_split(TestData4, test_size=0.5)
        '''

        medTime = datetime.datetime.now()
        print((medTime - startTime).seconds)
        t1 = Thread(target=MyCF.doEvaluate, args=(part1testData, KList[i]))
        t2 = Thread(target=MyCF.doEvaluate, args=(part2testData, KList[i]))
        '''
        t3 = Thread(target=MyCF.doEvaluate, args=(testData3, K))
        t4 = Thread(target=MyCF.doEvaluate, args=(testData4, K))
        t5 = Thread(target=MyCF.doEvaluate, args=(testData5, K))
        t6 = Thread(target=MyCF.doEvaluate, args=(testData6, K))
        t7 = Thread(target=MyCF.doEvaluate, args=(testData7, K))
        t8 = Thread(target=MyCF.doEvaluate, args=(testData8, K))
        '''

        t1.start()
        t2.start()
        '''
        t3.start()
        t4.start()
        t5.start()
        t6.start()
        t7.start()
        t8.start()
        '''

        t1.join()
        t2.join()
        '''
        t3.join()
        t4.join()
        t5.join()
        t6.join()
        t7.join()
        t8.join()
        '''

        endTime = datetime.datetime.now()
        print((endTime - startTime).seconds)
        MyCF.RMSE[KList[i]] = sqrt(mean_squared_error(MyCF.truerating, MyCF.predictions))
        MyCF.MAE[KList[i]] = mean_absolute_error(MyCF.truerating, MyCF.predictions)
        print("K=%d,RMSE:%f,MAE:%f" % (KList[i], MyCF.RMSE[KList[i]], MyCF.MAE[KList[i]]))

    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of UBCF in MovieLens 1M')
    plt.xlabel('K')
    plt.ylabel('value')
    plt.legend()
    plt.grid()
    plt.savefig('UBCF ml-1M.png')
    plt.show()

#! python3
# -*- coding: utf-8 -*-
from threading import Lock
from sklearn.metrics.pairwise import cosine_similarity
from threading import Thread
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import time, datetime, math

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
        self.RMSE = 0
        self.MAE = 0

    ### 平均加权策略，预测userId对itemId的评分
    def getRating(self, Train_data_matrix, userId, itemId, simility_matrix, knumber=20):
        jiaquanAverage = 0.0
        simSums = 0.0
        # 获取K近邻item(评过分的item集)
        itemset = Train_data_matrix[userId - 1].nonzero()
        averageOfItem = Train_data_matrix[:, (itemId - 1)][
            numpy.nonzero(Train_data_matrix[:, (itemId - 1)])].mean()  # 获取itemId 的平均值
        test = simility_matrix[:, userId - 1][itemset]
        test1 = numpy.argsort(test)[0:knumber]
        Neighborusers = self.get_K_Neighbors(itemId, itemset, simility_matrix, knumber)
        # 计算每个用户的加权，预测
        for other in Neighborusers:
            sim = Neighborusers[other]
            averageOther = Train_data_matrix[:, (other - 1)][
                numpy.nonzero(Train_data_matrix[:, (other - 1)])].mean()  # 该用户的平均分
            # 累加
            simSums += abs(sim)  # 取绝对值
            jiaquanAverage += (Train_data_matrix[userId - 1][other - 1] - averageOther) * sim  # 累加，一些值为负
        # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        if simSums == 0:
            if math.isnan(averageOfItem):
                return 0
            else:
                return averageOfItem
        else:
            return averageOfItem + jiaquanAverage / simSums

    # 给定用户实例编号，和相似度矩阵，得到最相似的K个item
    def get_K_Neighbors(self, iteminstance, neighborlist, SimNArray, k=10):
        rank = dict()
        for i in neighborlist[0]:
            rank.setdefault(i + 1, 0)  # 设置初始值，以便做下面的累加运算
            rank[i + 1] += SimNArray[iteminstance - 1][i]
        # test=
        myresult = dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[
                        0:k])  # 用sorted方法对推荐的物品进行排序，预计评分高的排在前面，再取其中nitem个，nitem为每个用户推荐的物品数量
        return myresult

    def doEvaluate(self, testDataFrame, K):
        print(testDataFrame.head())
        _truerating = []
        _predictions = []
        for row in testDataFrame.itertuples():
            prerating = self.getRating(self.train_data_matrix, row[1], row[2], self.SimilityMatrix,
                                       K)  # 基于训练集预测用户评分(用户数目<=K)
            _truerating.append(row[3])
            _predictions.append(prerating)
        self.lock.acquire()
        self.truerating.extend(_truerating)
        self.predictions.extend(_predictions)
        self.lock.release()
        print(len(self.predictions))


if __name__ == '__main__':
    # MyData = LoadMovieLens100k('Datas/ml-100k/u.data')
    startTime = datetime.datetime.now()
    MyData = LoadMovieLens100k('Datas/ml-100k/u.data')
    MyCF = CollaborativeFilter(MyData, test_size=0.2)
    print(type(MyCF.train_data))
    print(MyData.head())
    n_users = MyData.user_id.unique().shape[0]
    n_items = MyData.item_id.max()
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    # Create two user-item matrices, one for training and another for testing
    MyCF.train_data_matrix = numpy.zeros((n_users, n_items))
    for line in MyCF.train_data.itertuples():
        MyCF.train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    MyCF.test_data_matrix = numpy.zeros((n_users, n_items))
    for line in MyCF.test_data.itertuples():
        MyCF.test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

    MyCF.SimilityMatrix = cosine_similarity(MyCF.train_data_matrix.T)  # ItemSimility
    for K in [25]:  # , 50, 75, 100, 125, 150]:
        MyCF.predictions.clear()
        MyCF.truerating.clear()
        medTime = datetime.datetime.now()
        part1testData, part2testData = train_test_split(MyCF.test_data, test_size=0.5)
        print((medTime - startTime).seconds)
        '''
        t1 = Thread(target=MyCF.doEvaluate, args=(part1testData, K))
        t2 = Thread(target=MyCF.doEvaluate, args=(part2testData, K))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        '''
        MyCF.doEvaluate(MyCF.test_data, K)
        endTime = datetime.datetime.now()
        print((endTime - startTime).seconds)
        MyCF.RMSE = sqrt(mean_squared_error(MyCF.truerating, MyCF.predictions))
        MyCF.MAE = mean_absolute_error(MyCF.truerating, MyCF.predictions)
        print("RMSE:%f,MAE:%f" % (MyCF.RMSE, MyCF.MAE))

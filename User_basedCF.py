#! python3
# -*- coding: utf-8 -*-
from threading import Lock
from DataHelper import *
from EvaluationHelper import *
import heapq
import matplotlib.pyplot as plt


class UBCollaborativeFilter(object):
    def __init__(self):
        self.lock = Lock()
        self.SimilityMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.test_data_matrix = None
        self.testDataFrame = None
        self.RMSE = dict()
        self.MAE = dict()
        self.Recall = dict()
        self.Precision = dict()
        self.UserMeanMatrix = None

        self.hit = 0
        self.recall = 0
        self.precision = 0

    ### 平均加权策略，预测userId对itemId的评分
    def getRating(self, Train_data_matrix, userId, simility_matrix,neighborset):
        simSums = numpy.sum(simility_matrix[neighborset])  # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        averageOfUser = self.UserMeanMatrix[userId]  # 获取userId 的平均值
        jiaquanAverage = (Train_data_matrix[neighborset] - self.UserMeanMatrix[neighborset]).dot(
            simility_matrix[neighborset])  # 计算每个用户的加权，预测
        if simSums == 0:
            return averageOfUser
        else:
            return averageOfUser + jiaquanAverage / simSums

    def doEvaluate(self, testDataMatrix, K):
        a, b = testDataMatrix.nonzero()
        for userIndex, itemIndex in zip(a, b):
            # for row in testDataFrame.itertuples():
            neighborset = get_K_Neighbors(self.train_data_matrix[:, 0], self.SimilityMatrix[0],
                                          K)  # 用户最相似的K个用户
            recommendsettest = self.Recommender(userIndex, neighborset, 20)
            prerating = self.getRating(self.train_data_matrix[:, itemIndex], userIndex, self.SimilityMatrix[userIndex],
                                       neighborset)  # 基于训练集预测用户评分(用户数目<=K)
            self.lock.acquire()
            self.truerating.append(testDataMatrix[userIndex][itemIndex])
            self.predictions.append(prerating)
            self.lock.release()
            print(len(self.predictions))
        self.RMSE[K] = RMSE(self.truerating, self.predictions)
        self.MAE[K] = MAE(self.truerating, self.predictions)
        self.Recall[K] = self.hit / (self.recall * 1.0)
        self.Precision[K] = self.hit / (self.precision * 1.0)
        print("UBCF  K=%d,RMSE:%f,MAE:%f,RECALL:%f,PRECISION:%f" % (
            K, self.RMSE[K], self.MAE[K], self.Recall[K], self.Precision[K]))
        Savetxt('Datas/User-basedCF.txt', "UBCF  K=%d\tRMSE:%f\tMAE:%f\tRECALL:%f\tPRECISION:%f" % (
            K, self.RMSE[K], self.MAE[K], self.Recall[K], self.Precision[K]))

    def RecallAndPrecision(self, neighborset, Test_data_matrix):
        test = Test_data_matrix.nonzero()
        hit = numpy.intersect1d(neighborset, test)
        self.hit += len(hit)
        self.recall += len(test[0])
        self.precision += len(neighborset)

    def Clear(self):
        self.truerating = []
        self.predictions = []
        self.hit = 0
        self.recall = 0
        self.precision = 0

    def Recommender(self, userIndex, neighborset, N):
        teat1 = get_N_Recommends(neighborset, userIndex, self.train_data_matrix, self.SimilityMatrix, N)
        self.RecallAndPrecision(teat1, self.test_data_matrix[userIndex])
        return teat1

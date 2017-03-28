#! python3
# -*- coding: utf-8 -*-
from threading import Lock

from DataHelper import *
from EvaluationHelper import *
import matplotlib.pyplot as plt


class UBCollaborativeFilter(object):
    def __init__(self):
        self.lock = Lock()
        self.SimilityMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.testDataFrame = None
        self.RMSE = dict()
        self.MAE = dict()
        self.Recall = dict()
        self.Precision = dict()
        self.UserMeanMatrix = None

    ### 平均加权策略，预测userId对itemId的评分
    def getRating(self, Train_data_matrix, userId, simility_matrix, knumber=20):
        SIM = simility_matrix.copy()
        zeroset = numpy.where(Train_data_matrix == 0)
        SIM[zeroset] = 0
        test3 = sparse_argsort(-SIM)[0:knumber]
        simSums = numpy.sum(simility_matrix[test3])  # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        averageOfUser = self.UserMeanMatrix[userId - 1]  # 获取userId 的平均值
        jiaquanAverage = (Train_data_matrix[test3] - self.UserMeanMatrix[test3]).dot(
            simility_matrix[test3])  # 计算每个用户的加权，预测
        if simSums == 0:
            return averageOfUser
        else:
            return averageOfUser + jiaquanAverage / simSums

    def doEvaluate(self, testDataFrame, K):
        print(testDataFrame.head())
        for row in testDataFrame.itertuples():
            prerating = self.getRating(self.train_data_matrix[:, row[2] - 1], row[1], self.SimilityMatrix[row[1] - 1],
                                       K)  # 基于训练集预测用户评分(用户数目<=K)
            self.lock.acquire()
            self.truerating.append(row[3])
            self.predictions.append(prerating)
            self.lock.release()
            # print(len(self.predictions))
        self.RMSE[K] = RMSE(self.truerating, self.predictions)
        self.MAE[K] = MAE(self.truerating, self.predictions)
        self.Recall[K] = RECALL(self.truerating, self.predictions)
        self.Precision[K] = PRECISION(self.truerating, self.predictions)
        print("UBCF  K=%d,RMSE:%f,MAE:%f,RECALL:%f,PRECISION:%f" % (
            K, self.RMSE[K], self.MAE[K], self.Recall[K], self.Precision[K]))
        Savetxt('Datas/User-basedCF.txt', "UBCF  K=%d\tRMSE:%f\tMAE:%f\tRECALL:%f\tPRECISION:%f" % (
            K, self.RMSE[K], self.MAE[K], self.Recall[K], self.Precision[K]))


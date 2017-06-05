#! python3
# -*- coding: utf-8 -*-
from DataHelper import *
from EvaluationHelper import *


class IBCollaborativeFilter(object):
    def __init__(self):
        self.SimilityMatrix = None
        self.ItemMeanMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.RMSE = dict()
        self.MAE = dict()

    ### 平均加权策略，预测userId对itemId的评分
    def getRating(self, Train_data_matrix, itemId, simility_matrix, knumber=20):
        neighborset = get_K_Neighbors(Train_data_matrix, simility_matrix, knumber)  # 最相似的K个Item
        simSums = numpy.sum(simility_matrix[neighborset])  # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        averageOfUser = self.ItemMeanMatrix[itemId]  # 获取userId 的平均值
        jiaquanAverage = (Train_data_matrix[neighborset] - self.ItemMeanMatrix[neighborset]).dot(simility_matrix[neighborset])  # 计算每个用户的加权，预测
        if simSums == 0:
            return averageOfUser
        else:
            return averageOfUser + jiaquanAverage / simSums

    def doEvaluate(self, testDataMatrix, K):
        a, b = testDataMatrix.nonzero()
        for userIndex, itemIndex in zip(a, b):
            prerating = self.getRating(self.train_data_matrix[userIndex], itemIndex, self.SimilityMatrix[itemIndex],K)  # 基于训练集预测用户评分(用户数目<=K)
            self.truerating.append(testDataMatrix[userIndex][itemIndex])
            self.predictions.append(prerating)
            # print(len(self.predictions))
        self.RMSE[K] = RMSE(self.truerating, self.predictions)
        self.MAE[K] = MAE(self.truerating, self.predictions)
        print("IBCF  K=%d,RMSE:%f,MAE:%f" % (K, self.RMSE[K], self.MAE[K]))



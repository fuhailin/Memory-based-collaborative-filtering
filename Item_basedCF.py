#! python3
# -*- coding: utf-8 -*-
from threading import Lock
import math
from DataHelper import *
from EvaluationHelper import *
import matplotlib.pyplot as plt


class IBCollaborativeFilter(object):
    def __init__(self):
        self.lock = Lock()
        self.SimilityMatrix = None
        self.ItemMeanMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.test_data_matrix=None
        self.testDataFrame = None
        self.RMSE = dict()
        self.MAE = dict()
        self.Recall = dict()
        self.Precision = dict()

        self.hit = 0
        self.recall = 0
        self.precision = 0

    ### 平均加权策略，预测userId对itemId的评分
    def getRating(self, Train_data_matrix, itemId, simility_matrix, knumber=20):
        SIM = simility_matrix.copy()
        zeroset = numpy.where(Train_data_matrix == 0)
        SIM[zeroset] = 0
        recommendset = sparse_argsort(-SIM)[0:knumber]
        simSums = numpy.sum(simility_matrix[recommendset])  # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        averageOfUser = self.ItemMeanMatrix[itemId]  # 获取userId 的平均值
        jiaquanAverage = (Train_data_matrix[recommendset] - self.ItemMeanMatrix[recommendset]).dot(
            simility_matrix[recommendset])  # 计算每个用户的加权，预测
        self.RecallAndPrecision(recommendset, self.test_data_matrix[:,itemId])
        if simSums == 0:
            return averageOfUser
        else:
            return averageOfUser + jiaquanAverage / simSums

    def doEvaluate(self, testDataMatrix, K):
        a, b = testDataMatrix.nonzero()
        for q, w in zip(a, b):
            prerating = self.getRating(self.train_data_matrix[q], w, self.SimilityMatrix[w],
                                       K)  # 基于训练集预测用户评分(用户数目<=K)
            self.lock.acquire()
            self.truerating.append(testDataMatrix[q][w])
            self.predictions.append(prerating)
            self.lock.release()
            #print(len(self.predictions))
        self.RMSE[K] = RMSE(self.truerating, self.predictions)
        self.MAE[K] = MAE(self.truerating, self.predictions)
        self.Recall[K] = self.hit / (self.recall * 1.0)
        self.Precision[K] = self.hit / (self.precision * 1.0)
        print("IBCF  K=%d,RMSE:%f,MAE:%f,RECALL:%f,PRECISION:%f" % (
            K, self.RMSE[K], self.MAE[K], self.Recall[K], self.Precision[K]))
        Savetxt('Datas/Item-basedCF.txt',"IBCF  K=%d\tRMSE:%f\tMAE:%f\tRECALL:%f\tPRECISION:%f" % (
            K, self.RMSE[K], self.MAE[K], self.Recall[K], self.Precision[K]))

    def RecallAndPrecision(self, recommendset, Test_data_matrix):
        test = Test_data_matrix.nonzero()
        hit = numpy.intersect1d(recommendset, test)
        self.hit += len(hit)
        self.recall += len(test[0])
        self.precision += len(recommendset)

    def Clear(self):
        self.truerating = []
        self.predictions = []
        self.hit = 0
        self.recall = 0
        self.precision = 0
#! python3
# -*- coding: utf-8 -*-
from DataHelper import *
from EvaluationHelper import *


class UBCollaborativeFilter(object):
    def __init__(self):
        self.SimilityMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.test_data_matrix = None
        self.testDataFrame = None
        self.Recall = dict()
        self.Precision = dict()
        self.UserMeanMatrix = None

        self.hit = 0
        self.recall = 0
        self.precision = 0

    def doEvaluate(self, testDataMatrix, K):
        a, b = testDataMatrix.nonzero()
        for userIndex, itemIndex in zip(a, b):
            # for row in testDataFrame.itertuples():
            neighborset = get_K_Neighbors(self.train_data_matrix[:, 0], self.SimilityMatrix[0],K)  # 用户最相似的K个用户
            recommendsettest = self.Recommender(userIndex, neighborset, 20)
            print(len(self.predictions))
        self.Recall[K] = self.hit / (self.recall * 1.0)
        self.Precision[K] = self.hit / (self.precision * 1.0)
        print("UBCF  K=%d,RECALL:%f,PRECISION:%f" % (K, self.Recall[K], self.Precision[K]))

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

#! python3
# -*- coding: utf-8 -*-
from threading import Lock
import math
from DataHelper import *
from EvaluationHelper import *
import matplotlib.pyplot as plt


class IBCollaborativeFilter(object):
    def __init__(self):
        self.SimilityMatrix = None
        self.ItemMeanMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.test_data_matrix = None
        self.testDataFrame = None
        self.Recall = dict()
        self.Precision = dict()

        self.hit = 0
        self.recall = 0
        self.precision = 0


    def doEvaluate(self, testDataMatrix, K):
        a, b = testDataMatrix.nonzero()
        for q, w in zip(a, b):


        self.Recall[K] = self.hit / (self.recall * 1.0)
        self.Precision[K] = self.hit / (self.precision * 1.0)
        print("IBCF  K=%d,RECALL:%f,PRECISION:%f" % (K,  self.Recall[K], self.Precision[K]))

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

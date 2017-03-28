# -*- coding: utf-8 -*-
import numpy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def RMSE(true, prediction):
    rmse = numpy.sqrt(mean_squared_error(true, prediction))
    return rmse


def RECALL(true, prediction):
    recall = recall_score(true, prediction, average='macro')
    return recall


def PRECISION(true, prediction):
    prediction = precision_score(true, prediction, average='macro')
    return prediction


def MAE(true, prediction):
    mae = mean_squared_error(true, prediction)
    return mae


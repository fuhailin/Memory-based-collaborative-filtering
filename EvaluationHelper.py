# -*- coding: utf-8 -*-
import numpy
from sklearn.metrics import mean_squared_error


def RMSE(true, prediction):
    rmse = numpy.sqrt(mean_squared_error(true, prediction))
    return rmse


def MAE(true, prediction):
    mae = mean_squared_error(true, prediction)
    return mae

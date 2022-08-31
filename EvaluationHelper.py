import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def RMSE(true, prediction):
    rmse = np.sqrt(mean_squared_error(true, prediction))
    return rmse


def MAE(true, prediction):
    mae = mean_absolute_error(true, prediction)
    return mae

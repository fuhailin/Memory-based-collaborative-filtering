# -*- coding: utf-8 -*-
from sklearn.metrics import mean_squared_error
import numpy, math


def RMSE(prediction, testData):
    prediction = prediction[testData.nonzero()].flatten()
    testData = testData[testData.nonzero()].flatten()
    rmse = numpy.sqrt(mean_squared_error(prediction, testData))
    return rmse


def recallAndPrecision(self, train=None, test=None, k=8, nitem=10):
    hit = 0
    recall = 0
    precision = 0
    for user in train.keys():
        tu = test.get(user, {})  # 如果测试集中没有这个用户，则将tu初始化为空，避免test[user]报错
        rank = self.recommend(user, train=train, k=k, nitem=nitem)
        for item, _ in rank.items():
            if item in tu:
                hit += 1
        recall += len(tu)
        precision += nitem
    return hit / (recall * 1.0), hit / (precision * 1.0)


def coverage(self, train=None, test=None, k=8, nitem=10):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
        rank = self.recommend(user, train, k=k, nitem=nitem)
        for item, _ in rank.items():
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)


def popularity(self, train=None, test=None, k=8, nitem=10):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            item_popularity.setdefault(item, 0)
            item_popularity[item] += 1
    ret = 0
    n = 0
    for user in train.keys():
        rank = self.recommend(user, train, k=k, nitem=nitem)
        for item, _ in rank.items():
            ret += math.log(1 + item_popularity[item])
            n += 1
    return ret / (n * 1.0)

#! python3
# -*- coding: utf-8 -*-
import datetime
from numpy import *
from threading import Thread
from ThreadWithReturn import *
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from Item_basedCF import *
from User_basedCF import *
if __name__ == '__main__':
    startTime = datetime.datetime.now()
    # MyData = LoadMovieLens1M()
    MyData = LoadMovieLens100k('Datas/ml-100k/u.data')
    # MyData = LoadMovieLens10M()
    MyUBCF = UBCollaborativeFilter()
    train_data, test_data = train_test_split(MyData, test_size=0.1)
    print(type(train_data))
    print(MyData.head())
    n_users = MyData.user_id.max()
    n_items = MyData.item_id.max()
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    test1 = ThreadWithReturnValue(target=DataFrame2Matrix, args=(n_users, n_items, train_data))
    test2 = ThreadWithReturnValue(target=DataFrame2Matrix, args=(n_users, n_items, test_data))
    test1.start()
    test2.start()
    train_data_matrix = test1.join()
    test_data_matrix = test2.join()
    MyUBCF.train_data_matrix = train_data_matrix
    MyUBCF.test_data_matrix = test_data_matrix

    MyUBCF.SimilityMatrix = cosine_similarity(train_data_matrix)
    MyUBCF.UserMeanMatrix = numpy.true_divide(MyUBCF.train_data_matrix.sum(1),
                                              (MyUBCF.train_data_matrix != 0).sum(1))  # 按X轴方向获取非0元素均值，如果某行所有元素为0返回nan
    KList = [25]#, 50, 75, 100, 125, 150]
    for i in range(len(KList)):
        MyUBCF.Clear()

        medTime = datetime.datetime.now()
        print((medTime - startTime).seconds)
        t1 = Thread(target=MyUBCF.doEvaluate, args=(test_data_matrix, KList[i]))
        t1.start()
        t1.join()

        endTime = datetime.datetime.now()
        print((endTime - startTime).seconds)
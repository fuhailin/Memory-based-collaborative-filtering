#! python3
# -*- coding: utf-8 -*-
import datetime
from threading import Thread

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from Item_basedCF import *
from User_basedCF import *

MovieLensData = {
    1: 'Datas/ml-100k/u.data',
    2: 'Datas/ml-1M/ratings.dat',
    3: 'Datas/ml-10M100K/ratings.dat',
    4: 'Datas/ml-20m/ratings.csv'
}

if __name__ == '__main__':
    startTime = datetime.datetime.now()
    # MyData = LoadMovieLens1M()
    MyData = LoadMovieLens100k('Datas/ml-100k/u.data')
    # MyData = LoadMovieLens10M()
    MyUBCF = UBCollaborativeFilter()
    MyIBCF = IBCollaborativeFilter()
    train_data, test_data = train_test_split(MyData, test_size=0.1)
    print(type(train_data))
    print(MyData.head())
    n_users = MyData.user_id.max()
    n_items = MyData.item_id.max()
    print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))

    # Create two user-item matrices, one for training and another for testing
    train_data_matrix = numpy.zeros((n_users, n_items))
    for line in train_data.itertuples():
        train_data_matrix[line[1] - 1, line[2] - 1] = line[3]
    MyUBCF.train_data_matrix = train_data_matrix
    MyIBCF.train_data_matrix = train_data_matrix

    MyUBCF.SimilityMatrix = cosine_similarity(train_data_matrix)
    MyIBCF.SimilityMatrix = cosine_similarity(train_data_matrix.T)
    MyUBCF.UserMeanMatrix = numpy.true_divide(MyUBCF.train_data_matrix.sum(1),
                                              (MyUBCF.train_data_matrix != 0).sum(1))  # 按X轴方向获取非0元素均值，如果某行所有元素为0返回nan
    MyIBCF.ItemMeanMatrix = numpy.true_divide(MyUBCF.train_data_matrix.sum(0),
                                              (MyUBCF.train_data_matrix != 0).sum(0))  # 按X轴方向获取非0元素均值，如果某行所有元素为0返回nan
    KList = [25]#, 50, 75, 100, 125, 150]
    for i in range(len(KList)):
        MyUBCF.predictions.clear()
        MyUBCF.truerating.clear()
        MyIBCF.predictions.clear()
        MyIBCF.truerating.clear()

        medTime = datetime.datetime.now()
        print((medTime - startTime).seconds)
        t1 = Thread(target=MyUBCF.doEvaluate, args=(test_data, KList[i]))
        t2 = Thread(target=MyIBCF.doEvaluate, args=(test_data, KList[i]))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        endTime = datetime.datetime.now()
        print((endTime - startTime).seconds)
    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyUBCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyUBCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of UBCF in MovieLens 10M')
    plt.xlabel('K')
    plt.ylabel('value')
    plt.legend()
    plt.grid()
    plt.savefig('UBCF ml-10M.png')
    plt.show()
    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyIBCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyIBCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of IBCF in MovieLens 10M')
    plt.xlabel('K')
    plt.ylabel('value')
    plt.legend()
    plt.grid()
    plt.savefig('IBCF ml-10M.png')
    plt.show()


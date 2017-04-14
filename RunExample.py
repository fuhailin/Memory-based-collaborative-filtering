#! python3
# -*- coding: utf-8 -*-
import datetime

from numpy import *
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from Item_basedCF import *
from ThreadWithReturn import *
from User_basedCF import *

MovieLensData = {
    1: 'Datas/ml-100k/u.data',
    2: 'Datas/ml-1M/ratings.dat',
    3: 'Datas/ml-10M/ratings.dat',
    4: 'Datas/ml-20m/ratings.csv'
}

if __name__ == '__main__':
    startTime = datetime.datetime.now()
    filetype = 'ml-20m'
    MyData = LoadMovieLens(filetype)
    MyUBCF = UBCollaborativeFilter(filetype)
    MyIBCF = IBCollaborativeFilter(filetype)
    train_data, test_data = train_test_split(MyData, test_size=0.2)
    print(type(train_data))
    print(MyData.head())
    if filetype == 'ml-20m':
        n_users = MyData.userId.max()
        n_items = MyData.movieId.max()
    else:
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
    MyIBCF.train_data_matrix = train_data_matrix
    MyUBCF.test_data_matrix = test_data_matrix
    MyIBCF.test_data_matrix = test_data_matrix

    MyUBCF.SimilityMatrix = cosine_similarity(train_data_matrix)
    MyIBCF.SimilityMatrix = cosine_similarity(train_data_matrix.T)
    MyUBCF.UserMeanMatrix = numpy.true_divide(MyUBCF.train_data_matrix.sum(1),
                                              (MyUBCF.train_data_matrix != 0).sum(1))  # 按X轴方向获取非0元素均值，如果某行所有元素为0返回nan
    MyIBCF.ItemMeanMatrix = numpy.true_divide(MyUBCF.train_data_matrix.sum(0),
                                              (MyUBCF.train_data_matrix != 0).sum(0))  # 按X轴方向获取非0元素均值，如果某行所有元素为0返回nan
    MyIBCF.ItemMeanMatrix[isnan(MyIBCF.ItemMeanMatrix)] = 0
    KList = [25, 50, 75, 100, 125, 150]
    for i in range(len(KList)):
        MyUBCF.Clear()
        MyIBCF.Clear()

        medTime = datetime.datetime.now()
        print((medTime - startTime).seconds)
        t1 = Thread(target=MyUBCF.doEvaluate, args=(test_data_matrix, KList[i]))
        t2 = Thread(target=MyIBCF.doEvaluate, args=(test_data_matrix, KList[i]))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        endTime = datetime.datetime.now()
        print((endTime - startTime).seconds)

    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyUBCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyUBCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of UBCF in MovieLens ' + MyUBCF.FileType)
    plt.xlabel('K')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig('Datas/' + MyIBCF.FileType + '/UBCF ' + MyUBCF.FileType + '.png')
    plt.show()
    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyIBCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyIBCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of IBCF in MovieLens ' + MyIBCF.FileType)
    plt.xlabel('K')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.savefig('Datas/' + MyIBCF.FileType + '/IBCF ' + MyIBCF.FileType + '.png')
    plt.show()

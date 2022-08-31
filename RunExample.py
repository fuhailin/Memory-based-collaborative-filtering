#! python3
# -*- coding: utf-8 -*-
import argparse
import datetime

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from Item_basedCF import *
from ThreadWithReturn import *
from User_basedCF import *

MovieLensData = {
    1: 'Datas/ml-100k/u.data',
    2: 'Datas/ml-1M/ratings.dat',
    3: 'Datas/ml-10M100K/ratings.dat',
    4: 'Datas/ml-20m/ratings.csv'
}


def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ratings",
                        type=str,
                        default='ml-100k',
                        help="Ratings file")

    parser.add_argument("--testsize",
                        type=float,
                        default=0.2,
                        help="Percentage of test data")

    return parser.parse_args()


if __name__ == '__main__':
    myparser = parseargs()
    startTime = datetime.datetime.now()
    MyData = LoadMovieLensData(myparser.ratings)
    MyUBCF = UBCollaborativeFilter()
    MyIBCF = IBCollaborativeFilter()
    train_data, test_data = train_test_split(MyData, test_size=myparser.testsize)
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
    MyIBCF.train_data_matrix = train_data_matrix
    MyUBCF.test_data_matrix = test_data_matrix
    MyIBCF.test_data_matrix = test_data_matrix

    # 皮尔逊相关系数
    # MyUBCF.SimilityMatrix = np.corrcoef(train_data_matrix)
    # MyIBCF.SimilityMatrix = np.corrcoef(train_data_matrix.T)

    # # 余弦相似度
    MyUBCF.SimilityMatrix = cosine_similarity(train_data_matrix)
    # MyIBCF.SimilityMatrix = cosine_similarity(train_data_matrix.T)

    # 按X轴方向获取非0元素均值，如果某行所有元素为0返回nan，横着，求对应用户所有电影得平均分
    MyUBCF.UserMeanMatrix = np.true_divide(MyUBCF.train_data_matrix.sum(1), (MyUBCF.train_data_matrix != 0).sum(1))
    # 按Y轴方向获取非0元素均值，如果某行所有元素为0返回nan，竖着，求该物品得所有用户平均分
    # MyIBCF.ItemMeanMatrix = numpy.true_divide(MyUBCF.train_data_matrix.sum(0), (MyUBCF.train_data_matrix != 0).sum(0))
    # MyIBCF.ItemMeanMatrix[np.isnan(MyIBCF.ItemMeanMatrix)] = 0
    KList = [10, 20, 30, 40, 50, 60]
    for i in range(len(KList)):
        MyUBCF.truerating = []
        MyUBCF.predictions = []
        # MyIBCF.truerating = []
        # MyIBCF.predictions = []

        medTime = datetime.datetime.now()
        print((medTime - startTime).seconds)
        t1 = Thread(target=MyUBCF.doEvaluate, args=(test_data_matrix, KList[i]))
        # t2 = Thread(target=MyIBCF.doEvaluate, args=(test_data_matrix, KList[i]))
        t1.start()
        # t2.start()
        t1.join()
        # t2.join()

        endTime = datetime.datetime.now()
        print("Cost time:%d seconds" % (endTime - startTime).seconds)
        Savetxt("Docs/%s/UBCF %s %1.1f.txt" % (myparser.ratings, myparser.ratings, myparser.testsize),
                "UBCF  K=%d\tRMSE:%f\tMAE:%f\t" % (KList[i], MyUBCF.RMSE[KList[i]], MyUBCF.MAE[KList[i]]))
        # Savetxt('Docs/%s/IBCF %s %1.1f.txt' % (myparser.ratings, myparser.ratings, myparser.testsize),"IBCF  K=%d\tRMSE:%f\tMAE:%f\t" % (KList[i], MyIBCF.RMSE[KList[i]], MyIBCF.MAE[KList[i]]))
    # Check performance by plotting train and test errors
    plt.plot(KList, list(MyUBCF.RMSE.values()), marker='o', label='RMSE')
    plt.plot(KList, list(MyUBCF.MAE.values()), marker='v', label='MAE')
    plt.title('The Error of UBCF in MovieLens ' + myparser.ratings)
    plt.xlabel('K')
    plt.ylabel('value')
    plt.legend()
    plt.grid()
    plt.savefig('Docs/%s/UBCF %s %1.1f.png' % (myparser.ratings, myparser.ratings, myparser.testsize))
    plt.show()
    plt.gcf().clear()
    # Check performance by plotting train and test errors
    # plt.plot(KList, list(MyIBCF.RMSE.values()), marker='o', label='RMSE')
    # plt.plot(KList, list(MyIBCF.MAE.values()), marker='v', label='MAE')
    # plt.title('The Error of IBCF in MovieLens ' + myparser.ratings)
    # plt.xlabel('K')
    # plt.ylabel('value')
    # plt.legend()
    # plt.grid()
    # plt.savefig('Docs/%s/IBCF %s %1.1f.png' % (myparser.ratings, myparser.ratings, myparser.testsize))
    # plt.show()

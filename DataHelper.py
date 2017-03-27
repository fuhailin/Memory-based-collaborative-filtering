# -*- coding: utf-8 -*-
import numpy
import pandas
import pickle

import numpy as np
from sklearn.model_selection import train_test_split


def SaveData2pkl(DictData, FilePath='Datas/Mydata.pkl', mode='wb'):
    pkl_file = open(FilePath, mode)
    try:
        pickle.dump(DictData, pkl_file, protocol=2)
        return True
    except:
        return False
    finally:
        pkl_file.close()


def SaveData2cvs(MatrixData, FilePath='Datas/Mydata.pkl', Thisdelimiter=','):
    try:
        numpy.savetxt(FilePath, MatrixData, delimiter=Thisdelimiter)
        return True
    except Exception as e:
        print(repr(e))
        return False


def LoadData4pkl(FilePath='Datas/Mydata.pkl', mode='rb'):
    pkl_file = open(FilePath, mode)
    try:
        DataDict = pickle.load(pkl_file)
        return DataDict
    except:
        return None
    finally:
        pkl_file.close()


def LoadData4cvs(FilePath='Datas/Mydata.pkl', Thisdelimiter=',', mode='rb'):
    try:
        my_matrix = numpy.loadtxt(open(FilePath, mode), delimiter=Thisdelimiter, skiprows=0)
        return my_matrix
    except:
        return None


def LoadDoubanData(FilePath='Datas/Mydata.pkl'):
    LineNum = 1
    UserRating = dict()
    UserIndex = LoadData4pkl('Datas/UserIndex.pkl')
    ItemIndex = LoadData4pkl('Datas/ItemIndex.pkl')
    for line in open(FilePath, 'r', encoding='UTF-8'):
        LineNum += 1
        if len(line.rstrip('\n')) == 0:
            continue
        linelist = line.split(',')
        UserID = int(linelist[0])
        MovieID = int(linelist[1])
        Rating = float(linelist[2])
        tags = str(linelist[4].rstrip('\n')).lower()
        UserRating.setdefault(UserIndex[UserID], {})
        UserRating[UserIndex[UserID]][ItemIndex[MovieID]] = Rating
        print("第%d行数据：" % LineNum)
        # if z>30000: break
    return UserRating


def LoadMovieLens100k(FilePath='Datas/ml-100k/u.data'):
    """
    :param FilePath:
    :return: DataFrame
    """
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pandas.read_table(FilePath, header=None, names=header)
    return data


def LoadMovieLens1M(FilePath='Datas/ml-1M/ratings.dat'):
    """
    :param FilePath:
    :return: DataFrame
    """
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pandas.read_table(FilePath, sep="::", header=None, names=header, engine='python')
    return data


def LoadMovieLens10M(FilePath='Datas/ml-10M100K/ratings.dat'):
    """
    :param FilePath:
    :return: DataFrame
    """
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pandas.read_table(FilePath, sep="::", header=None, names=header, engine='python')
    return data


def SpiltData(DataSet, SpiltRate=0.25):
    TrainData, TestData = train_test_split(DataSet, test_size=SpiltRate)
    return TrainData, TestData


### 平均加权策略，预测userId对itemId的评分
def getRating(Train_data_matrix, userId, itemId, simility_matrix, knumber=20):
    jiaquanAverage = 0.0
    simSums = 0.0
    # 获取K近邻用户(评过分的用户集)
    userset = Train_data_matrix[:, itemId - 1].nonzero()
    averageOfUser = Train_data_matrix[userId - 1][numpy.nonzero(Train_data_matrix[userId - 1])].mean()  # 获取userId 的平均值
    test = simility_matrix[:, userId - 1][userset]
    test1 = numpy.argsort(simility_matrix[:, userId - 1][userset])[0:knumber]
    test2 = simility_matrix[:, userId - 1][test1]
    Neighborusers = get_K_Neighbors(userId, userset, simility_matrix, knumber)

    # 计算每个用户的加权，预测
    for other in Neighborusers:
        sim = Neighborusers[other]
        averageOther = Train_data_matrix[other - 1][numpy.nonzero(Train_data_matrix[other - 1])].mean()  # 该用户的平均分
        # 累加
        simSums += abs(sim)  # 取绝对值
        jiaquanAverage += (Train_data_matrix[other - 1][itemId - 1] - averageOther) * sim  # 累加，一些值为负

    # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
    if simSums == 0:
        return averageOfUser
    else:
        return averageOfUser + jiaquanAverage / simSums


# 给定用户实例编号，和相似度矩阵，得到最相似的K个用户
def get_K_Neighbors(userinstance, neighborlist, SimNArray, k=10):
    rank = dict()
    for i in neighborlist[0]:
        rank.setdefault(i + 1, 0)  # 设置初始值，以便做下面的累加运算
        rank[i + 1] += SimNArray[userinstance - 1][i]
    # test=
    myresult = dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[
                    0:k])  # 用sorted方法对推荐的物品进行排序，预计评分高的排在前面，再取其中nitem个，nitem为每个用户推荐的物品数量
    return myresult


'''
def DataFrame2Matrix(ThisDataFrame, n_users, n_items):
    ThisMatrix = numpy.zeros((n_users, n_items))
    for line in ThisDataFrame.itertuples():
        ThisMatrix[line[1] - 1, line[2] - 1] = line[3]
    return ThisMatrix


class MyThread(Thread):
    def __init__(self, ThisDataFrame, n_users, n_items):
        Thread.__init__(self)
        self.DataFrame = ThisDataFrame
        self.n_users = n_users
        self.n_items = n_items

    def run(self):
        self.result = DataFrame2Matrix(self.DataFrame, self.n_users, self.n_items)

    def get_result(self):
        return self.result
'''

def sparse_argsort(arr):
    indices = np.nonzero(arr)[0]
    return indices[np.argsort(arr[indices])]

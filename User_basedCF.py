from DataHelper import *
from EvaluationHelper import *


class UBCollaborativeFilter(object):
    def __init__(self):
        self.SimilityMatrix = None
        self.truerating = []
        self.predictions = []
        self.train_data_matrix = None
        self.RMSE = dict()
        self.MAE = dict()
        self.UserMeanMatrix = None

    # 平均加权策略，预测userId对itemId的评分
    def getRating(self, Train_data_matrix, userId, simility_matrix, neighborset):
        # simSums为0，即该项目尚未被其他用户评分，这里的处理方法：返回用户平均分
        simSums = np.sum(simility_matrix[neighborset])
        # 获取userId 的平均值
        averageOfUser = self.UserMeanMatrix[userId]
        # 计算每个用户的加权，预测得分
        jiaquanAverage = (Train_data_matrix[neighborset]).dot(simility_matrix[neighborset])
        if simSums == 0:
            return 0
        else:
            return jiaquanAverage / simSums

    def doEvaluate(self, testDataMatrix, K):
        a, b = testDataMatrix.nonzero()
        for userIndex, itemIndex in zip(a, b):
            # 用户最相似的K个用户
            neighborset = get_K_Neighbors(self.train_data_matrix[:, itemIndex], self.SimilityMatrix[userIndex], K)
            # 基于训练集预测用户评分(用户数目<=K)
            prerating = self.getRating(self.train_data_matrix[:, itemIndex], userIndex, self.SimilityMatrix[userIndex], neighborset)
            self.truerating.append(testDataMatrix[userIndex][itemIndex])
            self.predictions.append(prerating)
            # print(len(self.predictions))
        self.RMSE[K] = RMSE(self.truerating, self.predictions)
        self.MAE[K] = MAE(self.truerating, self.predictions)
        print("UBCF  K=%d,RMSE:%f,MAE:%f" % (K, self.RMSE[K], self.MAE[K]))

# -*- coding: utf-8 -*-
import math

'''
# 1) 用scikit cosine_similarity计算余弦相似度
    from sklearn.metrics.pairwise import cosine_similarity
    user_similarity=cosine_similarity(user_item_matric)

# 2) 用scikit pairwise_distances计算相似度,用pairwise_distances计算的Cosine distance是1-（cosine similarity）结果
    from sklearn.metrics.pairwise import pairwise_distances
    user_similarity = pairwise_distances(user_item_matric, metric='cosine')
 '''


class DistanceHelper(object):
    # 1) given two data points, calculate the euclidean distance between them
    def Euclidean_distance(self, vector1, vector2):
        points = zip(vector1, vector2)
        diffs_squared_distance = [pow(a - b, 2) for (a, b) in points]
        return math.sqrt(sum(diffs_squared_distance))

    def Cosin_distance(self, vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product / ((normA * normB) ** 0.5)

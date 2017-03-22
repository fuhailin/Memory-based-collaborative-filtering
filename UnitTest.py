import random
import math
import time


class UserBasedCF:
    def __init__(self, datafile=None):
        self.datafile = datafile
        self.readData()
        self.splitData(3, 47)

    def readData(self, datafile=None):
        """ 
        read the data from the data file which is a data set 
        把文件中的内容读到data中"""
        self.datafile = datafile or self.datafile
        self.data = []
        for line in open(self.datafile):
            userid, itemid, record, _ = line.split()
            self.data.append((userid, itemid, int(record)))

    def splitData(self, k, seed, data=None, M=8):
        """ 
        split the data set 
        testdata is a test data set 
        traindata is a train set 
        test data set / train data set is 1:M-1 
        """
        self.testdata = {}
        self.traindata = {}
        data = data or self.data
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, M) == k:
                self.testdata.setdefault(user, {})
                self.testdata[user][item] = record
            else:
                self.traindata.setdefault(user, {})
                self.traindata[user][item] = record

    def userSimilarity(self, train=None):
        train = train or self.traindata
        self.userSim = dict()
        for u in train.keys():
            for v in train.keys():
                if u == v:
                    continue
                self.userSim.setdefault(u, {})
                self.userSim[u][v] = len(set(train[u].keys()) & set(train[v].keys()))
                self.userSim[u][v] /= math.sqrt(len(train[u]) * len(train[v]) * 1.0)

    def userSimilarityBest(self, train=None):
        """ 
        the other method of getting user similarity which is better than above 
        you can get the method on page 46 
        In this experiment，we use this method 
        """
        train = train or self.traindata
        self.userSimBest = dict()
        item_users = dict()
        for u, item in train.items():
            for i in item.keys():
                item_users.setdefault(i, set())
                item_users[i].add(u)
        user_item_count = dict()
        count = dict()
        for item, users in item_users.items():
            for u in users:
                user_item_count.setdefault(u, 0)
                user_item_count[u] += 1
                for v in users:
                    if u == v: continue
                    count.setdefault(u, {})
                    count[u].setdefault(v, 0)
                    count[u][v] += 1
        for u, related_users in count.items():
            self.userSimBest.setdefault(u, dict())
            for v, cuv in related_users.items():
                self.userSimBest[u][v] = cuv / math.sqrt(user_item_count[u] * user_item_count[v] * 1.0)

    def recommend(self, user, train=None, k=8, nitem=40):
        train = train or self.traindata
        rank = dict()
        interacted_items = train.get(user, {})
        for v, wuv in sorted(self.userSimBest[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in train[v].items():
                if i in interacted_items:
                    continue
                rank.setdefault(i, 0)
                rank[i] += wuv * rvi
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:nitem])

    def recallAndPrecision(self, train=None, test=None, k=8, nitem=10):
        """ 
        Get the recall and precision, the method you want to know is listed 
        in the page 43 
        """
        train = train or self.traindata
        test = test or self.testdata
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user, {})
            rank = self.recommend(user, train=train, k=k, nitem=nitem)
            for item, _ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem
        return (hit / (recall * 1.0), hit / (precision * 1.0))

    def coverage(self, train=None, test=None, k=8, nitem=10):
        train = train or self.traindata
        test = test or self.testdata
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
        """ 
        Get the popularity 
        the algorithm on page 44 
        """
        train = train or self.traindata
        test = test or self.testdata
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


class ItemBasedCF(object):
    def __init__(self, datafile=None):
        print("init:")
        self.datafile = datafile
        self.readData(self.datafile)
        self.splitData(3, 47)

    def readData(self, datafile=None):
        """ 
        read the data from the data file which is a data set 
        """
        print("ireadData:")
        self.datafile = datafile or self.datafile
        self.data = []
        for line in open(self.datafile):
            userid, itemid, record, _ = line.split()
            self.data.append((userid, itemid, int(record)))
            #      格式 [('196', '242', 3), ('186', '302', 3), ('22', '377', 1)]

    def splitData(self, k, seed, data=None, M=8):
        """ 
        split the data set
        testdata is a test data set 
        traindata is a train set  
        test data set / train data set is 1:M-1 
        """
        self.testdata = {}
        self.traindata = {}
        data = data or self.data
        random.seed(seed)
        for user, item, record in self.data:
            if random.randint(0, M) == k:
                self.testdata.setdefault(user, {})
                self.testdata[user][item] = record
            else:
                self.traindata.setdefault(user, {})
                self.traindata[user][item] = record
                # print(self.testdata)
                #        格式{'291': {'1042': 4, '118': 2}, '200': {'222': 5}, '308': {'1': 4}, '167': {'486': 4}, '122': {'387': 5}, '210': {'40': 3},

    def ItemSimilarity(self, train=None):
        """ 
        the other method of getting user similarity which is better than above 
        you can get the method on page 46 
        In this experiment，we use this method 
        """
        train = train or self.traindata
        self.itemSimBest = dict()
        N = dict()
        C = dict()
        for u, items in train.items():
            for i in items:
                N.setdefault(i, 0)
                N[i] += 1
                for j in items:
                    if i == j:
                        continue
                    C.setdefault(i, {})
                    C[i].setdefault(j, 0)
                    C[i][j] += 1
        for i, related_items in C.items():
            self.itemSimBest.setdefault(i, dict())
            for j, cij in related_items.items():
                self.itemSimBest[i][j] = cij / math.sqrt(N[i] * N[j] * 1.0)

    def recommend(self, user, train=None, k=8, nitem=10):
        train = train or self.traindata
        rank = dict()
        ru = train.get(user, {})
        for i, pi in ru.items():
            for j, wj in sorted(self.itemSimBest[i].items(), key=lambda x: x[1], reverse=True)[0:k]:
                if j in ru:
                    continue
                rank.setdefault(j, 0)
                rank[j] += pi * wj
        return dict(sorted(rank.items(), key=lambda x: x[1], reverse=True)[0:nitem])

    def recallAndPrecision(self, train=None, test=None, k=8, nitem=10):
        """ 
        Get the recall and precision, the method you want to know is listed  
        in the page 43 
        """
        train = train or self.traindata
        test = test or self.testdata
        hit = 0
        recall = 0
        precision = 0
        for user in train.keys():
            tu = test.get(user, {})
            rank = self.recommend(user, train=train, k=k, nitem=nitem)
            for item, _ in rank.items():
                if item in tu:
                    hit += 1
            recall += len(tu)
            precision += nitem
        return (hit / (recall * 1.0), hit / (precision * 1.0))

    def coverage(self, train=None, test=None, k=8, nitem=10):
        train = train or self.traindata
        test = test or self.testdata
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
        """ 
        Get the popularity 
        the algorithm on page 44 
        """
        train = train or self.traindata
        test = test or self.testdata
        item_popularity = dict()
        for user, items in train.items():
            for item in items.keys():
                item_popularity.setdefault(item, 0)
                item_popularity[item] += 1
        ret = 0
        n = 0
        # 对每一个user进行推荐 计算其流行度
        for user in train.keys():
            rank = self.recommend(user, train, k=k, nitem=nitem)
            for item, _ in rank.items():
                ret += math.log(1 + item_popularity[item])
                n += 1
        return ret / (n * 1.0)


def testUBCFRecommend():
    ubcf = ItemBasedCF('Datas/ml-100k/u.data')
    ubcf.readData()
    ubcf.splitData(4, 100)
    ubcf.ItemSimilarity()
    user = "345"
    rank = ubcf.recommend(user, k=3)
    for i, rvi in rank.items():
        items = ubcf.testdata.get(user, {})
        record = items.get(i, 0)
        print("%5s: %.4f--%.4f" % (i, rvi, record))


def testUserBasedCF():
    startTime = time.clock()
    cf = UserBasedCF('Datas/ml-100k/u.data')
    cf.userSimilarityBest()
    print("%3s%20s%20s%20s%20s%20s" % ('K', "recall", 'precision', 'coverage', 'popularity', 'time'))
    for k in [5, 10, 20, 40, 80, 160]:
        recall, precision = cf.recallAndPrecision(k=k)
        coverage = cf.coverage(k=k)
        popularity = cf.popularity(k=k)
        print("%3d%19.3f%%%19.3f%%%19.3f%%%20.3f%19.3fs" % (
        k, recall * 100, precision * 100, coverage * 100, popularity, time.clock() - startTime))


def testIBCFRecommend():
    ibcf = ItemBasedCF('Datas/ml-100k/u.data')
    ibcf.readData()
    ibcf.splitData(4, 100)
    ibcf.ItemSimilarity()
    user = "345"
    rank = ibcf.recommend(user, k=3)
    for i, rvi in rank.items():
        items = ibcf.testdata.get(user, {})
        record = items.get(i, 0)
        print("%5s: %.4f--%.4f" % (i, rvi, record))


def testItemBasedCF():
    startTime = time.clock()
    cf = ItemBasedCF('Datas/ml-100k/u.data')
    cf.ItemSimilarity()
    print("%3s%20s%20s%20s%20s%20s" % ('K', "recall", 'precision', 'coverage', 'popularity', 'time'))
    for k in [5, 10, 20, 40, 80, 160]:
        recall, precision = cf.recallAndPrecision(k=k)
        coverage = cf.coverage(k=k)
        popularity = cf.popularity(k=k)
        print("%3d%19.3f%%%19.3f%%%19.3f%%%20.3f%19.3fs" % (
        k, recall * 100, precision * 100, coverage * 100, popularity, time.clock() - startTime))


if __name__ == "__main__":
    testUserBasedCF()
    testItemBasedCF()

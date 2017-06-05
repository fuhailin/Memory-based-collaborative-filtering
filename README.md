# Memory-based-collaborative-filtering
contain User-based CF([UBCF](https://github.com/fuhailin/Memory-based-collaborative-filtering/blob/master/kNNUBCF.py)),Item-based CF([IBCF](https://github.com/fuhailin/Memory-based-collaborative-filtering/blob/master/kNNIBCF.py))
A k-nearest neighbors Recommender System use MovieLens dataset in Python

## User-based collaborative filter
K=25    RunTime：39s

RMSE:0.9426714727093027
MAE:0.738091820385
![image](https://github.com/fuhailin/Memory-based-collaborative-filtering/blob/master/Datas/IBCF.png)
![image](https://github.com/fuhailin/Memory-based-collaborative-filtering/blob/master/Datas/UBCF.png)

Memory-based algorithms are easy to implement and produce reasonable prediction quality.
The drawback of memory-based CF is that it doesn’t scale to real-world scenarios and doesn’t address the well-known cold-start problem, that is when new user or new item enters the system.

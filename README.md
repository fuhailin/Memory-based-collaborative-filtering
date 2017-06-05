# Memory-based-collaborative-filtering
contain User-based CF([UBCF](https://github.com/fuhailin/Memory-based-collaborative-filtering/blob/master/User_basedCF.py)),Item-based CF([IBCF](https://github.com/fuhailin/Memory-based-collaborative-filtering/blob/master/Item_basedCF.py))
A robust k-nearest neighbors Recommender System use MovieLens dataset in Python

## User-based collaborative filter
> *K=25    RunTime：1s
    RMSE:0.940611
    MAE:0.884748.*


![image](https://github.com/fuhailin/Memory-based-collaborative-filtering/blob/master/Docs/ml-100k/UBCF%20ml-100k%200.2.png)
![image](https://github.com/fuhailin/Memory-based-collaborative-filtering/blob/master/Docs/ml-100k/IBCF%20ml-100k%200.2.png)

Memory-based algorithms are easy to implement and produce reasonable prediction quality.
The drawback of memory-based CF is that it doesn’t scale to real-world scenarios and doesn’t address the well-known cold-start problem, that is when new user or new item enters the system.

import matplotlib.pyplot as plt
KList = [25, 50, 75, 100, 125, 150]
rmse=[0.821169,0.826170,0.830468,0.833642,0.836080,0.837998]
mae=[0.674318,0.682557,0.689677,0.694958,0.699030,0.702240]
plt.plot(KList, rmse, marker='o', label='RMSE')
plt.plot(KList, mae, marker='v', label='MAE')
plt.title('The Error of IBCF in MovieLens ml-10M')
plt.xlabel('K')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.savefig('123.png')
plt.show()
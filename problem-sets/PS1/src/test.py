import numpy as np
import util

train_path = '../data/ds1_train.csv'
x_train, y_train = util.load_dataset(train_path, add_intercept= False)

n,d = x_train.shape
# a = np.array([[1,10],[2,20],[3,30]])
# print(a)
# print(a[:,0:1])
# print(a[:,1:2])
# print(np.append(a[:,0:1], a[:,1:2], axis=1))
print(x_train)
print(np.append(x_train[:,0:1],np.log(x_train[:,1:2]), axis=1))
print(np.stack((x_train[:,0], np.log(x_train[:,1])), axis=1))

# n,d = x_train.shape
# y1 = sum(y_train)
# phi = y1/n
# mu1 = np.zeros(d)
# for i in range(n):
#     if y_train[i] : mu1 += x_train[i]
# mu0 = (sum(x_train) - mu1) / y1
# mu1 /= y1
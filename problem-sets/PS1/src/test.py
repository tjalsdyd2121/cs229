import numpy as np
import util

train_path = '../data/ds1_train.csv'
x_train, y_train = util.load_dataset(train_path, add_intercept= False)

n,d = x_train.shape
a = np.array([[1,10,100],[2,20,200],[3,30,300]])
print(a)
print(a[:,0:2])
print(a[:,2:3])
print(np.append(a[:,0:2], a[:,2:3], axis=1))
# print(x_train)
# print(np.append(x_train[:,0:1],np.log(x_train[:,1:2]), axis=1))
# print(np.stack((x_train[:,0], np.log(x_train[:,1])), axis=1))
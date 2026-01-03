import numpy as np
import util

train_path = '../data/ds1_train.csv'
x_train, y_train = util.load_dataset(train_path, add_intercept= False)

n,d = x_train.shape

x = np.array([[1,1,3], 
              [1,5,8],
              [5,2,9]])
y = np.array([1,1,0])
theta = np.array([1,2,3])

print(x.dot(theta)- y)
print(y)

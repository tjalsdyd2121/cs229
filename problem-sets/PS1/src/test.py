import numpy as np
import util
print("Hello, World!")

train_path = '../data/ds1_train.csv'
x_train, y_train = util.load_dataset(train_path, add_intercept= True)

n,d = x_train.shape
print(n,d)

import numpy as np
import util
import matplotlib
matplotlib.use('Agg')
#
import matplotlib.pyplot as plt

#from linear_model import LinearModel
from p01b_logreg import main as p01b
from p01b_logreg import LogisticRegression as LR
from p01e_gda import main as p01e
from p01e_gda import GDA

def main(train_path, eval_path, pred_path):

    x_train, y_train = util.load_dataset(train_path, add_intercept = False)
    gda = GDA()
    gda.fit(x_train,y_train)
    logreg = LR(eps=1e-5)
    logreg.fit(util.add_intercept(x_train), y_train)

    print(train_path,logreg.theta,gda.theta)

    #x,y = util.load_dataset(eval_path, add_intercept=True)
    x = util.add_intercept(x_train)
    y = y_train
    
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    models = [[logreg, 'red', 'logistic regression'],[gda,'black', 'GDA']]
    for i,j,k in models :
        margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
        x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
        x2 = -(i.theta[0] / i.theta[2]  + i.theta[1] / i.theta[2] * x1)
        plt.plot(x1, x2, c=j, linewidth=2)
    #plt.legend(loc="upper left")
    plt.savefig('output/p01fg_{}.png'.format(pred_path[-5]))
    # by the figure, GDA performs badly at ds1, 
    # whereas GDA and Logistic Regression perform exactly samely at ds2.
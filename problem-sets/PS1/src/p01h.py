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
    # The reason of GDA does not perform well at ds1 
    # is x2 of ds1 is all positive and its scale doesn't match with x1.
    # in other words, p(x | y) is not gaussian form.
    # so we can take the log to x2 
    # this is called 'Box-cox transformation'.

    x_trans = np.append(x_train[:,0:1], np.log(x_train[:,1:2]), axis=1)
    t_gda = GDA()
    t_logreg = LR(eps = 1e-5)
    t_gda.fit(x_trans,y_train)
    t_logreg.fit(util.add_intercept(x_trans),y_train)

    x = util.add_intercept(x_trans)
    y = y_train
    
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    models = [[t_logreg, 'red', 'logistic regression'],[t_gda,'black', 'GDA']]
    for i,j,k in models :
        margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
        x1 = np.arange(min(x[:, -2])-margin1, max(x[:, -2])+margin1, 0.01)
        x2 = -(i.theta[0] / i.theta[2]  + i.theta[1] / i.theta[2] * x1)
        plt.plot(x1, x2, c=j, linewidth=2)
    #plt.legend(loc="upper left")
    plt.savefig('output/p01h_{}.png'.format(pred_path[-5]))

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    x_eval_trans = np.append(x_eval[:,0:2], np.log(x_eval[:,2:3]), axis=1)
    y_pred = t_gda.predict(x_eval_trans)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    print("transformed gda accuracy: {}".format(np.mean((y_pred > 0.5) == y_eval)))


    


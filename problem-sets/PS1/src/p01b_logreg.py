import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    logReg_Netwon = LogisticRegression(eps=1e-5)
    logReg_Netwon.fit(x_train, y_train)
    
    # plot the data and model.
    util.plot(x_train, y_train, logReg_Netwon.theta, 'output/p01b_{}.png'.format(pred_path[-5]))

    # prediction
    y_pred = logReg_Netwon.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    print("accuracy: {}".format(np.mean((y_pred > 0.5) == y_eval)))


    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        """
        Out feature is two dimensions.
        Let d be the #feature which is 3 since we have intercept.
        Let n be the #training set.
        Then we can extract d & n from actual data file named ds1_train.csv
        """
        n,d = x.shape
        # n = 800, d = 3
        # initializing the theta as zero vector.
        self.theta = np.zeros(d) 
        # d is 3 dimesion array. so it is [0,0,0]

        # run Newton's method until acheive max_iter or eps
        # To runt the Newton's method, we save the previous theta and
        # find inverse of Hessian matrix and gradient of l(theta)).

        while True :
            pre_theta = np.copy(self.theta) 
            # by definition of logistic regression
            h_x =  1 / (1 + np.exp(-x.dot(self.theta)))
            # hessain and gradient we found as (a)
            gradient = x.T.dot(h_x - y)/ n
            H = (x.T * h_x * (1 - h_x)).dot(x) / n
            # update the theta
            self.theta -= np.linalg.inv(H).dot(gradient)

            # break condition
            if np.linalg.norm(self.theta - pre_theta, ord=2) < self.eps:
                break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        
        # in logistic regression, predicted by h_x which is prob. that y = 1
        h_x =  1 / (1 + np.exp(-x.dot(self.theta)))
        return h_x

        # *** END CODE HERE ***

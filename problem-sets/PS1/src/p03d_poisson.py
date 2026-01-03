import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    x_eval, y_eval = util.load_dataset(eval_path,add_intercept= True)
    PR = PoissonRegression(step_size= lr)
    # Fit a Poisson Regression model
    PR.fit(x_train, y_train)
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # plot the data and model.
    util.plot(x_train, y_train, PR.theta, 'output/p03d_{}.png'.format(pred_path[-5]))
    y_pred = PR.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    print("LR accuracy: {}".format(np.mean((y_pred > 0.5) == y_eval)))
    
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        n,d = x.shape
        # here, ds4 has x1,x2,x3 and x4.
        # so n = 800, d = 4 + 1 = 5 including intercept.
        # note that x1 and x2 are {0,1}, x3 and x4 are [0,1]

        # initializing theta as zero vector with d-dimensions.
        self.theta = np.zeros(d)

        while self.max_iter > 0:
            pre_theta = np.copy(self.theta)
            # use batch gradient ascent.
            for i in range(n):
                self.theta += self.step_size * x[i].T.dot(np.sum(y - np.exp(x.dot(self.theta))))
            self.max_iter -= 1
            if np.norm(pre_theta - self.theta, ord= 2) < self.eps: break

        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x.dot(self.theta))
        # *** END CODE HERE ***

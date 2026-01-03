import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    # Since we have closed form for theta_0 and theta, we have to calculate them with d-dimensional vectors : mu's
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    
    
    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train,y_train)
    print(model.theta)
    # plot the data and model
    util.plot(x_train, y_train, model.theta, 'output/p01e_{}.png'.format(pred_path[-5]))



    # prediction
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    print("accuracy: {}".format(np.mean((y_pred > 0.5) == y_eval)))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        n,d = x.shape
        # useful lang & package... python & numpy...!!
        y_1 = sum(y == 1)
        phi = y_1 / n
        mu_0 = np.sum(x[y == 0], axis=0) / (n - y_1)
        mu_1 = np.sum(x[y == 1], axis=0) / y_1
        sigma = ((x[y == 0] - mu_0).T.dot(x[y == 0] - mu_0) + (x[y == 1] - mu_1).T.dot(x[y == 1] - mu_1)) / n
        sigma_inv = np.linalg.inv(sigma)
        # from (c)
        self.theta = np.zeros(d+1)

        self.theta[0] = 0.5 * mu_0.T.dot(sigma_inv).dot(mu_0) - 0.5 * mu_1.T.dot(sigma_inv).dot(mu_1) - np.log((1-phi)/phi)
        self.theta[1:] = sigma_inv.dot(mu_1-mu_0)
        # Alternatively, by some algebra
        # self.theta[0] = 0.5 * (mu_0 + mu_1).dot(sigma_inv).dot(mu_0 - mu_1) - np.log((1 - phi) / phi)
        # Or by other expression,
        # ".dot()" can be replaced by "@"
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta)))
        # *** END CODE HERE

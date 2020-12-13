"""
Linear regression implementation and demo.

Reference:
- Friedman J, Hastie T, Tibshirani R. The elements of statistical learning.
  2001. Chapter 3.2
"""

import numpy as np


class LinearRegression:
    """
    y = X @ beta

    y.shape is (n_samples, 1)
    X.shape is (n_samples, n_features)
    beta.shape is (n_features, 1)
    `@` denotes matrix multiplication
    """
    def __init__(self, fit_intercept=True):
        """
        If fit_intercept is True, add y-intercept coefficient to the beta.
        """
        self.beta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        Find beta that minimizes the loss using least-squares.

        loss = |X @ beta - y|^2
             = (X @ beta - y).T @ (X @ beta - y)
             = y.T @ y - y.T @ X @ beta - beta.T @ X.T @ y
               + beta.T @ X.T @ X @ beta

        d(loss)/d(beta) = -y.T @ X - X.T @ y + 2 beta.T @ X.T @ X
                        = -2 y.T @ X + 2 beta.T @ X.T @ X
                        = 0 (to find the minimum point)

        2 beta.T @ X.T @ X = 2 y.T @ X
        beta.T @ X.T @ X   = y.T @ X
        X.T @ X @ beta     = X.T @ y

        beta               = (X.T @ X)^-1 @ X.T @ y
        """
        if self.fit_intercept:
            intercept = np.ones(X.shape[0])
            X = np.column_stack((intercept, X))

        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        y = X @ beta
        """
        return X @ self.beta


def demo_linear_regression(*,
                           noise_scale=0.1,
                           n_samples=100,
                           true_beta=[0.1, 0.5, 0.9, 0.1, 0.7]) -> None:
    """
    Try adjusting `noise_scale` and see how the fitted beta
    and mean-squared loss changes.
    """
    n_features = len(true_beta)
    X = np.random.rand(n_samples, n_features)
    noise = np.random.normal(scale=noise_scale, size=(n_samples,))
    true_beta = np.array(true_beta)
    y = (X @ true_beta) + noise

    linear_model = LinearRegression(fit_intercept=False)
    linear_model.fit(X, y)
    y_pred = linear_model.predict(X)

    diff = y_pred - y
    msl = np.sum(diff.T @ diff) / n_samples
    print(f"noise_scale = {noise_scale}")
    print(f"mean-squared loss = {msl:.3f}")
    print(f"true_beta = {true_beta}")
    print(f"fitted beta = {linear_model.beta}")


if __name__ == "__main__":
    for noise in [0.1, 0.5, 1]:
        demo_linear_regression(noise_scale=noise)
        print()

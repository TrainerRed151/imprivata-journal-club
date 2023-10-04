import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=1e-3, iterations=1000000, normalize=False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.normalize = normalize

    def _gradient(self, X, b, y):
        return (X.T @ X @ b) - (X.T @ y)

    def _normalize(self, X, y):
        norm_X_trans = []
        for col in X.T:
            norm_X_trans.append((col - col.mean())/col.std())

        X = np.array(norm_X_trans).T
        y = y = (y - y.mean())/y.std()

        return X, y

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(X.shape[0], 1)

        if self.normalize:
            X, y = self._normalize(X, y)

        X = np.hstack([X, np.ones(X.shape[0]).reshape(X.shape[0], 1)])
        b = np.zeros((X.shape[1], 1))

        for _ in range(self.iterations):
            grad = self._gradient(X, b, y)
            b -= self.learning_rate*grad

        self.coef_ = b.flatten()[:-1]
        self.intercept_ = b.flatten()[-1]

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression as LinearRegressionSK
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)

    reg_me = LinearRegression().fit(X, y)
    reg_sk = LinearRegressionSK().fit(X, y)

    print('Me:')
    print(reg_me.coef_)
    print(reg_me.intercept_)

    print()

    print('SK:')
    print(reg_sk.coef_)
    print(reg_sk.intercept_)

    print()

    print('Analytical:')
    print((np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)).T)

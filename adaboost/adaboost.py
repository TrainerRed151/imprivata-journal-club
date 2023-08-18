import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators

    def boost(self, X, y, sample_weight):
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y, sample_weight)
        y_pred = clf.predict(X)

        total_error = 0
        incorrect_label_set = set()
        for i in range(len(y)):
            if y_pred[i] != y[i]:
                total_error += sample_weight[i]
                incorrect_label_set.add(i)

        if total_error == 0:
            print('WARNING: Divide by zero')

        alpha = 0.5*np.log((1 - total_error)/total_error)

        new_sample_weight = []
        for i, weight in enumerate(sample_weight):
            if i in incorrect_label_set:
                new_weight = weight*np.exp(alpha)
            else:
                new_weight = weight*np.exp(-alpha)

            new_sample_weight.append(new_weight)

        new_sample_weight = np.array(new_sample_weight)
        sample_weight = new_sample_weight / np.sum(new_sample_weight)

        return clf, alpha, sample_weight

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n = len(y)

        sample_weight = np.ones(n)/n

        self.weak_learners = []
        for _ in range(self.n_estimators):
            clf, alpha, sample_weight = self.boost(X, y, sample_weight)
            self.weak_learners.append((clf, alpha))

    def predict(self, X):
        X = np.array(X)
        n = X.shape[0]
        y0 = np.zeros(n)
        y1 = np.zeros(n)
        y = np.zeros(n)

        for clf, alpha in self.weak_learners:
            y_weak = clf.predict(X)
            y0 += alpha*(1 - y_weak)
            y1 += alpha*y_weak

        for i in range(n):
            if y1[i] > y0[i]:
                y[i] = 1

        return y


if __name__ == '__main__':
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=10000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    X_test, _ = make_classification(n_samples=20, n_features=4, n_informative=2, n_redundant=0, random_state=22, shuffle=False)

    clf_mine = AdaBoost()
    clf_mine.fit(X, y)
    y_pred_mine = clf_mine.predict(X_test)
    print(f'Mine: {y_pred_mine}')

    clf_sk = AdaBoostClassifier(n_estimators=50)
    clf_sk.fit(X, y)
    y_pred_sk = clf_sk.predict(X_test)
    print(f'sklearn: {y_pred_sk}')

    print(f'Comp: {y_pred_mine == y_pred_sk}')

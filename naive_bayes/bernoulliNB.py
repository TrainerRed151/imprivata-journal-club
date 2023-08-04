import numpy as np
import heapq


class BernoulliNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        self.classes, counts = np.unique(y, return_counts=True)
        self.priors = dict(zip(self.classes, counts/sum(counts)))

        self.likelihoods = {}
        for cl in self.classes:
            class_X = X[y == cl]
            feature_occurances = np.sum(class_X, axis=0) + self.alpha
            self.likelihoods[cl] = feature_occurances / sum(feature_occurances)

    def predict(self, X):
        X = np.array(X)
        y = np.zeros(X.shape[0])

        for i, row in enumerate(X):
            class_probabilities = []

            for cl in self.classes:
                prob = np.log(self.priors[cl])
                for feature, frequency in enumerate(row):
                    prob += np.log(self.likelihoods[cl][feature])*frequency

                heapq.heappush(class_probabilities, (-prob, cl))

            y[i] = class_probabilities[0][1]

        return y


if __name__ == '__main__':
    rng = np.random.RandomState(22)
    X = rng.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 4, 5])

    clf_mine = BernoulliNB()
    clf_mine.fit(X, y)
    print(f'Mine: {clf_mine.predict(X[2:3])}')


    from sklearn.naive_bayes import BernoulliNB as skBernoulliNB
    clf_sk = skBernoulliNB()
    clf_sk.fit(X, y)
    print(f'sklearn: {clf_sk.predict(X[2:3])}')

    X2 = rng.randint(5, size=(3, 100))
    print(f'Mine: {clf_mine.predict(X2)}')
    print(f'sklearn: {clf_sk.predict(X2)}')

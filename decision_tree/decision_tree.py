import numpy as np


class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, data=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.data = data

    def get_feature(self):
        return self.feature

    def get_value(self):
        return value

    def get_data(self):
        return self.data

    def is_binary(self):
        return self.value is None

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def set_left(self, left)
        self.left = left

    def set_right(self, right)
        self.right = right


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = Node(None)

    def feature_split(self, X, y)
        gini_impurity = []

        for i in range(X.shape[1]):
            split = (0, 0, 0, 0)
            col = X.A[:,i]
            if set(X.A[:,i]) == set([0, 1]):
                split[0] = np.sum(np.logical_and(X[:,i].T, y))
                split[1] = np.sum(np.logical_and(X[:,i].T, np.logical_not(y)))
                split[2] = np.sum(np.logical_and(np.logical_not(X[:,i].T), y))
                split[3] = np.sum(np.logical_and(np.logical_not(X[:,i].T), np.logical_not(y)))

                left = 1 - (split[0]/(split[0]+split[1]))**2 - (split[1]/(split[0]+split[1]))**2
                right = 1 - (split[2]/(split[2]+split[3]))**2 - (split[2]/(split[2]+split[3]))**2
                gini = ((split[0]+split[1])/sum(split))*left + ((split[2]+split[3])/sum(split))*right

            else:
                l = X[:,i].T.tolist()[0]
                l.sort()
                mids = []
                for i in range(len(l)-1):
                    mids.append((l[i] + l[i+1])/2)

                gini = 1
                for m in mids:
                    


            gini_impurity.append(gini)

            return np.argmin(gini_impurity)

    def fit(self, X, y):
        X = np.matrix(X)
        y = np.matrix(y)
        root = Node(data=(X, y))


if __name__ == '__main__':
    from sklearn.datasets import load_wine
    from sklearn.cluster import KMeans as sKMeans

    data_dict = load_wine()

    kmeans = KMeans(3)
    kmeans.load_data(data_dict['data'])

    kmeans.fit()

    print('True:')
    print(data_dict['target'])

    print('\nMine:')
    print(kmeans.classes)

    skmeans = sKMeans(n_clusters=3, n_init='auto', random_state=24)
    skmeans.fit(data_dict['data'])

    print('\nsklearn:')
    print(skmeans.labels_)


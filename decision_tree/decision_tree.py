import numpy as np


class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, X=None, y=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.X = X
        self.y = y
        self.term = -1

    def get_feature(self):
        return self.feature

    def get_value(self):
        return self.value

    def getX(self):
        return self.X

    def getY(self):
        return self.y

    def is_binary(self):
        return self.value is None

    def get_left(self):
        return self.left

    def get_right(self):
        return self.right

    def set_left(self, left):
        self.left = left

    def set_right(self, right):
        self.right = right

    def set_term(self, v):
        self.term = v

    def get_term(self):
        return self.term


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = Node()

    def gini(self, col, y):
        split = [0, 0, 0, 0]
        split[0] = np.sum(np.logical_and(col, y))
        split[1] = np.sum(np.logical_and(col, np.logical_not(y)))
        split[2] = np.sum(np.logical_and(np.logical_not(col), y))
        split[3] = np.sum(np.logical_and(np.logical_not(col), np.logical_not(y)))

        left = 1 - (split[0]/(split[0]+split[1]))**2 - (split[1]/(split[0]+split[1]))**2
        right = 1 - (split[2]/(split[2]+split[3]))**2 - (split[3]/(split[2]+split[3]))**2
        gini = ((split[0]+split[1])/sum(split))*left + ((split[2]+split[3])/sum(split))*right

        return gini

    def feature_split(self, X, y):
        gini_impurity = []
        value = []

        for i in range(X.shape[1]):
            mid = None
            col = X.A[:,i]

            if set(col) == set([0, 1]):
                min_gini = self.gini(col, y)
                mid = None
            else:
                l = col.tolist()
                l.sort()
                mids = []
                for j in range(len(l)-1):
                    mids.append((l[j] + l[j+1])/2)

                min_gini = 2
                for m in mids:
                    binary_col = col < m
                    temp_gini = self.gini(binary_col, y)
                    if temp_gini < min_gini:
                        mid = m
                        min_gini = temp_gini



            value.append(mid)
            gini_impurity.append(min_gini)


        if len(gini_impurity) == 0:
            return 0, None

        idx = np.argmin(gini_impurity)
        return idx, value[idx]

    def tree_split(self, node, big_feat_list):
        y = node.getY()
        X = node.getX()
        if X is None or X.shape[1] == 0:
            v = np.round(np.sum(y)/y.shape[1])
            node.set_term(v)
            return

        feature = big_feat_list.index(node.get_feature())
        X_left = []
        y_left = []
        X_right = []
        y_right = []

        if node.is_binary():
            for r in range(X.shape[0]):
                if X[r,feature] == 1:
                    y_left.append(y[0,r])
                    X_left.append(np.delete(X.A[r,:].tolist(), feature))
                else:
                    y_right.append(y[0,r])
                    X_right.append(np.delete(X.A[r,:].tolist(), feature))

        else:
            value = node.get_value()
            for r in range(X.shape[0]):
                if X[r,feature] < value:
                    y_left.append(y[0,r])
                    X_left.append(np.delete(X.A[r,:].tolist(), feature))
                else:
                    y_right.append(y[0,r])
                    X_right.append(np.delete(X.A[r,:].tolist(), feature))


        X_left = np.matrix(X_left)
        y_left = np.matrix(y_left)
        X_right = np.matrix(X_right)
        y_right = np.matrix(y_right)

        left_feature, left_value = self.feature_split(X_left, y_left)
        right_feature, right_value = self.feature_split(X_right, y_right)

        big_feat_list.pop(feature)

        if len(big_feat_list) == 0:
            v = np.round(np.sum(y)/y.shape[1])
            node.set_term(v)
            return

        left = Node(feature=big_feat_list[left_feature], value=left_value, X=X_left, y=y_left)
        right = Node(feature=big_feat_list[right_feature], value=right_value, X=X_right, y=y_right)

        node.set_left(left)
        node.set_right(right)

        left_big_feat_list = big_feat_list.copy()
        right_big_feat_list = big_feat_list.copy()

        self.tree_split(left, left_big_feat_list)
        self.tree_split(right, right_big_feat_list)

    def fit(self, X, y):
        X = np.matrix(X)
        y = np.matrix(y)


        feature, value = self.feature_split(X, y)
        self.root = Node(feature=feature, X=X, y=y, value=value)

        self.tree_split(self.root, list(range(X.shape[1])))

    def predict(self, X):
        y = []
        for row in X:
            curr = self.root
            while curr.get_term() == -1:
                feature = curr.get_feature()
                if curr.get_value() is not None:
                    if row[feature] < curr.get_value():
                        curr = curr.get_left()
                    else:
                        curr = curr.get_right()

                else:
                    if row[feature] == 1:
                        curr = curr.get_left()
                    else:
                        curr = curr.get_right()

            y.append(curr.get_term())

        return y

    def print_helper(self, node):
        if node is None:
            return

        print(f'({node.get_feature()}, {node.get_value()}, {node.get_term()})')

        self.print_helper(node.left)
        self.print_helper(node.right)

    def print_tree(self):
        self.print_helper(self.root)



if __name__ == '__main__':
    X = [[1, 1, 7],
         [1, 0, 12],
         [0, 1, 18],
         [0, 1, 35],
         [1, 1, 38],
         [1, 0, 50],
         [0, 0, 83]]

    y = [0, 0, 1, 1, 1, 0, 0]

    tree = DecisionTree()
    tree.fit(X, y)
    tree.print_tree()

    print(tree.predict(X))

import numpy as np

class NeuralNetwork:
    def __init__(self, hidden_layer_sizes=[100], learning_rate=0.001, max_iter=200, random_state=None):
        self.hidden_size = hidden_layer_sizes[0]
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        if random_state:
            np.random.seed(random_state)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self.sigmoid(final_input)

        return final_output

    def backward(self, X, y, yp):
        y = y.reshape(y.shape[0], 1)
        residual = y - yp
        delta_yp = residual * self.sigmoid_derivative(yp)

        error_hidden = delta_yp.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(delta_yp) * self.learning_rate
        self.bias_output += np.sum(delta_yp) * self.learning_rate
        self.weights_input_hidden += X.T.dot(delta_hidden) * self.learning_rate
        self.bias_hidden += np.sum(delta_hidden) * self.learning_rate

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        input_size = X.shape[1]
        output_size = 1 if len(y.shape) == 1 else y.shape[1]

        self.weights_input_hidden = 2*np.random.rand(input_size, self.hidden_size) - 1
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = 2*np.random.rand(self.hidden_size, output_size) - 1
        self.bias_output = np.zeros((1, output_size))

        for i in range(self.max_iter):
            yp = self.forward(X)
            self.backward(X, y, yp)

        return self

    def predict_proba(self, X):
        return self.forward(X)[:,0]

    def predict(self, X):
        yp = self.forward(X)
        yb = []
        for p in yp:
            yb.append(1 if p > 0.5 else 0)

        return yb

if __name__ == "__main__":
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    clf_me = NeuralNetwork(random_state=1, max_iter=5000).fit(X_train, y_train)
    clf_sk = MLPClassifier(random_state=1, max_iter=5000, activation='logistic', solver='sgd', batch_size=X_train.shape[0], shuffle=False).fit(X_train, y_train)

    print(f'Me: {clf_me.predict(X_test[:6, :])}')
    print(f'SK: {clf_sk.predict(X_test[:6, :])}')

    print()

    print(f'Me: {clf_me.predict_proba(X_test[:6, :])}')
    print(f'SK: {clf_sk.predict_proba(X_test[:6, :])[:,1]}')

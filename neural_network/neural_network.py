import numpy as np

class HiddenLayer:
    def __init__(self, input_size, output_size):
        self.weights = 2 * np.random.rand(output_size, input_size+1) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.X = np.vstack([X, np.ones((1, X.shape[1]))])  # Add a row of 1s for bias
        z = self.weights @ self.X
        self.output = self.sigmoid(z)
        return self.output

    def backward(self, dz):
        dw = dz @ self.X.T
        dx = self.weights.T @ dz
        dx = dx[:-1, :]  # Remove the last row corresponding to the bias
        return dw, dx


class NeuralNetwork:
    def __init__(self, hidden_layer_sizes=[100], learning_rate=1e-3, max_iter=1000, batch_size=1000, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size

        if random_state:
            np.random.seed(random_state)

    def forward_propagation(self, X):
        x = X
        for layer in self.hidden_layers:
            x = layer.forward(x)
        return self.output_layer.forward(x)

    def backward_propagation(self, X, y, y_hat):
        m = X.shape[1]
        dz = y_hat - y
        gradients = []

        # Output layer
        dw, dx = self.output_layer.backward(dz)
        gradients.append((dw, dx))

        # Hidden layers
        for i in range(len(self.hidden_layers)-1, -1, -1):
            dz = dx * self.hidden_layers[i].sigmoid_derivative(self.hidden_layers[i].output)
            dw, dx = self.hidden_layers[i].backward(dz)
            gradients.insert(0, (dw, dx))

        return gradients

    def update_parameters(self, gradients, learning_rate):
        for i, layer in enumerate(self.hidden_layers + [self.output_layer]):
            layer.weights -= learning_rate * gradients[i][0]

    def fit(self, X, y):
        if len(y.shape) == 1:
            y = np.array(y).reshape(X.shape[0], 1)

        self.input_size = X.shape[1]
        self.output_size = y.shape[1]

        X = X.T
        y = y.T

        # Initialize Hidden Layers
        self.hidden_layers = [HiddenLayer(self.input_size, self.hidden_layer_sizes[0])]
        for i in range(1, len(self.hidden_layer_sizes)):
            self.hidden_layers.append(HiddenLayer(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i]))
        self.output_layer = HiddenLayer(self.hidden_layer_sizes[-1], self.output_size)

        for _ in range(self.max_iter):
            y_hat = self.forward_propagation(X)
            gradients = self.backward_propagation(X, y, y_hat)
            self.update_parameters(gradients, self.learning_rate)

        return self

    def predict_proba(self, X):
        X = X.T
        return self.forward_propagation(X)

    def predict(self, X):
        y_hat = self.predict_proba(X)
        yp = y_hat > 0.5

        return yp.astype(int)


if __name__ == "__main__":
    from sklearn.neural_network import MLPClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

    clf_me = NeuralNetwork(random_state=1, max_iter=10_000, hidden_layer_sizes=[100]).fit(X_train, y_train)
    clf_sk = MLPClassifier(random_state=1, max_iter=10_000, hidden_layer_sizes=[100], activation='logistic', solver='sgd', batch_size=X_train.shape[0], shuffle=False).fit(X_train, y_train)

    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    print(f'Me: {clf_me.predict(X_test[:6, :])}')
    print(f'SK: {clf_sk.predict(X_test[:6, :])}')

    print()

    print(f'Me: {clf_me.predict_proba(X_test[:6, :])}')
    print(f'SK: {clf_sk.predict_proba(X_test[:6, :])[:,1]}')

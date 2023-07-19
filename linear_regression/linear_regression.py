import numpy as np

filename = 'data.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

y = data[:,0]
n = len(y)
y = y.reshape((n, 1))

y = (y - y.mean())/y.std()

X = data[:,1:]
#X = np.hstack([X, np.ones((n, 1))])
#X = np.concatenate([X, np.ones_like(y)], axis=1)
#print(X)

norm_X_trans = []
for col in X.T:
    norm_X_trans.append((col - col.mean())/col.std())

X = np.array(norm_X_trans).T

learning_rate = 1e-3
iterations = int(1e5)

w = np.zeros((X.shape[1], 1))
b = 0

for _ in range(iterations):
    yp = np.dot(X, w) + b
    residuals = yp - y
    dw = np.dot(X.T, residuals) / n
    #db = np.sum(residuals) / n

    w = w - learning_rate * dw
    #b = b - learning_rate * db


print('my weights:')
print(w.T)
#print('my bias:')
#print(b)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)
print('sklearn weights:')
print(reg.coef_)
print('sklearn bias:')
print(reg.intercept_[0])

print('Diff:')
print(f'w: {np.sum(np.abs(w.T - reg.coef_))}')
#print(f'b: {abs(b - reg.intercept_[0])}')

print((np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)).T)

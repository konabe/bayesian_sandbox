import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import multivariate_normal, norm
# model y_n = w.Tx_n + \epsilon
# noise \epsion \sim \mathcal{N}(\epsilon_n|0, \lambda^{-1})

M = 4
lamb = 10.0

# 基底関数を通したもの
def phi(x, M):
    return np.array([np.power(x, m) for m in range(M)])


plt.figure()
# prior p(w) = N(w|m, Lambda)plt.figure()
m = np.zeros(M)
Lambda = np.identity(M)
p = multivariate_normal(mean=m, cov=Lambda)

for i in range(20):
    w = p.rvs()
    x = np.linspace(-1, 1, 1000)
    y = np.dot(phi(x, M).T, w)
    plt.plot(x, y)

plt.ylim([-3, 3])

plt.figure()

epsilon = norm(loc=0, scale=1./lamb)
x = np.linspace(-1, 1, 1000)
plt.plot(x, np.dot(phi(x, M).T, w))
x = np.linspace(-1, 1, 10)
plt.plot(x, np.dot(phi(x, M).T, w) + epsilon.rvs(len(x)), '.')
plt.ylim([-3, 3])

plt.plot()

plt.show()

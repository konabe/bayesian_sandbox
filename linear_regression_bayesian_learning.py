import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

from scipy.stats import multivariate_normal, norm

# sin(x)の学習
plt.figure()

# 基底関数
def phi(x, M):
    return np.array([np.power(x, m) for m in range(M)])

# 学習データ
lamb = 10
N = 10
M = 10
x = np.random.rand(N) * 6
noise = norm(0, 1./lamb)
y = 1.5*np.sin(x) + noise.rvs(N)

# 事前分布
m = np.zeros(M)
Lamb = np.identity(M)
# 特徴量
x_phi = phi(x, M)

# 事後分布
Lamb_post = lamb*np.sum([np.tensordot(x_n, x_n, axes=0) for x_n in x_phi.T], axis=0) + Lamb
m_post = np.dot(LA.inv(Lamb_post), lamb*np.sum([y_n*x_n for x_n, y_n in zip(x_phi.T, y)], axis=0) + np.dot(Lamb, m))

# 予測分布
x_for_d = np.linspace(-1, 7, 100)
x_phi_for_d = phi(x_for_d, M)
mu_star = np.dot(m_post, x_phi_for_d)
lamb_star = 1. / (1./lamb + np.array([np.dot(np.dot(x_phi_n.T, LA.inv(Lamb_post)), x_phi_n) for x_phi_n in x_phi_for_d.T]))

plt.plot(x, y, '.')
plt.plot(x_for_d, mu_star)
plt.fill_between(x_for_d, mu_star-np.sqrt(1./lamb_star), mu_star+np.sqrt(1./lamb_star), alpha=0.1)
plt.xlim([-1, 7])
plt.ylim([-2, 2])
plt.title("predictive distribution for linear regression")

plt.show()

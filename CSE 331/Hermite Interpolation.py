"""
@Author-anonymousP
@__WHEN YOU FEEL LIKE QUITTING, THINK ABOUT WHY YOU STARTED__@
"""


import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from numpy.polynomial import Polynomial


def l(k, x):
    n = len(x)
    assert (k < len(x))

    x_k = x[k]
    x_copy = np.delete(x, k)

    denominator = np.prod(x_copy - x_k)

    coeff = []

    for i in range(n):
        coeff.append(sum([np.prod(x) for x in combinations(x_copy, i)]) * (-1) ** (i) / denominator)

    coeff.reverse()

    return Polynomial(coeff)

def h(k, x):
    # initialize with none
    l_k = None
    l_k_sqr = None
    l_k_prime = None
    coeff = None
    p = None
    # place your code here!!!!!!!!!!!!!!!!!!!!!!!

    l_k = l(k,x)
    l_k_sqr = l_k*l_k
    l_k_prime = l_k.deriv(1)
    coeff = [1 + 2*x[k]*l_k_prime(x[k]), (-2)*l_k_prime(x[k])]
    p = Polynomial(coeff)

    return p * l_k_sqr

def h_hat(k, x):
    # Initialize with none
    l_k = None
    l_k_sqr = None
    coeff = None
    p = None
    # place your code here!!!!!!!!!!!!!!!!!!!!!!!

    l_k = l(k,x)
    l_k_sqr = l_k*l_k
    coeff = [(-1)*x[k], 1]
    p = Polynomial(coeff)

    return p * l_k_sqr


def hermit(x, y, y_prime):
    assert (len(x) == len(y))
    assert (len(y) == len(y_prime))

    f = Polynomial([0.0])
    for i in range(len(x)):
        # f += ?
        pass  # pass statement does nothing
        # place your code here!!!!!!!!!!!!!!!!!!!!!!!
        f += y[i] * h(i, x) + y_prime[i] * h_hat(i, x)
    return f


pi = np.pi
x       = np.array([0.0, pi/2.0,  pi, 3.0*pi/2.0])
y       = np.array([0.0,    1.0, 0.0,       -1.0])
y_prime = np.array([1.0,    0.0, 1.0,        0.0])


n = 1
f3     = hermit(x[:(n+1)], y[:(n+1)], y_prime[:(n+1)])
data   = f3.linspace(n=50, domain=[-3, 3])
test_x = np.linspace(-3, 3, 50, endpoint=True)
test_y = np.sin(test_x)

plt.plot(data[0], data[1])
plt.plot(test_x, test_y)
plt.show()

n = 2
f5     = hermit(x[:(n+1)], y[:(n+1)], y_prime[:(n+1)])
data   = f5.linspace(n=50, domain=[-0.7, 3])
test_x = np.linspace(-2*pi, 2*pi, 50, endpoint=True)
test_y = np.sin(test_x)

plt.plot(test_x, test_y)
plt.plot(data[0], data[1])
plt.show()


n = 3
f7     = hermit(x[:(n+1)], y[:(n+1)], y_prime[:(n+1)])
data   = f7.linspace(n=50, domain=[-0.3, 3])
test_x = np.linspace(-2*pi, 2*pi, 50, endpoint=True)
test_y = np.sin(test_x)

plt.plot(data[0], data[1])
plt.plot(test_x, test_y)
plt.show()

#defining new set of given node information: x, y and y'
x       = np.array([0.0, 1.0,          2.0       ])
y       = np.array([1.0, 2.71828183,  54.59815003])
y_prime = np.array([0.0, 5.43656366, 218.39260013])


f7      = hermit( x, y, y_prime)
data    = f7.linspace(n=50, domain=[-0.5, 2.2])
test_x  = np.linspace(-0.5, 2.2, 50, endpoint=True)
test_y  = np.exp(test_x**2)

plt.plot(data[0], data[1])
plt.plot(test_x, test_y)
plt.show()

#defining new set of given node information: x, y and y'
x       = np.array([1.0, 3.0, 5.0])
y       = np.array([5.0, 1.0, 5.0])
y_prime = np.array([-4.0, 0.0, 4.0])

f7      = hermit( x, y, y_prime)
data    = f7.linspace(n=50, domain=[-10, 10])
test_x  = np.linspace(-10, 10, 50, endpoint=True)
test_y  = (test_x-3)**2 + 1

plt.plot(data[0], data[1])
plt.plot(test_x, test_y)
plt.show()

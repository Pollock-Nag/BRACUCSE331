"""
@Author-anonymousP
@__WHEN YOU FEEL LIKE QUITTING, THINK ABOUT WHY YOU STARTED__@
"""

import numpy as np
import matplotlib.pyplot as plt

#__Polynimial Class BEGINS__#
class Polynomial:

    def __init__(self, coeff):
        self.coefficients = coeff
        #order = degree#
        self.order = len(coeff) - 1

    def __call__(self, x):
    # To make Object callable
    # Here we get value of THE POLYNOMIAL

        res = 0.0
        ## loop cholbe degree+1 porjonto
        for i in range(0, self.order + 1):
            a_i = self.coefficients[i]
            res += a_i * (x ** i)
        return res

    def __repr__(self):
    # toString_java

        res_str = \
            f"{self.order}th order Polynomial with coefficients - {self.coefficients}"
        return res_str

    def degree(self):
        return self.order

    def coeff(self):
        return self.coefficients

#__Polynimial Class ENDS__#

#*****************************************************#
def get_van_poly_Coeff(x, y):
# THIS METHOD FIND  VALUES OF COEFFICIENTS #

    # VA=F#
    # A=V^(-1)*F#

    length = len(x)
    X = np.zeros((length, length))
    for i in range(0, length):
        for j in range(0, length):
            X[i, j] = x[i] ** j     #(x1^2) i=1; j=2#

    X_inverse = np.linalg.pinv(X)  # pseudo inverse
    a = np.dot(X_inverse, y)
    p = Polynomial(a)

    return p
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#
#*****************************************************#
def get_lagrange_poly(x, y, X):
#This method calculate LAGRANGE polynomial#

    degree=len(x)-1
    y_interp=np.array([])
    for idx in range(len(X)):
        yp = 0
        for k in range(degree + 1):
            lk = 1
            # new_x = np.delete(x, k)
            for j in range(degree + 1):
                if (k != j):
                    lk *= ((X[idx] - x[j]) / (x[k] - x[j]))
            yp += y[k] * lk
        y_interp = np.append(y_interp, yp)
    return y_interp
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#

#*****************************************************#
##Tester Class For Vandermonde Polinomial MERTHOD##
x = np.array([-3., -2., -1., 0., 1., 3.])
y = np.array([-80., -13., 6., 1., 5., 16.])

# get_van_poly_Coeff(x, y) returus the value of Coefficients#
print(get_van_poly_Coeff(x, y))


xplt = np.linspace(-3, 3, 100)
#p is as Object of Polynomial Class#
p = get_van_poly_Coeff(x, y)
yplt =p(xplt)  #p(xplt) returns the value of polynomial#

#plot
plt.plot(xplt, yplt, 'k--')
plt.plot(x, y, 'bo')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend(['interpolated', 'node points'], loc = 'lower right')
plt.text(-4.3, 50.5, '@anonymousP', bbox=dict(facecolor='gray', alpha=0.5))
plt.title('General Polynomial')
plt.show()

#*****************************************************#
'''
##Tester Class For LAGRANGE MERTHOD##

x = np.array([-3., -2., -1., 0., 1., 3.])
y = np.array([-80., -13., 6., 1., 5., 16.])
xplt = np.linspace(-3, 3, 100)
yplt=get_lagrange_poly(x,y,xplt)

#plot
plt.plot(xplt, yplt, 'k--')
plt.plot(x, y, 'bo')
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.legend(['interpolated', 'node points'], loc = 'lower right')
plt.text(-4.3, 50.5, '@anonymousP', bbox=dict(facecolor='gray', alpha=0.5))
plt.title('Lagrange Polynomial')
plt.show()
'''
"""
@Author-anonymousP
@__WHEN YOU FEEL LIKE QUITTING, THINK ABOUT WHY YOU STARTED__@
"""


import numpy as np
import matplotlib.pyplot as plt


class Newtons_Divided_Differences:
    def __init__(self, differences):
        self.differences = differences

    def __call__(self, x):
        """
        this function is for calculating y from given x using all the difference coefficients
        x can be a single value or a numpy
        the formula being used:
        f(x) = f [x0] + (x-x0) f[x0,x1] + (x-x0) (x-x1) f[x0,x1,x2] + . . . + (x-x0) (x-x1) . . . (x-xk-1) f[x0, x1, . . ., xk]

        work on this after implementing 'calc_div_diff'. Then you should have
        f[x0], f[x0,x1]. . . . . ., f[x0, x1, . . ., xk] stored in self.differences

        'res' variable must return all the results (corresponding y for x)
        """
        #debug
        #print("This is data_x", data_x)
        #print("This is x", x)
        #print("This is coeff", self.differences )

        res = np.zeros(len(x))
        degree = len(data_x)-1
        res = self.differences[degree]
        for i in range (1,degree+1):
            res= self.differences[degree-i]+(x-data_x[degree-i])*res

        return res


# basic rule for calculating the difference, implanted in the lambda function.
# You may use it if you wish
difference = lambda y2, y1, x2, x1: (y2 - y1) / (x2 - x1)


def calc_div_diff(x, y):
    assert (len(x) == len(y))
    # write this function to calculate all the divided differences in the list 'b'
    b = []
    n = len(x)
    #print(n)
    for i in range(n):
        b.append(y[i])
    #print(b)
    for i in range(1, n):
        for j in range(n - 1, i - 1,-1):
            y1 = b[j]
            y0 = b[j - 1]
            x1 = x[j]
            x0 = x[j-i]
            b[j]=((y1-y0)/(x1-x0))

    #print(b)
    return b


data_x = np.array([-3.,-2.,-1.,0.,1.,3.,4.])
data_y = np.array([-60.,-80.,6.,1.,45.,30.,16.])

differences = calc_div_diff(list(data_x), list(data_y))
obj = Newtons_Divided_Differences(list(differences))

#generating 50 points from -3 to 4 in order to create a smooth line
X = np.linspace(-3, 4, 50, endpoint=True)
F = obj(X)
#(debug)
#print("This is data_x", data_x)
#print("This is x of p(x)", X)
#print("This is coeff", differences )

plt.plot(X,F,'k--')
plt.plot(data_x, data_y, 'bo')
plt.title('Newton Polynomial')
plt.show()
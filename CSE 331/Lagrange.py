"""
@author AnonymousP
@__WHEN YOU FEEL LIKE QUITTING,
THINK ABOUT WHY YOU STARTED__@
"""

import numpy as np
import matplotlib.pyplot as plt


class LagrangePolynomial:
    def __init__(self, data_x, data_y):
        """
        First we need to check whether the input vectors (numpy arrays) are equal
        or not.
        assert (condition), "msg"
        this command checks if the condition is true or false. If true, the code
        runs normally. But if false, then the code returns an error message "msg"
        and stops execution
        """
        assert len(data_x) == len(data_y), "length of data_x and data_y must be equal"

        '''
        Since lagrange polynomials do not use coefficeints a_i, rather the nodes 
        (x_i, y_i), we just need to store these inside the object
        '''

        self.x = data_x
        self.y = data_y

        self.degree = len(data_x) - 1

    @property
    def __repr__(self):
        strL = f"LagrangePolynomial of order {self.degree}\n"
        strL += "p(x) = "
        for i in range(self.y):
            if self.y == 0:
                continue
            elif self.y >= 0:
                strL += f"+ {self.y}*l_{i + 1}(x) "
            else:
                strL += f"- {-self.y}*l_{i + 1}(x) "

        return strL

    def __call__(self, x):
        """
        The method to make the object callable (see the code of the matrix method).
        """

        '''
        #method-->1
        y_interp = np.array([])


        for idx in range (len(x)):
            yp = 0
            for k in range(self.degree+1):
                lk=1
                for j in range(self.degree+1):
                #new_x = np.delete(self.x, j)
                    if(k !=j):
                        lk*=  ((x[idx] - self.x[j])/(self.x[k]-self.x[j]))
                yp +=self.y[k]*lk
            y_interp=np.append(y_interp,yp)



        return y_interp
        '''
        """
        # method-->2

        y_interp = np.array([])
        for idx in range(len(x)):
            res = 0
            for k in range(self.degree+1):
                lk = 1
                new_x = np.delete(self.x, k)
                for j in range(len(new_x)):
                    lk *= ((x[idx] - new_x[j]) / (self.x[k] - new_x[j]))
                res += self.y[k]*lk
            y_interp = np.append(y_interp, res)

        return y_interp
            """
        """
    # method-->3
        y_interp = np.array([])
        res = 0
        for k in range(self.degree+1):
            lk = 1
            new_x = np.delete(self.x, k)
            for j in range(len(new_x)):
                lk *= ((x - new_x[j]) / (self.x[k] - new_x[j]))
            res += self.y[k]*lk
        y_interp = np.append(y_interp, res)

        return y_interp
        """
    # method-->4
        y_interp = np.array([])
        for idx in x:
            res = 0
            for k , j in zip(self.x,self.y):
                res +=j*np.prod ((idx - self.x[self.x != k]) / (k - self.x[self.x != k]))
            y_interp = np.append(y_interp, res)
        return y_interp


##TESTER

##print(np.long(0.3-(3*0.1)))
data_x = np.array([-3.,-2.,-1.,0.,1.,3.,4.])
data_y = np.array([-60.,-80.,6.,1.,45.,30.,16.])

#data_x = np.array([-3., -2., -1., 0., 1., 3.])
#data_y = np.array([-80., -13., 6., 1., 5., 16.])

p = LagrangePolynomial(data_x,data_y)

#generating 100 points from -3 to 4 in order to create a smooth line
#x = np.linspace(-3, 3, 100)
x = np.linspace(-3, 4, 100)
y_interp = p(x)

# plot to see if your implementation is correct
#google the functions to understand what each parameters mean, if not apparent

plt.plot(x, y_interp, 'k--')
plt.plot(data_x, data_y, 'bo')
plt.legend(['interpolated', 'node points',], loc = 'lower right')
plt.xlabel('x')
plt.ylabel('y')
plt.text(-4.5, 155.5, '@anonymousP', bbox=dict(facecolor='gray', alpha=0.5))

plt.title('Lagrange Polynomial')
plt.show()

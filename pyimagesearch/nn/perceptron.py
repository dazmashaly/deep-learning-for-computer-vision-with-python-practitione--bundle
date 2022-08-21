import imp


import numpy as np

class Perceptron:
    def __init__(self,n,alpha =0.1):
        #initialize the weight matrix and store the learning rate
        self.w = np.random.randn(n+1)/np.sqrt(n)
        self.alpha = alpha

    def step(self,x):
        return 1 if x > 0 else 0

    def fit(self,X,y,epochs = 10):
        # insert a column of ones as the last entry
        X=np.c_[X,np.ones((X.shape[0]))]

        for epoch in np.arange(0,epochs):
            #loop over every data point
            for (x,target) in zip(X,y):
                # take the dot product and pass the value to step function
                p = self.step(np.dot(x,self.w))
                # update weight only if prediction is wrong
                if p != target:
                    error = p-target
                   
                    self.w += -self.alpha * error * x
                
                         

    def predict(self,x,addbias =True):
        x = np.atleast_2d(x)
        if addbias:
            x = np.c_[x,np.ones((x.shape[0]))]
        return self.step(np.dot(x,self.w))

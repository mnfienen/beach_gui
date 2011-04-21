#PLS1 algorithm from Wikipedia
import numpy as np
import model
from numpy import matrix
import copy
from matplotlib import mlab



    


class PLS(model.Model):

    def __init__(self, data, target):
        #Instantiate the base class
        model.Model.__init__(self, copy.copy(data), target.lower())

        #Pop the response from the data dictionary
        Y = self.data.pop(self.target)    

        #Get the predictors together
        columns = self.columns = self.data.keys()
        X = np.array(self.data.values()).T
        
        #Put the predictors and response into the matrix representation expected by the PLS1 algorithm
        self.X = matrix(X)
        self.Y = matrix(Y).T

        #Algorithm for partial least squares
        self.PLS1(X,Y)        


    def PLS1(self, X, Y):
        X = matrix(X)
        Y = matrix(Y).T
        
        B = list()
        B0 = list()

        W = matrix( [ [] for i in range(X.shape[1]) ] )
        P = matrix( [ [] for i in range(X.shape[1]) ] )
        q = matrix( [ [] for i in range(Y.shape[1]) ] )
        
        finished = False
        k = 0
    
        T = X*X.T*Y/np.linalg.norm(X.T*Y)
        
        while not finished:
            t = (T.T * T)[0,0]
            T = T/t
            P_k = X.T * T
            q_k = (Y.T * T)           

            if q_k.all==0 or k>=X.shape[1]:
                finished=True
             
            else:
                P = np.hstack( (P,P_k) )
                q = np.hstack( (q,q_k) )            

                X = X - t*T*P_k.T
                W_k = X.T*Y
                T = X*W_k
            
                W = np.hstack( (W,W_k) )
    
                B.append( W*(P.T*W).I*q.T )
                B0.append( q_k - (P_k.T * B[k]) )

            k = k+1

        if k>1:
            self.beta = B
            self.beta0 = B0
            B0 = np.atleast_2d( np.array(B0).squeeze() ).T
            B = np.array(B).squeeze().T
            #self.coefficients = np.matrix(np.hstack( (B0,B) ))

    def Normalize(self, x):
        sigma = np.std(x)
        mu = np.mean(x)

        x = (x-mu)/sigma
        return [x, mu, sigma]







def pls_algorithm(x, y):
    X = list()
    for i in range( x.shape[1] ):
        X.append( Normalize(x[:,i]) )

    X = [np.array(X).T]
    Y = [np.array(y).T]
    y_hat = [mean(y)]
    Z = list()
    phi = list()
    theta = list()

    for j in range( x.shape(1) ):
        phi.append( dot(X[j].T, Y[0]) )
        Z.append( phi[j]*X[j] )

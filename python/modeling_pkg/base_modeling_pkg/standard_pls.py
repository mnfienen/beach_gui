from numpy import inner, sqrt, array, matrix, linalg, dot, concatenate, mean
import copy

def pls(X,y):
    X = copy.copy(X)
    X_original = copy.copy(X)

    Y = copy.copy(y)
    Y_mean = mean(Y)
    Y_centered = Y-Y_mean
    
    intercept = matrix([Y_mean for i in Y]).T

    # Standard PLS algorithm for single Y-variable.
    
    for i in range(X.shape[1]):
        w = (Y_centered.T * X).T
        w = w/linalg.norm(w)
        t = X * w

        if i > 1:
            t = t - T*((T.T*T).I * (t.T*T).T)

        u = t/(linalg.norm(t)**2)
        p = (u.T * X).T
        X = X - t*p.T

        if i>1:
            T = concatenate((T,t),1)
            P = concatenate((P,p),1)
            W = concatenate((W,w),1)
            q = concatenate((q,y.T*u))
            B = concatenate((B,W*(P.T*W).I*q),1)

        else:
            T = t
            P = p
            W = w
            q = Y.T*u
            B = W*(P.T*W).I*q


    fitted = concatenate((intercept, X_original*B+Y_mean),1)
    residuals = Y-fitted

    return {'coefs':B, 'score':T, 'X-loading':P, 'weight':W, 'Y-loading':q, 'fitted':fitted, 'residuals':residuals}
from numpy import inner, sqrt, array, matrix, linalg, dot, concatenate, mean
import copy

def simpls(X, y):
    X = copy.copy(X)
    Y = copy.copy(y)
    Y_mean = mean(Y)
    Y_centered = Y-Y_mean

    # SIMPLS algorithm for single Y-variable.
    s = (Y_centered.T * X).T
    
    intercept = matrix([Y_mean for i in Y]).T

    for i in range(X.shape[1]):
        r = copy.copy(s)
        t = X * r

        if i > 1:
            t = t - T*((t.T * T).T)

        normt = sqrt(t.T*t)
        t = t/normt
        r = r/normt
        p = (t.T * X).T
        v = p

        if i > 1:
            v = p - V * (p.T * V).T

        v = v/linalg.norm(v)
        s = s - v*(v.T * s)
        
        if i>1:
            V = concatenate((V,v),1)
            T = concatenate((T,t),1)
            P = concatenate((P,p),1)
            R = concatenate((R,r),1)
            q = concatenate((q,Y_centered.T*t))
            B = concatenate((B,R*q),1)
        else:
            V = v
            T = t
            P = p
            R = r
            q = Y_centered.T*t
            B = R*q

    
    fitted = concatenate((intercept, X*B+Y_mean),1)
    residuals = Y-fitted

    return {'coefs':B, 'score':T, 'X-loading':P, 'weight':R, 'Y-loading':q, 'fitted':fitted, 'residuals':residuals}
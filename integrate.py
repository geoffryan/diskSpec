import numpy as np

def trap(xa, xb, n, f, *args):
    x = np.linspace(xa, xb, num=n+1, endpoint=True)
    dx = (xb-xa) / n
    y = f(x, *args)

    return (y[0] + 2*y[1:-1].sum() + y[-1])*dx/2.0

def simp(xa, xb, n, f, *args):
    N = n
    if N < 2:
        N = 2
    if n%2 != 0:
        N = n+1
    x = np.linspace(xa, xb, num=N+1, endpoint=True)
    dx = (xb-xa) / N
    y = f(x, *args)

    return (y[0] + 4*y[1:-1:2].sum() + 2*y[2:-2:2].sum() + y[-1])*dx/3.0

def romb(xa, xb, n, atol, rtol, f, *args):

    if atol < 0.0:
        atol = -atol
    if rtol < 0.0:
        rtol = -rtol

    K = int(np.log2(n)) + 1
    R = np.zeros(K, dtype=np.float)

    N = 1
    h = 0.5*(xb-xa) 
    R[0] = h*(f(xa, *args) + f(xb, *args))

    for k in range(1,K):

        R = np.roll(R, 1)

        x = np.linspace(xa+h, xb-h, num=N, endpoint=True)
        R[0] = 0.5*R[1] + h*f(x,*args).sum()

        fpm = 1
        err = np.inf
        for m in range(1,k+1):
            fpm *= 4
            err = (R[m-1] - R[m]) / (fpm - 1)
            R[m] = (fpm*R[m-1] - R[m]) / (fpm - 1)
        
        if k < K-1:
            if np.fabs(err) < atol+rtol*np.fabs(R[k]):
                R[-1] = R[k]
                break
            else:
                R[-2] = R[k]
                N *= 2
                h *= 0.5

    return R[-1]



if __name__ == "__main__":

    def f(x, p1, p2):
        return np.exp(p1*x)+p2

    a = -1.0
    b =  1.0

    p1 = 3.0
    p2 = -1.0

    NN = 10
    
    exact = (np.exp(p1*b) - np.exp(p1*a))/p1 + p2*(b-a)

    N = [int(np.power(2,k)) for k in range(1,NN+1)]

    errT = []
    errS = []
    errR = []

    for n in N:
        print(n)
        iT = trap(a, b, n, f, p1, p2)
        print("trap: {0:.6f} ({1:.6f})".format(iT, exact))
        iS = simp(a, b, n, f, p1, p2)
        print("simp: {0:.6f} ({1:.6f})".format(iS, exact))
        iR = romb(a, b, n, 0.0, 1.0e-10, f, p1, p2)
        print("romb: {0:.6f} ({1:.6f})".format(iR, exact))

        errT.append(abs((iT-exact)/exact))
        errS.append(abs((iS-exact)/exact))
        errR.append(abs((iR-exact)/exact))

    N = np.array(N).astype(float)
    errT = np.array(errT)
    errS = np.array(errS)
    errR = np.array(errR)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1)

    ax.plot(N, errT, 'k+')
    ax.plot(N, errS, 'b+')
    ax.plot(N, errR, 'r+')
    ax.plot(N, errT[0]*np.power(N/N[0],-2), color='k', ls='-') 
    ax.plot(N, errS[0]*np.power(N/N[0],-4), color='b', ls='-') 
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$L_1$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.show()


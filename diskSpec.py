import numpy as np
import integrate as ig
import matplotlib.pyplot as plt

c = 2.99792458e10
h = 6.626070040e-27
eV = 1.6021766208e-12
pc = 3.08567758149e18
RSsolar = 2.95325024e5
Msolar = 1.98848e33
yr = 3.15569252e7
Jy = 1.0e-23

kB = 1.38064852e-16
hbar = h/(2*np.pi)
sb = np.pi*np.pi / (60*hbar*hbar*hbar*c*c)
sb_K = sb * kB*kB*kB*kB

GMsolar = 0.5*RSsolar*c*c
Rgsolar = 0.5*RSsolar
c2 = c*c

Zcut = 1.0e-2

def disk1Dfunc_nobl(x, a):
    X = np.atleast_1d(x)
    y = 0.5*a*X

    # e^(2y)-1 = 2 e^y sinh(y)
    # The second form is more accurate for small y and has
    # same accuracy for large y.

    f = np.power(X,5.0/3.0) / (2*np.exp(y)*np.sinh(y))

    return f

def disk1Dfunc_bl(x, a):
    X = np.atleast_1d(x)
    fs = 1.0 - np.power(X, -2.0/3.0)
    g = np.empty(X.shape)
    g[fs>0] = np.power(fs[fs>0], -0.25)
    g[fs<=0] = np.inf
    y = 0.5*a*X*g

    # e^(2y)-1 = 2 e^y sinh(y)
    # The second form is more accurate for small y and has
    # same accuracy for large y.

    f = np.power(X,5.0/3.0) / (2*np.exp(y)*np.sinh(y))

    return f

def disk1Dfunc_bl_zoom(z, a):
    # This is for accurate integration of disk1Dfunc_bl near x=1.
    # A change of variables to z = 1 - x^{-2/3} has been performed,
    # so the integrand gets multiplied by 1.5 x^{5/3}.

    Z = np.atleast_1d(z)
    X = np.power(1-Z, -1.5)
    g = np.empty(X.shape)
    g[Z>0] = np.power(Z[Z>0], -0.25)
    g[Z<=0] = np.inf
    y = 0.5*a*X*g

    # e^(2y)-1 = 2 e^y sinh(y)
    # The second form is more accurate for small y and has
    # same accuracy for large y.

    f = 1.5*np.power(X,10.0/3.0) / (2*np.exp(y)*np.sinh(y))

    return f

def disk1Dfunc_bl_zoom_log(u, a):
    # This is for accurate integration of disk1Dfunc_bl near x=1.
    # A change of variables to u = log(z) has been performed,
    # so the integrand gets multiplied by 1.5 z x^{5/3}.

    U = np.atleast_1d(u)
    Z = np.exp(U)
    X = np.power(1-Z, -1.5)
    g = np.empty(X.shape)
    g[Z>0] = np.power(Z[Z>0], -0.25)
    g[Z<=0] = np.inf
    y = 0.5*a*X*g

    # e^(2y)-1 = 2 e^y sinh(y)
    # The second form is more accurate for small y and has
    # same accuracy for large y.

    f = 1.5*Z*np.power(X,10.0/3.0) / (2*np.exp(y)*np.sinh(y))

    return f

def diskBB(nu, GM, Mdot, r1, r2, D, inc, bl=True,
            n=10000, atol=0.0, rtol=1.0e-10, bruteforce=False):


    nu = np.atleast_1d(nu)

    Ts = np.power(3*GM*Mdot / (8*np.pi*r1*r1*r1*sb), 0.25)

    Fnu = np.empty(nu.shape)

    x1 = 1.0
    x2 = np.power(r2/r1, 0.75)

    for i,v in enumerate(nu):
        if bl:
            if bruteforce:
                Fnu[i] = ig.romb(x1, x2, n, atol, rtol, disk1Dfunc_bl, h*v/Ts)
            else:
                a = h*v/Ts
                zs1 = Zcut*a*a*a*a
                zs2 = Zcut
                z1 = 0.0
                xs = np.power(1-zs2, -1.5)
                if xs > x2:
                    xs = x2
                    zs2 = 1.0 - np.power(xs,-2.0/3.0)
                if a < 0.5:
                    us1 = np.log(zs1)
                    us2 = np.log(zs2)
                    F1 = ig.romb(z1, zs1, n, atol, rtol, disk1Dfunc_bl_zoom, a)
                    F2 = ig.romb(us1, us2, n, atol, rtol, disk1Dfunc_bl_zoom_log, a)
                else:
                    F1 = ig.romb(z1, zs2, n, atol, rtol, disk1Dfunc_bl_zoom, a)
                    F2 = 0.0
                F3 = ig.romb(xs, x2, n, atol, rtol, disk1Dfunc_bl, a)
                #print(F1, F2, F3)
                Fnu[i] = F1 + F2 + F3
        else:
            Fnu[i] = ig.romb(x1, x2, n, atol, rtol, disk1Dfunc_nobl, h*v/Ts)

    C = 16*np.pi*h*r1*r1*np.cos(np.pi*inc/180.0) / (3.0 *c*c * D*D)

    Fnu *= C * nu*nu*nu

    return Fnu

def convergence_test():

    GM = 10.0 * GMsolar
    r1 = 2*GM/(c*c)
    r2 = 10.0 * r1
    Mdot = 8*np.pi*r1*r1*r1*sb / (3*GM) * h*h*h*h
    D = np.sqrt(16*np.pi*h/c) * r1

    nu = np.logspace(-4.0, 1.0, 100)
    Fnu = diskBB(nu, GM, Mdot, r1, r2, D, 0.0, n=1000)

    fig, ax = plt.subplots(1,1)
    ax.plot(nu, Fnu, 'k-')
    ax.set_xlabel(r"$\nu$ (Hz)")
    ax.set_ylabel(r"$F_\nu$ (erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    NN = 20
    NNU = 4
    NU = np.logspace(-5.0, 2.0, NNU)
    N = np.power(2, np.arange(NN)+1)
    
    Ts = np.power(3*GM*Mdot / (8*np.pi*r1*r1*r1*sb), 0.25)
    x1 = 1.0
    x2 = np.power(r2/r1, 0.75)
    x = np.linspace(x1, x2, 1000)
    
    FNU = np.empty((NN,NNU))
    for i,n in enumerate(N):
        print(i,n)
        FNU[i,:] = diskBB(NU, GM, Mdot, r1, r2, D, 0.0, n=n, atol=0.0, rtol=0.0, bl=True)

    ERR = (FNU[:-1,:] - FNU[-1,:]) / FNU[-1,:]

    fig, ax = plt.subplots(1,1)
    ax.plot(N[:-1], np.fabs(ERR[:,0]), 'r')
    ax.plot(N[:-1], np.fabs(ERR[:,1]), 'y')
    ax.plot(N[:-1], np.fabs(ERR[:,2]), 'g')
    ax.plot(N[:-1], np.fabs(ERR[:,3]), 'b')
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$L_1$")
    ax.set_xscale("log")
    ax.set_yscale("log")

    Fnu = diskBB(nu[::3], GM, Mdot, r1, r2, D, 0.0, n=10000, rtol=0.0)
    Fnu_bf = diskBB(nu[::3], GM, Mdot, r1, r2, D, 0.0, n=100000, rtol=0.0, bruteforce=True)

    fig, ax = plt.subplots(2,1)
    ax[0].plot(nu[::3], Fnu_bf, 'k')
    ax[0].plot(nu[::3], Fnu, 'b')
    ax[1].plot(nu[::3], np.fabs((Fnu-Fnu_bf)/Fnu_bf), 'k')
    ax[1].set_xlabel(r"$\nu$")
    ax[0].set_ylabel(r"$F_\nu$")
    ax[1].set_ylabel(r"$\Delta F_\nu$")
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    plt.show()

if __name__ == "__main__":

    GM = 10.0 * GMsolar
    Mdot = 1.0e-8 * Msolar / yr
    r1 = 6*GM/c2
    r2 = 10.0 * r1
    D = 1.0e5 * GM/c2
    inc = 60.0

    nu = np.logspace(15, 18, 100)
    Fnu = diskBB(nu, GM, Mdot, r1, r2, D, inc)

    fig, ax = plt.subplots(1,1)
    ax.plot(nu, Fnu, 'k-')
    ax.set_xlabel(r"$\nu$ (Hz)")
    ax.set_ylabel(r"$F_\nu$ (erg cm$^{-2}$ s$^{-1}$ Hz$^{-1}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    print(sb_K)

    fig.savefig("spec.pdf")

    plt.show()


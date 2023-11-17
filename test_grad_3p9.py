import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

## 2D case with parameterization
#r = 2
#W = np.random.rand(3,r)
#WtW = W.T@W

#N = 1000
#x = np.linspace(0.1,1.9,N)

#f = 2*WtW[1,0]*(1/x + 1/(2-x))

#plt.plot(x,f)
#plt.show()

## Monte Carlo sampling of the landscape#W = np.random.rand(2,2)
r = 2
W = np.random.randn(100,r)
W[W<0]=0
WtW = W.T@W

N = 1000
u = np.random.rand(r,N)

f = np.array([np.sum((WtW@u[:,i])/u[:,i]) for i in range(N)])
f1 = np.sum(WtW)

print(f"Value at 1:{f1}, min value on Monte Carlo:{np.min(f)} at {u[:,np.argmin(f)]}")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(u[0,:], u[1,:], np.log(1+f), c=f)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D contour')

## Contour plot
#fig2, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
#WtW = np.random.rand(2,2) # not symmetric !
#WtW[1,0]=0.01
# symmetric
#W = np.random.rand(2,2)
#WtW = W.T@W
#WtW[0,1]=0
#WtW[1,0]=0
#X = np.arange(1e-6, 10, 0.5)
#Y = np.arange(1e-6, 10, 0.5)
#M = len(X)
#f = np.array([[np.sum((WtW@np.array([X[i],Y[j]]))/np.array([X[i],Y[j]])) for i in range(M)] for j in range(M)])
#X, Y = np.meshgrid(X, Y)
## Plot the surface.

#surf = ax.plot_surface(X, Y, np.log(1+f), cmap=cm.coolwarm,
                       #linewidth=0, antialiased=False)

## Add a color bar which maps values to colors.
#fig2.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()

## SDP Hessian? dim 3
N = 100
r = 2
W = np.random.rand(5,r)
WtW = W.T@W
for i in range(N):
    H = np.zeros([r,r])
    x = np.random.rand(r)
    x = x/np.sum(x)
    for k in range(r):
        for l in range(r):
            if k==l:
                H[k,k] = (2/x[k]**2)*(WtW[k,:]@x/x[k] - WtW[k,k])
            else:
                H[k,l] = WtW[k,l]*(-1/x[k]**2 - 1/x[l]**2)
    
    eigvals = np.linalg.eigvals(H)
    #print(eigvals)
    if np.min(eigvals)<0:
        print(eigvals,x,H)
        break


plt.show()

# Testing with pytorch?
import mpctools as mpc
import matplotlib.pyplot as plt
import numpy as np

# %% Paramteters from Shadden Physica D
A = .5
eps = .25
omega = 2*np.pi
t0 = 0.8

path = np.array([[1. ,  1.04 ,1.04 ,1.05 ,0.99 ,0.93, 0.96, 1.02, 1.09, 1.05, 1.06, 1.06, 1.06, 1.06,\
    1.15, 1.12, 1.19, 1.13, 1.17, 1.09, 1], [0.8,  0.71, 0.61, 0.55, 0.47, 0.39, 0.46, 0.54, 0.59, 0.5,  0.56, \
    0.66, 0.56, 0.66, 0.69, 0.62, 0.69, 0.63, 0.71, 0.77, 0.8]])
print(path.shape,np.linspace(0,2,21).shape)
path = np.vstack([path,np.linspace(0,2,21)+t0])
print(path.shape)

import casadi

# this is a workaround for issue #2743:
# it shoud be obsolete for casadi version >= 3.6

# source of this function https://github.com/jcotem/casadi/commit/08c5f39156140817200e207b7696f59223458ce0#diff-3dc05d8ab5896084984b420dd6495df43d2e19e5f7e64693cd516bc2da0dfbf7
# referenced here https://github.com/casadi/casadi/issues/2743

def __array__(self,*args,**kwargs):
    import numpy as n
    if len(args) > 1 and isinstance(args[1],tuple) and isinstance(args[1][0],n.ufunc) and isinstance(args[1][0],n.ufunc) and len(args[1])>1 and args[1][0].nin==len(args[1][1]):
      if len(args[1][1])==3:
        raise Exception("Error with %s. Looks like you are using an assignment operator, such as 'a+=b'. This is not supported when 'a' is a numpy type, and cannot be supported without changing numpy itself. Either upgrade a to a CasADi type first, or use 'a = a + b'. " % args[1][0].__name__)
      return n.array([n.nan])
    else:
      if hasattr(self,'__array_custom__'):
        return self.__array_custom__(*args,**kwargs)
      else:
        try:
          return self.full()
        except:
          if self.is_scalar(True):
            # Needed for #2743
            E=n.empty((),dtype=object)
            E[()] = self
            return E
          else:
            raise Exception("!!Implicit conversion of symbolic CasADi type to numeric matrix not supported.\n"
                      + "This may occur when you pass a CasADi object to a numpy function.\n"
                      + "Use an equivalent CasADi function instead of that numpy function.")


# monkey-patch the casadi datatype

casadi.casadi.SX.__array__ = __array__


# def f(x,t):
#     return eps*np.sin(omega*t)*x**2 + x - 2*eps*np.sin(omega*t)*x
# def df(x,t):
#     return 2*eps*np.sin(omega*t)*x + 1 - 2*eps*np.sin(omega*t)
# def ddf(x,t):
#     return 2*eps*np.sin(omega*t)
# def U(x,y,t):
#     return -np.pi*A*np.sin(np.pi*f(x,t))*np.cos(np.pi*y)
# def V(x,y,t):
#     return np.pi*A*np.cos(np.pi*f(x,t))*np.sin(np.pi*y)*df(x,t)
# def _get_v(t,loc):
#     x,y = loc
#     u = U(x,y,t) #-np.pi*A*np.sin(np.pi*f(x,t))*np.cos(np.pi*y)
#     v = V(x,y,t) #np.pi*A*np.cos(np.pi*f(x,t))*np.sin(np.pi*y)*df(x,t)
#     return np.array([u,v])
# def curl(x,y,t):
#     return -np.pi**2*A*np.sin(np.pi*f(x,t))*np.sin(np.pi*y)*(1+df(x,t)**2) + \
#         np.pi*A*np.cos(np.pi*f(x,t))*np.sin(np.pi*y)*ddf(x,t)

# def get_data(t):
#     X,Y = np.meshgrid(np.linspace(0,x,width),np.linspace(0,y,height))
#     return curl(X,Y,t)

# %% Define model and parameters.
Delta = .01
ubotmax = 2.5
Nt = 10 # it was 15
Nx = 3
Nu = 2

# Define stage cost and terminal weight.
Q1 = 1
Q = np.eye(Nx)*Q1
Q[2,2] = 0
Q2 = 0
R1 = 1e-4
R = R1*np.eye(Nu)

def dgyre(x, u, A = A, eps = eps, om = omega):
    """Continuous-time ODE model."""
    
    a = eps * np.sin(om * x[2])
    b = 1 - 2 * a
    
    f = a * x[0]**2 + b * x[0]
    df = 2 * a * x[0] + b
    
    dxdt = [
        -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * x[1]) + u[0],
        np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * x[1]) * df + u[1],
        1,
    ]
    return np.array(dxdt)

# Create a simulator. This allows us to simulate a nonlinear plant.
ocean = mpc.DiscreteSimulator(dgyre, Delta, [Nx,Nu], ["x","u"])

# Then get casadi function for rk4 discretization.
ode_rk4_casadi = mpc.getCasadiFunc(dgyre, [Nx,Nu], ["x","u"], funcname="F",
    rk4=True, Delta=Delta, M=1)


def lfunc(x,u,goal):
    """Standard quadratic stage cost."""
    return mpc.mtimes((x-goal).T, Q, (x-goal)) + mpc.mtimes(u.T, R, u)

def Pffunc(x,goal):
    return Q2*mpc.mtimes((x-goal).T,(x-goal))

# Bounds on u. Here, they are all [-1, 1]
lb = {"u" : -ubotmax*np.ones((Nu,))}
ub = {"u" : ubotmax*np.ones((Nu,))}

# Make optimizers. 
N = {"x":Nx, "u":Nu, "t":Nt}

# %% Now simulate.
Nsim = 10
traj = [path[:,0]]
energy = []
xi = path[:,0]

for i in range(path.shape[1]-1):
    # xi = path[:,i]
    # traj.append(xi)
    goal = path[:,i+1]
    Pf = mpc.getCasadiFunc(lambda x: Pffunc(x,goal), [Nx], ["x"], funcname="Pf") 
    l = mpc.getCasadiFunc(lambda x,u: lfunc(x,u,goal), [Nx,Nu], ["x","u"], funcname="l")
    solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, Pf = Pf, x0=xi, lb=lb, ub=ub,verbosity=0)

    u = np.zeros((Nsim,Nu))
    pred = []
    upred = []
    # Fix initial state.
    #solver.fixvar("x", 0, x[0,:])  
    for t in range(Nsim):
        #solver.fixvar("x", 0, x[t,:]) 
        # Solve nlp.
        solver.solve()   
        
        # Print stats.
        # print("%d: %s" % (t,solver.stats["status"]))
        # solver.saveguess()
        solver.fixvar("x",0,solver.var["x",1])
        
        u[t,:] = np.array(solver.var["u",0,:]).flatten() 
        # calling solver variables. Can pull out predicted trajectories from here.
        pred += [solver.var["x",:,:]]
        upred += [solver.var["u",:,:]]
        # pred += [np.vstack([np.array(solver.var["x",:,:]),np.zeros((t,3,1))])]
        # upred += [np.vstack([np.array(solver.var["u",:,:]),np.zeros((t,2,1))])]

        # xi = np.array(solver.var["x",1]).flatten()
        # solver = mpc.nmpc(f=ode_rk4_casadi, N={"x":Nx, "u":Nu, "t":Nt}, l=l, Pf = Pf, x0=xi, lb=lb, ub=ub,verbosity=0)

    pred = np.array(pred)[:,:,:,0]
    # print(pred)
    upred = np.array(upred)[:,:,:,0]
    # print(upred)
    traj.append(pred[:,1,:])
    energy.append(np.sum(upred[:,0,:]**2)*Delta)
    print('-'*20)
    print('error:',np.linalg.norm(pred[-1,1,:]-goal)/np.linalg.norm(path[:,i]-goal))
    print('energy:',np.sum(upred[:,0,:]**2)*Delta)
    print(xi,solver.var["x",1],goal)
    # print(upred[:,0,:])
    # print(pred[:,1,:])
    xi = np.array(solver.var["x",1]).flatten()

traj = np.vstack(traj)
print(traj.shape)
energy = np.array(energy)
print(energy.shape)

np.savez('double_gyre_no_flow_path_{}'.format(t0),traj=traj,energy=energy)
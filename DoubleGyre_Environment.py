import copy
import random
from collections import namedtuple
import gym
from gym import wrappers
import numpy as np
import matplotlib as mpl
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot
from random import randint
from pydmd import DMD
from gym import spaces
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import matplotlib.pyplot as plt

class DoubleGyre_Environment(gym.Env):
    '''
    One mobile sensor path planning game on a sparse random dynamics on a torus. 
    '''
    environment_name = "DoubleGyre"

    def __init__(self, seed=None, lamb=0.02, rho = 10, fix_init=None):
        '''
        size: size (height and width) of canvas
        dt: time step
        T: total time
        l: path periodicity
        '''
        self.A = 0.5
        self.omega = 2*np.pi
        self.eps = 0.25
        self.x, self.y = 2, 1 
        self.width = 201
        self.height = 101
        # self.T = 1
        self.dt = 0.1
        self.L = 20
        self.land_event = lambda t,y: y[0]*(y[0]-self.x)*y[1]*(y[1]-self.y)
        self.land_event.terminal = True
        self.collide_cost = 100

        optdmd = loadmat('../Datasets/DoubleGyreDMD.mat')
        dt_dmd = optdmd['dt'] # 0.005
        self.Psi = optdmd['w']
        self.Theta = np.diag(optdmd['eigs'].flatten()**20)
        self.n_modes = 10
        
        self.lamb = lamb # weight of movement cost
        self.rho = rho # return penalty weight
        self.fix_init = fix_init
        
        # intialization
        self.seed(seed)

        self.max_action=1
        self.action_space = spaces.Box(
            low=-self.max_action*np.ones(2, dtype=np.float32), high=self.max_action*np.ones(2, dtype=np.float32), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,-5,-5], dtype=np.float32), 
            high=np.array([2,1,1,5,5], dtype=np.float32)
        )

    def step(self, action, store_traj = False):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        reward = 0

        temp, collide = self._solve_ivp(action, store_traj)
        if collide:
            # print(self.s,action,temp,collide)
            reward -= self.collide_cost

        # time = self.init_t + self.l * self.dt
        # sol = solve_ivp(lambda t, y: self._get_v(t,y)+action, (time, time+self.dt), self.s, max_step = self.dt/10, events = self.land_event)
        # temp = sol.y[:,-1]
        # print(self.s,self._get_v(time,self.s), action, temp)
        # if store_traj:
        #     tostore = sol.y[:,1:] if sol.success else self.s.reshape(2,1)
        #     self.traj.append(tostore)
        if temp[0] < 0 or temp[0] > self.x or temp[1] < 0 or temp[1] > self.y:
            print(temp, self.land_event(0,temp))
            raise 
        self.s = temp # np.clip(temp, [0,0], [self.x,self.y])
        self.l += 1

        if self.l < self.L:
            ind = self._grid_to_ind(self._loc_to_grid(self.s))
            self.P[self.l] = ind
            self.Obs[:,self.l] = self.D[ind].T
            self.D = self.D @ self.Theta
            # gained information (log det observability) reward
            T = self.Obs[:,:self.l+1].conj() @ self.Obs[:,:self.l+1].T
            # T = self.Obs[:,:self.l+1].T @ self.Obs[:,:self.l+1].conj()
            new_logdet = np.linalg.slogdet(T)[1]
            reward += (new_logdet - self.logdet)
            self.logdet = new_logdet

        if self.l > 0: 
            # distance penalties
            dist1 = np.sum(action**2) # movement cost proportional to distance^2
            dist2 = np.sum((self.s-self.start)**2) # distance cost from inital location
            reward -= self.lamb * dist1
            reward -= self.rho * (dist2 - self.prev_dist_to_start)
            self.actions.append(action)            
            self.prev_dist_to_start = dist2

        self.done = (self.l == self.L)
        return self._get_obs(), reward, self.done, {}

    def reset(self, init_loc=None, init_t=None):
        """Resets the environment and returns the start state"""
        if self.fix_init is not None and init_loc is None:
            init_loc = self.fix_init
        if init_loc is not None:
            self.s =self._grid_to_loc(self._ind_to_grid(init_loc))
        else:
            init_loc = np.random.choice(self.width*self.height)
            self.s = self._grid_to_loc(self._ind_to_grid(init_loc))

        if init_t is not None:
            self.init_t = init_t
        else:
            self.init_t = np.random.choice(100)/100
            
        self.l = 0 
        self.start = self.s.copy()
        self.prev_dist_to_start = 0
        self.Obs = np.zeros((self.n_modes,self.L),dtype=complex)
        self.Obs[:,0] = self.Psi[init_loc].T
        self.P = np.zeros((self.L, 1),dtype=int)
        self.P[0] = init_loc
        self.D = self.Psi @ self.Theta
        T = self.Obs[:,:1].conj() @ self.Obs[:,:1].T
        # T = self.Obs[:,:1].T @ self.Obs[:,:1].conj() 
        self.logdet = np.linalg.slogdet(T)[1]

        self.done = False
        self.actions = [] 
        self.traj = [] # [self.s.reshape(2,1)]
        
        return self._get_obs()
    
    def _get_obs(self):
        x, y = self.s
        t = self.init_t + self.l * self.dt
        dx,dy = self._get_v(t,self.s)
        return np.array([x,y,self.l/self.L,dx,dy])

    def render(self):
        pass

    def close(self):
        pass 
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _ind_to_grid(self, ind):
        return np.array([ind % self.width, ind // self.width])

    def _grid_to_ind(self, grid):
        val = grid[0] + grid[1] * self.width
        return int(val)

    def _grid_to_loc(self,grid):
        x = grid[0] * self.x / (self.width-1)
        y = grid[1] * self.y / (self.height-1)
        return np.array([x,y])
    
    def _loc_to_grid(self, loc):
        x = np.round(loc[0] * (self.width-1) / self.x).astype(int)
        y = np.round(loc[1] * (self.height-1) / self.y).astype(int)
        return np.array([x,y])

    def get_path(self):
        return np.array([self._ind_to_grid(i) for i in self.P.flatten()])

    # double gyre functions
    def f(self,x,t):
        return self.eps*np.sin(self.omega*t)*x**2 + x - 2*self.eps*np.sin(self.omega*t)*x
    def df(self,x,t):
        return 2*self.eps*np.sin(self.omega*t)*x + 1 - 2*self.eps*np.sin(self.omega*t)
    def ddf(self,x,t):
        return 2*self.eps*np.sin(self.omega*t)
    def U(self,x,y,t):
        return -np.pi*self.A*np.sin(np.pi*self.f(x,t))*np.cos(np.pi*y)
    def V(self,x,y,t):
        return np.pi*self.A*np.cos(np.pi*self.f(x,t))*np.sin(np.pi*y)*self.df(x,t)
    def _get_v(self,t,loc):
        x,y = loc
        u = self.U(x,y,t) #-np.pi*self.A*np.sin(np.pi*self.f(x,t))*np.cos(np.pi*y)
        v = self.V(x,y,t) #np.pi*self.A*np.cos(np.pi*self.f(x,t))*np.sin(np.pi*y)*self.df(x,t)
        return np.array([u,v])
    def curl(self,x,y,t):
        return -np.pi**2*self.A*np.sin(np.pi*self.f(x,t))*np.sin(np.pi*y)*(1+self.df(x,t)**2) + \
            np.pi*self.A*np.cos(np.pi*self.f(x,t))*np.sin(np.pi*y)*self.ddf(x,t)

    def get_data(self,t):
        X,Y = np.meshgrid(np.linspace(0,self.x,self.width),np.linspace(0,self.y,self.height))
        return self.curl(X,Y,t)

    def _solve_ivp(self, action, store_traj):
        time = self.init_t + self.l * self.dt
        rhs = lambda t, y: self._get_v(t,y)+action # if y[0] > 0 and y[0] < self.x and y[1] > 0 and y[1] < self.y else 0
        sol = solve_ivp(rhs, (time, time+self.dt), self.s, events = self.land_event, max_step = self.dt/10)
        used_time = sol.t[-1]-time
        # if the sensor crashes on land, stop before event so that sensor doesn't crash on land
        # while self.land_event(sol.t[-1], sol.y[:,-1]) < 0:
        while np.any(sol.y[:,-1] < [0,0]) or np.any(sol.y[:,-1] > [self.x,self.y]):
            used_time = used_time - self.dt/10
            if used_time < self.dt/10:
                sol.success=False
                used_time = 0
                # return self._get_obs(), reward, True, {}
                break
            sol = solve_ivp(rhs, (time, time+used_time), self.s, events = self.land_event, max_step = self.dt/10)
        new_s = sol.y[:,-1] if sol.success else self.s

        if store_traj:
            if sol.success:
                idx = np.unique(sol.t,return_index=True)[1]
                tostore = np.vstack([sol.t[idx], sol.y[:,idx]])
            else:
                tostore = np.vstack([[time, time+self.dt], np.tile(self.s.reshape(2,1),(1,2))])
            self.traj.append(tostore)
        return new_s, abs(used_time - self.dt) > np.finfo(float).eps

    def plot_traj(self,traj = None,head_width=0.02, lw=1):
        if traj is None:
            locs = np.vstack([item[:,-1] for item in self.traj])
            traj = np.hstack(self.traj)
        else:
            locs = np.vstack([item[:,-1] for item in traj])
            traj = np.hstack(traj)

        snapshot = self.get_data(self.init_t)
        
        # plot
        # plt.figure()
        for i in range(traj.shape[1]-1):
            dx, dy = traj[1:,i+1] - traj[1:,i]
            plt.arrow(traj[1,i],traj[2,i],dx,dy,color='r')#,head_width=head_width,length_includes_head=True)
        plt.scatter(locs[1:-1,1], locs[1:-1,2],color='r',zorder=3,s=4*lw**2)
        plt.scatter(locs[0,1], locs[0,2],marker='*',color='r',zorder=3,s=6*lw**2)
        plt.scatter(locs[-1,1], locs[-1,2],marker='*',color='yellow',zorder=3,s=6*lw**2)

        plt.imshow(snapshot, origin='lower',extent=[0,self.x,0,self.y])
        X,Y = np.meshgrid(np.linspace(0,self.x,20),np.linspace(0,self.y,10))
        UU = self.U(X,Y,self.init_t)
        VV = self.V(X,Y,self.init_t)
        plt.quiver(X,Y,UU,VV)
        

       
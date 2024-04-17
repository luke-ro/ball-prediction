from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

g = -9.81

class Actor:
    def __init__(self, dynamics:Callable, x0):
        self.dynamics = dynamics
        self.x0 = x0
        self.y = None #`simulate` will populate this with the state wrt time

class Ball(Actor):
    def __init__(self, dynamics:Callable, x0):
        Actor.__init__(self, dynamics, x0)
        self.radius = x0[7]
    

class Car(Actor):
    pass

class Camera:
    def __init__(self, h_pixels, w_pixels):
        self.pos    = [0,0,0] #position relative to car
        self.h_pixels = h_pixels
        self.w_pixels = w_pixels
        self.aspect = float(w_pixels)/h_pixels
        self.screen = (-1, 1/self.aspect, 1, -1/self.aspect)
        self.origin = (-1,0,.1)
        
        
def ball_dynamics(t,x):
    """
    state = [x, y, z, #[m]
             dx, dy, dz, #[m/s] 
             m, #[kg]
             r #[m] 
             w, # [noise]
             Cr #[coeff rolling resistance]]
    """    
    dx = np.zeros(10)

    #rolling resitance
    F_rr = x[9]*g*x[6] # force of rolling resitance acts opposite of motion
    
    dx[0:3] = x[3:6] 

    v = np.linalg.norm(x[3:6])

    if v!=0:
        dx[3:6] = F_rr * x[3:6]/v


    return dx

def car_dynamics(t,x):
    """
    state = [x, y, z, #[m]
             theta, #[rad] 
             dx, dy, dz, #[m/s],
             steering angle, #[rad]
             m, # [kg]
             l, # [m] (wheel base) 
             w  # [noise]]
    """
    dx = np.zeros(11)
    v = np.linalg.norm(x[4:7])
    #position
    dx[0] = np.cos(x[3])*v
    dx[1] = np.sin(x[3])*v

    #dtheta
    R = x[9]/np.tan(x[7]) #turning radius
    # dx[3] = np.linalg.norm(x[4:6])/R
    dx[3] = np.linalg.norm(x[4:6])*np.tan(x[7])/x[9]

    # acelleration
    dx[4] = -np.sin(x[3])*v/R # *dv?
    dx[5] = np.cos(x[3])*v/R # *dv?
    # print(v)

    return dx

def plotBallState(t,y):
    fig,ax = plt.subplots(y.shape[0],1)
    [ax[i].plot(t, y[i,:]) for i in range(y.shape[0])]
    return fig,ax

def plotCarState(t,y):
    fig,ax = plt.subplots(y.shape[0],1)
    [ax[i].plot(t, y[i,:]) for i in range(y.shape[0])]
    return fig,ax

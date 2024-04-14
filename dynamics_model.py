import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = -9.18

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
    dx[3:6] = F_rr * x[3:6]/np.linalg.norm(x[3:6])
    return dx

def car_dynamics(x):
    """
    state = [x, y, z, #[m]
             dx, dy, dz, #[m/s] 
             m, # [kg]
             l, # [m] (wheel base) 
             w  # [noise]]
    """
    pass

def plotBallState(t,y):
    fig,ax = plt.subplots(y.shape[0],1)
    [ax[i].plot(t, y[i,:]) for i in range(y.shape[0])]

if __name__ == "__main__":
    t_span = [0,10]
    t_eval = np.linspace(0,10,100)

    x0_ball = [10,10,0,
              -.5,-.5,0,
              0.25,.1,0,.025]

    x0_car = [0,0]

    sol = solve_ivp(ball_dynamics,t_span,x0_ball, t_eval=t_eval)

    # print(sol.y[0:3,:])
    plotBallState(sol.t, sol.y)

    plt.show()
  



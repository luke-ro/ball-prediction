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

    #position
    dx[0] = np.cos(x[3])*x[4]
    dx[1] = np.sin(x[3])*x[5]

    #turning radius
    R = np.linalg.norm(x[0:3])/np.tan(x[7])

    #dtheta
    dx[3] = x[7]/x[9]

    #velocity
    F = np.array([0,0,0])
    R = np.array([[np.cos(dx[3]),-np.sin(dx[3])],
                  [np.sin(dx[3]), np.cos(dx[3])]])
    dx[4:6] = R@(F[0:2]+x[4:6])
    return dx

def plotBallState(t,y):
    fig,ax = plt.subplots(y.shape[0],1)
    [ax[i].plot(t, y[i,:]) for i in range(y.shape[0])]

def plotCarState(t,y):
    fig,ax = plt.subplots(y.shape[0],1)
    [ax[i].plot(t, y[i,:]) for i in range(y.shape[0])]



if __name__ == "__main__":
    t_span = [0,10]
    t_eval = np.linspace(0,10,100)

    x0_ball = [10,10,0,
              -.5,-.5,0,
              0.25,.1,0,.025]

    x0_car = [0,0,0,
              0,
              2,0,0,
              0,
              0,.1,0]

    sol_ball = solve_ivp(ball_dynamics,t_span,x0_ball, t_eval=t_eval)
    sol_car = solve_ivp(car_dynamics,t_span,x0_car, t_eval=t_eval)

    # print(sol.y[0:3,:])
    plotBallState(sol_ball.t, sol_ball.y)
    plotBallState(sol_car.t, sol_car.y)

    plt.show()
  



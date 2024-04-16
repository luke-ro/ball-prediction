import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from PIL import Image as im
from PIL import ImageOps
from datetime import datetime
import os
import json
from typing import Callable


g = -9.18

def normalize(v):
    return v/np.linalg.norm(v)

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

# https://omaraflak.medium.com/ray-tracing-from-scratch-in-python-41670e6a96f9
def sphereIntersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

# also https://omaraflak.medium.com/ray-tracing-from-scratch-in-python-41670e6a96f9
def nearestIntersectedObject(t, objects:list[Ball], ray_origin, ray_direction):
    distances = [sphereIntersect(obj.y[0:3,t], obj.radius, ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


class Environment:
    def __init__(self, car_actor:Car, objects:list[Ball], cam:Camera, dt, time_span):
        self.car       = car_actor
        self.objects   = objects
        self.dt        = dt
        self.time_span = time_span
        self.cam       = cam

        self.n_steps = int(np.ceil((time_span[1]-time_span[0])/dt))
        print("n_steps",self.n_steps)
        # make the timespan be a multiple of dt 
        self.time_span[1] = time_span[0] + dt*self.n_steps
        self.t_eval = np.linspace(self.time_span[0], self.time_span[1], self.n_steps)
    
    def getTimeVec(self):
        return self.t_eval

    def simulate(self):
        self.solutions = []
        for ob in self.objects:
            sol = solve_ivp(ob.dynamics, self.time_span, ob.x0, t_eval=self.t_eval, rtol=1e-6)
            self.solutions.append(sol)
            ob.y = sol.y
        
        #simulate car
        sol = solve_ivp(car.dynamics, self.time_span, car.x0, t_eval=self.t_eval, rtol=1e-6)

        self.car.y = sol.y

        return

    def generateFrames(self):
        #preallocate
        frames = np.zeros([self.n_steps, self.cam.h_pixels, self.cam.w_pixels, 3],np.uint8)

        for i,t in enumerate(np.linspace(self.time_span[0], self.time_span[1], self.n_steps)):
            print(f"Printing frame {i} at time {t}")
            for j,z in enumerate(np.linspace(self.cam.screen[1],self.cam.screen[3], self.cam.h_pixels)):
                for k,y in enumerate(np.linspace(self.cam.screen[0],self.cam.screen[2], self.cam.w_pixels)):
                    pixel = np.array([0,y,z])
                    orig = self.car.y[0:3,i]+self.cam.origin
                    direction = normalize(pixel - orig)
                    print(z,y,direction)

                    nearest_object, min_distance = nearestIntersectedObject(i, self.objects, orig, direction)
                    
                    if nearest_object is None:
                        continue
                    intersection = orig + min_distance*direction
                    

                    print(f"{y},{z},{intersection}")
                    frames[i,j,k] = 254
                    #intersection point:
            plt.imshow(frames[i,:,:])
            plt.show()
        self.frames = frames
        return frames

    def exportData(self,savedir):
        today = datetime.now()
        if today.hour < 12:
            h = "00"
        else:
            h = "12"
        folder = today.strftime(r'%Y%m%d')+ h + "%02d"%(today.minute,)
        os.mkdir(savedir+folder)

        to_save = []
        for i in range(self.n_steps):
            img = im.fromarray(self.frames[i])
            img = ImageOps.grayscale(img)

            finame = str(i)+".jpg"
            img.save(savedir+folder+"\\"+finame)

            data = {"img_file":finame,
                    "car_pos":self.car.y[0:2,i].tolist(),
                    "car_vel":self.car.y[4:6,i].tolist(),
                    "ball_pos":self.objects[0].y[0:2,i].tolist()}
            to_save.append(data)
        
        final = json.dumps(to_save, indent=2)
        print(to_save)
        with open(savedir+folder+"\\data.json", "w") as f: 
            f.write(final)
        
                    
                    

        

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



if __name__ == "__main__":
    t_span = [0,5]

    r_ball = 0.5
    x0_ball = [3, 0, r_ball,
              -.5, 0, 0,
              0.25, r_ball, 0, .025]

    x0_car = [0,0,0,
              0.0,
              0,0,0,
              0.0,
              0,.1,0]

    car = Car(car_dynamics, x0_car)
    ball = Ball(ball_dynamics, x0_ball)
    camera = Camera(40,60)

    env = Environment(car_actor=car, objects=[ball], cam=camera, time_span=t_span, dt=0.1)

    env.simulate()

    env.generateFrames()
    env.exportData("C:\\Users\\luker\\Downloads\\")

    t = env.getTimeVec()
    plotCarState(t,env.car.y)
    plotBallState(t,env.objects[0].y)

    # sol_ball = solve_ivp(ball_dynamics,t_span,x0_ball, t_eval=t_eval)
    # sol_car = solve_ivp(car_dynamics,t_span,x0_car, t_eval=t_eval, rtol=1e-6)

    # print(sol.y[0:3,:])
    # plotBallState(sol_ball.t, sol_ball.y)
    # plotBallState(sol_car.t, sol_car.y)

    fig,ax = plt.subplots(2,1)
    ax[0].plot(env.car.y[0,:],env.car.y[1,:])
    ax[1].plot(env.car.y[4,:],env.car.y[5,:])
    fig.suptitle("Pos and Vel")

    
    # fig,ax = plt.subplots()
    # ax.plot(np.linalg.norm(sol_car.y[4:7,:], axis=0))

  
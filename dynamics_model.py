import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from PIL import Image as im
from PIL import ImageOps
from datetime import datetime
import os
import json


import Actors

g = -9.81

def normalize(v):
    return v/np.linalg.norm(v)

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
def nearestIntersectedObject(t, objects:list[Actors.Ball], ray_origin, ray_direction):
    distances = [sphereIntersect(obj.y[0:3,t], obj.radius, ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


class Environment:
    def __init__(self, car_actor:Actors.Car, objects:list[Actors.Ball], cam:Actors.Camera, dt, time_span):
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

        self.ball_in_frame = np.zeros(self.n_steps,dtype=bool)

        for i,t in enumerate(np.linspace(self.time_span[0], self.time_span[1], self.n_steps)):
            print(f"Printing frame {i} at time {t}")
            for j,z in enumerate(np.linspace(self.cam.screen[1],self.cam.screen[3], self.cam.h_pixels)):
                for k,y in enumerate(np.linspace(self.cam.screen[0],self.cam.screen[2], self.cam.w_pixels)):
                    pixel = np.array([0,y,z])
                    orig = self.car.y[0:3,i]+self.cam.origin
                    direction = normalize(pixel - orig)
                    # print(z,y,direction)

                    nearest_object, min_distance = nearestIntersectedObject(i, self.objects, orig, direction)
                    
                    if nearest_object is None:
                        continue
                    intersection = orig + min_distance*direction
                    self.ball_in_frame[i] = True

                    # print(f"{y},{z},{intersection}")
                    frames[i,j,k] = 254
                    #intersection point:
            # plt.imshow(frames[i,:,:])
            # plt.show()
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
                    "ball_pos":self.objects[0].y[0:2,i].tolist(),
                    "ball_in_frame":int(self.ball_in_frame[i])}
            to_save.append(data)
        
        final = json.dumps(to_save, indent=2)
        print(to_save)
        with open(savedir+folder+"\\data.json", "w") as f: 
            f.write(final)





if __name__ == "__main__":
    t_span = [0,5]
    dt = 0.2
    r_ball = 0.5
    x0_ball = [3, 0, r_ball,
              -1.5, 0, 0,
              0.25, r_ball, 0, .025]

    x0_car = [0,0,0,
              0.0,
              0,0,0,
              0.0,
              0,.1,0]

    car = Actors.Car(Actors.car_dynamics, x0_car)
    ball = Actors.Ball(Actors.ball_dynamics, x0_ball)
    camera = Actors.Camera(40,60)

    env = Environment(car_actor=car, objects=[ball], cam=camera, time_span=t_span, dt=dt)

    env.simulate()

    env.generateFrames()
    env.exportData("C:\\Users\\luker\\Downloads\\")

    t = env.getTimeVec()
    Actors.plotCarState(t,env.car.y)
    Actors.plotBallState(t,env.objects[0].y)

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

  
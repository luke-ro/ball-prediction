import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from PIL import Image as im
from PIL import ImageOps
from datetime import datetime
import os
import json

import Actors

from scipy.spatial.transform import Rotation as R

g = -9.81

def R3D(r3,r2,r1):
    r = R.from_euler('ZYX', [r3, r2, r1], degrees=False)
    return r.as_matrix()

def T3D(r3,r2,r1,p):
    R = R3D(r3,r2,r1)
    T = np.zeros([3,4])
    T[0:3,0:3] = R
    T[0:3,3] = p
    return T

def R2D(ang):
    R_mat = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang), np.cos(ang)]]) 
    return R_mat

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
        sol = solve_ivp(self.car.dynamics, self.time_span, self.car.x0, t_eval=self.t_eval, rtol=1e-6)

        self.car.y = sol.y

        return

    def generateFrames(self,n=None):
        if n is None:
            n = self.n_steps

        #preallocate
        frames = np.zeros([n, self.cam.h_pixels, self.cam.w_pixels, 3],np.uint8)

        self.ball_in_frame = np.zeros(n,dtype=bool)

        for i,t in enumerate(np.arange(self.time_span[0], self.time_span[0]+self.dt*n, self.dt)):
            print(f"Printing frame {i} at time {t}")
            theta = self.car.y[3,i]
            R2 = R2D(theta)
            T3 = T3D(theta,0,0,self.car.y[0:3,i])
            screen_z = self.cam.screen_zlim
            screen_xy = np.zeros((2,2))
            screen_xy[:,0] = (T3@[0,self.cam.screen_ylim[0],0,1])[0:2] 
            screen_xy[:,1] = (T3@[0,self.cam.screen_ylim[1],0,1])[0:2]

            xy_points = np.linspace(screen_xy[:,0], screen_xy[:,1], self.cam.w_pixels)
            orig = (T3@np.concatenate((self.cam.origin,[1]),axis=0))
            
            for j,z in enumerate(np.linspace(screen_z[0], screen_z[1], self.cam.h_pixels)):
                for k in range(xy_points.shape[0]):
                    xy = xy_points[k,:]

                    pixel = np.concatenate((xy,[z]),axis=0)
                    direction = normalize(pixel - orig)
                    # print(z,y,direction)

                    nearest_object, min_distance = nearestIntersectedObject(i, self.objects, orig, direction)
                    if nearest_object is None:
                        continue
                    # print(min_distance)
                    intersection = orig + min_distance*direction
                    self.ball_in_frame[i] = True

                    # print(f"{y},{z},{intersection}")
                    frames[i,j,k] = 254
                    #intersection point:
            # plt.imshow(frames[i,:,:])
            # plt.show()
        self.frames = frames
        return frames

    def exportData(self,savedir,foldername=None):
        
        if foldername is None:
            today = datetime.now()
            if today.hour < 12:
                h = "00"
            else:
                h = "12"
            folder = today.strftime(r'%Y%m%d')+ h + "%02d"%(today.minute,)+"\\"
        else:
            folder = foldername

        os.mkdir(savedir+folder)

        to_save = {}
        n = self.frames.shape[0]
        for i in range(n):
            img = im.fromarray(self.frames[i])
            img = ImageOps.grayscale(img)

            finame = str(i)+".jpg"
            img.save(savedir+folder+finame)

            data = {"img_file":finame,
                    "car_pos":self.car.y[0:2,i].tolist(),
                    "car_vel":self.car.y[4:6,i].tolist(),
                    "ball_in_frame":int(self.ball_in_frame[i])}

            to_save[str(i)] = data
        
        to_save["ball_trj"] = {"pos": self.objects[0].y[0:2,n:].tolist(),
                               "vel": self.objects[0].y[3:5,n:].tolist()}
        
        final = json.dumps(to_save, indent=2)
        print(to_save)
        with open(savedir+folder+"\\data.json", "w") as f: 
            f.write(final)
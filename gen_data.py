import matplotlib.pyplot as plt
import numpy as np
import os

import Actors
import Environment

# BALL_X_MIN = 2
# BALL_X_MAX = 4
# BALL_Y_MIN = -1
# BALL_Y_MAX = 1
# BALL_dX_MIN = -1
# BALL_dX_MAX = 1
# BALL_dY_MIN = -1
# BALL_dY_MAX = 1
BALL_X_MIN = 3
BALL_X_MAX = 1
BALL_Y_MIN = -1
BALL_Y_MAX = 1
BALL_dX_MIN = -1
BALL_dX_MAX = 0
BALL_dY_MIN = -1
BALL_dY_MAX = 1

CAR_V_MIN_BODY = 0
CAR_V_MAX_BODY = 0
CAR_THETA_MIN = 0
CAR_THETA_MAX = 0
CAR_TURN_ANGLE_MIN = -3.14/24
CAR_TURN_ANGLE_MAX = 3.14/24

WHEEL_BASE = 0.1


if __name__ == "__main__":
    
    save_dir = "C:\\Users\\luker\\Documents\\repos\\ball-prediction\\data\\temp\\"
    t_span = [0,4]
    dt = 0.2
    RADIUS_BALL = 0.25

    for i in range(10000):
        print(f"i={i}")
        x0_ball = [np.random.uniform(BALL_X_MIN,BALL_X_MAX), np.random.uniform(BALL_Y_MIN,BALL_Y_MAX), RADIUS_BALL,
                np.random.uniform(BALL_dX_MIN,BALL_dX_MAX), np.random.uniform(BALL_dY_MIN,BALL_dY_MAX), 0,
                0.25, RADIUS_BALL, 0, .025]

        car_ang = np.random.uniform(CAR_THETA_MIN,CAR_THETA_MAX)
        car_v = Environment.R2D(car_ang)@[np.random.uniform(CAR_V_MIN_BODY,CAR_V_MAX_BODY), 0] #rotate vel 
        x0_car = [0,0,0,
                car_ang,
                car_v[0],car_v[1],0,
                np.random.uniform(CAR_TURN_ANGLE_MIN,CAR_TURN_ANGLE_MAX),
                0,WHEEL_BASE,0]

        car = Actors.Car(Actors.car_dynamics, x0_car)
        ball = Actors.Ball(Actors.ball_dynamics, x0_ball)
        camera = Actors.Camera(20,30)

        env = Environment.Environment(car_actor=car, objects=[ball], cam=camera, time_span=t_span, dt=dt)

        env.simulate()

        env.generateFrames(n=10)

        # os.mkdir(save_dir+str(i)+"\\")

        env.exportData(save_dir,foldername=str(i)+"\\")

    # t = env.getTimeVec()
    # Actors.plotCarState(t,env.car.y)
    # Actors.plotBallState(t,env.objects[0].y)

    # sol_ball = solve_ivp(ball_dynamics,t_span,x0_ball, t_eval=t_eval)
    # sol_car = solve_ivp(car_dynamics,t_span,x0_car, t_eval=t_eval, rtol=1e-6)

    # print(sol.y[0:3,:])
    # plotBallState(sol_ball.t, sol_ball.y)
    # plotBallState(sol_car.t, sol_car.y)

    # fig,ax = plt.subplots(2,1)
    # ax[0].plot(env.car.y[0,:],env.car.y[1,:])
    # ax[1].plot(env.car.y[4,:],env.car.y[5,:])
    # fig.suptitle("Pos and Vel")

    
    # fig,ax = plt.subplots()
    # ax.plot(np.linalg.norm(sol_car.y[4:7,:], axis=0))
    plt.show()

  
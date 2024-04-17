import matplotlib.pyplot as plt

import Actors
import Environment

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

    env = Environment.Environment(car_actor=car, objects=[ball], cam=camera, time_span=t_span, dt=dt)

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

  
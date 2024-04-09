import math

import matplotlib.pyplot as plt
import numpy as np
import USV_model

show_animation = True

# Wraps angle to [-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

def propagate(USV, SS, U_c):

    State_projection = []
    dt = 5 # time in seconds
    
    # list of rudder angles
    rudder_angles = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

    for rudder_angle in rudder_angles:
        new_state = SS.copy()
        rudder_angle = rudder_angle * np.pi/180 # convert to radians            
        
        new_nu = USV.dynamics(USV.nu, rudder_angle, dt)

        U = new_nu[0]  # linear velocity of the vessel
        r = new_nu[5]  # angular velocity of the vessel

        for t in range (20):

            # update the state in the inertial frame
            new_state[2] += dt * r
            new_state[0] += dt * U * np.cos(new_state[2])
            new_state[1] += dt * U * np.sin(new_state[2])
            
            new_state[2] = wraptopi(new_state[2])


            # adding the ocean current effect
            new_state[0] += U_c[0] * dt
            new_state[1] += U_c[1] * dt


            State_projection.append(new_state.copy())

    return State_projection

def otter_propagate(USV, SS, U_c):

    State_projection = []
    dt = 5 # time in seconds
    
    # list of rudder angles
    rotor1 = []
    rotor2 = []


    for i in range(13):
        rotor1.append(5 + 0.1 * (i-6))
        rotor2.append(5 + 0.1 * (6-i))

    rotor_speeds = [[x, y] for x, y in zip(rotor1, rotor2)]

    for u_actual in rotor_speeds:
        new_state = SS.copy()

        for t in range (21):
            eta = np.array([new_state[0], new_state[1], 0, 0, 0, new_state[2]]) 
            new_nu = USV.dynamics(eta, USV.nu,u_actual, dt)
            

            U = new_nu[0]  # linear velocity of the vessel
            r = new_nu[5]  # angular velocity of the vessel

            # update the state in the inertial frame
            new_state[2] += dt * r
            new_state[2] = wraptopi(new_state[2])

            new_state[0] += dt * U * np.cos(new_state[2])
            new_state[1] += dt * U * np.sin(new_state[2])
            



            # adding the ocean current effect
            new_state[0] += U_c[0] * dt
            new_state[1] += U_c[1] * dt


            State_projection.append(new_state.copy())

    return State_projection

def plot_arrow(x, y, yaw, length=5.0, width=3, fc="r", ec="k"):  # pragma: no cover
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def main():
    print("Executing: " + __file__)

       
    v_cx = 3     # current in the x direction on the inertial frame
    v_cy = 3     # current in the y direction on the inertial frame
    V_c = np.sqrt((v_cx**2) + (v_cy**2))
    beta_c = np.arctan2(v_cy,v_cx)

    U_c = [v_cx, v_cy]      # Ocean current w.r.t. the inertial frame

    USV = USV_model.frigate(U = 5, r = 0 )  # r = 0 represent that the vessile can't rotate in place
    # USV = USV_model.otter(V_c, beta_c)

    SS = [0, 0 , 0]   # start state

    projection = propagate(USV, SS, U_c)
    # projection = otter_propagate(USV,SS, U_c) # for the otter model

    print(len(projection))
    print(USV.nu[0])

    # Draw projection
    
    # plt.scatter([state[0] for state in projection], [state[1] for state in projection])
    plot_arrow([state[0] for state in projection], [state[1] for state in projection],
               [state[2] for state in projection])
    plt.grid(True)
    plt.pause(0.001) # Necessary for macs
    plt.show()

if __name__ == '__main__':
    main()




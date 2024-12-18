import matplotlib.pyplot as plt
import numpy as np
import os
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer
from numba import cfunc, types
from OdeSolver import solve_ode




# start by only solving for Rocket's path




def rocket_equations(t, vars, params, dfdt):
    x = vars[0]
    vx = vars[1]
    y = vars[2]
    vy = vars[3]
    F_thrust = params[0]
    GM_earth = params[1]
    m0 = params[2]
    dot_m = params[3]
    g0 = params[4] # Earth's acceleration of gravity at surface


    m = m0 - dot_m * t
    v = np.sqrt(vx**2 + vy**2)


    
  


  # what we aim is to align our thrust with our velocity and let gravity do the turn. We don't want
    
    r = np.sqrt(x**2 + y**2)
    g = GM_earth/(r**2)
    ay = (F_thrust * vy/v - m*g * (y/r)) / m
    ax = (F_thrust * vx/v - m * g * (x/r)) / m  

    dfdt[0] = vx
    dfdt[1] = ax
    dfdt[2] = vy
    dfdt[3] = ay

# stage 1: 200 ton fuel and 20 ton dry mass
g0 = 9.81
m0 = 400000.0
dot_m = 1300
Isp = 800
F_thrust = dot_m * Isp *g0

GM_earth = 3.986004418e14

params = [F_thrust, GM_earth, m0, dot_m, g0]

t_span = [0.0, 153.8465] # this amount of time burns fuel of 200,000 kg. Which is what I desired in the first stage.

nsteps = 10000
y0 = 6371000.0    # Earth radius in meters
x0 = 0
itheta = 89.70
vy0 = 0.1 * np.sin(np.deg2rad(itheta))
vx0 = 0.1 * np.cos(np.deg2rad(itheta))


y0 = [x0, vx0, y0, vy0]


t, vars = solve_ode(rocket_equations, t_span, nsteps, y0, args=params, method="RK4")

x = vars[:,0]
vx = vars[:,1]
y = vars[:,2]
vy = vars[:,3]

# we will eject the 20 ton dry mass and give the associated momentum in the direction of the rocket's velocity 


vx_2 = vx[-1]
vy_2 = vy[-1]
x_2 = x[-1]
y_2 = y[-1]

r_array = np.sqrt(x**2 + y**2)
theta_array = np.arctan2(y, x)
vr_array = (x*vx + y*vy)/r_array
vtheta_array = (-y*vx + x*vy)/r_array


plt.figure()
plt.plot(t, vr_array, label='v_r')
plt.plot(t, vtheta_array, label='v_θ')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Radial and Tangential Velocities vs Time (Stage 1)')
plt.grid(True)
plt.legend()
plt.show()


final_angle = np.arctan2(vy_2, vx_2)
print("Final angle (degrees): ", np.degrees(final_angle))



# we will eject a mass of 20 tons but it won't impart any signiticant momentum as v_eject = 0

#so the remaining mass is 180000 and we are ready for stage 2

m2 = 180000
dot_m2= 300
F_thrust2 = dot_m2 * Isp *g0

params2 = [F_thrust2, GM_earth, m2, dot_m2, g0]



t_span2 = [0.0, 105] # this amount of time burns fuel of 200,000 kg. Which is what I desired in the first stage.





y2 = [x_2, vx_2, y_2, vy_2]



t2, vars2 = solve_ode(rocket_equations, t_span2, nsteps, y2, args=params, method="RK4")

x2 = vars2[:,0]
vx2 = vars2[:,1]
y2 = vars2[:,2]
vy2 = vars2[:,3]

theta2 = np.arctan(vy2[-1]/ vx2[-1])
print(np.rad2deg(theta2))

# Convert stage 2 results to polar and radial/tangential velocities:
r2 = np.sqrt(x2**2 + y2**2)
theta2_array = np.arctan2(y2, x2)
vr2_array = (x2*vx2 + y2*vy2)/r2
vtheta2_array = (-y2*vx2 + x2*vy2)/r2

final_altitude = r2[-1] - 6371000.0
final_vr = vr2_array[-1]
final_vtheta = vtheta2_array[-1]
final_theta = theta2_array[-1]


print("Final altitude after stage 2:", final_altitude, "m")
print("Final v_r after stage 2:", final_vr, "m/s")
print("Final v_θ after stage 2:", final_vtheta, "m/s")
print("Final θ after stage 2:", final_theta, "rad")
with open("final_conditions_stage2.txt", "w") as f:
    f.write(f"{x2[-1]} {y2[-1]} {vx2[-1]} {vy2[-1]}\n")

# Plot trajectory (y vs x) for stage 2
plt.figure()
plt.plot(x2, y2)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Rocket Trajectory (Stage 2)')
plt.grid(True)

# Plot r vs theta(Stage 2)
plt.figure()
plt.plot(np.degrees(theta2_array), r2 - 6371000.0) # Just to see altitude as a function of angle
plt.xlabel('Theta (degrees)')
plt.ylabel('r - R_earth (m)')
plt.title('Radius vs Theta (Stage 2)')
plt.grid(True)

# Plot v_r and v_theta vs time (Stage 2)
plt.figure()
plt.plot(t2, vr2_array, label='v_r')
plt.plot(t2, vtheta2_array, label='v_θ')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Radial and Tangential Velocities vs Time (Stage 2)')
plt.grid(True)
plt.legend()

plt.show()


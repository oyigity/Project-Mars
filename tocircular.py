import matplotlib.pyplot as plt
import numpy as np
import os
import ctypes
from ctypes import *
from numpy.ctypeslib import ndpointer
from numba import cfunc, types
from OdeSolver import solve_ode
from launch import rocket_equations


x_final, y_final, vx_final, vy_final = np.loadtxt("final_conditions_stage2.txt", unpack=True)
# Define a new rocket_equations function but this time we assume no thrust to just see the orbit

def orbit_equations(t, vars, params, dfdt):
    x = vars[0]
    vx = vars[1]
    y = vars[2]
    vy = vars[3]
    GM_earth = params[0]

    # no thrust now, just gravity
    r = np.sqrt(x**2 + y**2)
    g = GM_earth/(r**2)

    # acceleration only due to gravity
    
    ax = -g * (x/r)
    ay = -g * (y/r)

    dfdt[0] = vx
    dfdt[1] = ax
    dfdt[2] = vy
    dfdt[3] = ay

# Earths GM
GM_earth = 3.986004418e14

# Parameters for orbit_equations (just need GM_earth as the main parameter)
params_orbit = [GM_earth]

# Initial conditions from file
y_orbit = [x_final, vx_final, y_final, vy_final]

# Let's simulate the orbit for, say, a few hours to see its shape
# time span in seconds: let's pick something like 10,000 seconds (2.7 hours)
t_span_orbit = [0.0, 10000]
nsteps_orbit = 10000

t_orbit, vars_orbit = solve_ode(orbit_equations, t_span_orbit, nsteps_orbit, y_orbit, args=params_orbit, method="RK4")

x_orb = vars_orbit[:,0]
vx_orb = vars_orbit[:,1]
y_orb = vars_orbit[:,2]
vy_orb = vars_orbit[:,3]


r_final = np.sqrt(x_orb[-1]**2 + y_orb[-1]**2)
r_orb = np.sqrt(x_orb**2 + y_orb**2)
theta_orb = np.arctan2(y_orb, x_orb)
vr_orb = (x_orb*vx_orb + y_orb*vy_orb)/r_orb
vtheta_orb = (-y_orb*vx_orb + x_orb*vy_orb)/r_orb
# Plot the resulting orbit trajectory
plt.figure()
plt.plot(x_orb, y_orb)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Elliptical Orbit Trajectory After Stage 2')
plt.grid(True)
plt.axis('equal')  # make it equal scale to see ellipse shape better
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')  # Ensures equal scaling

# Plot radial and tangential velocities vs time to confirm orbit shape
# the orbit is only slightly eliptical or is it really, the apogee and perigee have an altitude difference about 900 kms, so 

plt.figure()
plt.plot(t_orbit, vr_orb, label='v_r')
plt.plot(t_orbit, vtheta_orb, label='v_θ')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Radial and Tangential Velocities vs Time (No Thrust Phase)')
plt.grid(True)
plt.legend()

plt.show()

# we will get the parameter values at vr =0 (apogee) 
for i in range(len(vr_orb)-1):
    if vr_orb[i] > 0 and vr_orb[i+1] < 0:
        # Zero crossing between t[i] and t[i+1]
        # Find t_zero by linear interpolation
        t_i = t_orbit[i]
        t_ip1 = t_orbit[i+1]

        vr_i = vr_orb[i]
        vr_ip1 = vr_orb[i+1]

        # Linear interpolation for t_zero:
        # vr(t) ~ vr_i + (vr_ip1 - vr_i)*((t - t_i)/(t_ip1 - t_i))
        # Set vr(t_zero)=0 and solve for t_zero:
        t_zero = t_i - vr_i * (t_ip1 - t_i) / (vr_ip1 - vr_i)

        # Now interpolate x, y, vx, vy at t_zero
        def lin_interp(val_i, val_ip1):
            return val_i + (val_ip1 - val_i)*((t_zero - t_i)/(t_ip1 - t_i))

        x_zero = lin_interp(x_orb[i], x_orb[i+1])
        y_zero = lin_interp(y_orb[i], y_orb[i+1])
        vx_zero = lin_interp(vx_orb[i], vx_orb[i+1])
        vy_zero = lin_interp(vy_orb[i], vy_orb[i+1])

        r_zero = np.sqrt(x_zero**2 + y_zero**2)
        theta_zero = np.arctan2(y_zero, x_zero)
        vr_zero = (x_zero*vx_zero + y_zero*vy_zero)/r_zero
        vtheta_zero = (-y_zero*vx_zero + x_zero*vy_zero)/r_zero
        index = i

        print("At v_r=0:")
        print("Array Index:", i)
        print("Time:", t_zero)
        print("r:", r_zero, "m")
        print("theta:", theta_zero, "rad")
        print("v_theta:", vtheta_zero, "m/s")
        print("v_r (should be about 0):", vr_zero)

        break  # Stop after finding the first

# we can now compute what is the desired orbital velocity to get a circular orbit
v_desired = np.sqrt(GM_earth/(r_zero))

# time to use the rocket equation to see how much mass we should burn
# remember from our launch we have 180000 - 105* 300 = 149500 kg mass remaining
# we are still in stage 2 having 28500 kg left

m0 = 149500
del_v = np.abs(v_desired) - np.abs(vtheta_zero)

Isp = 800
g0 = 9.81

mf  = m0/np.exp(del_v /(Isp*g0))

mburned = m0 - mf

#check
print("mburned:", mburned)
# mass burned is 6889 kg

# so lets burn it
dot_m = 10000 # assume a very fast burn rate to get a perfect circle(if the speed gain is not instantenous then we might get velocity in radial direction as the object moves in its original eliptical trajectory during the time that passes ) 
F_thrust = dot_m * Isp *g0 

params = [F_thrust, GM_earth, m0, dot_m, g0]

t_span1 = [0.0, mburned/dot_m] # something like 20 Seconds

# parameters at apogee in cartesian coordinates
x_apo = x_orb[index]
y_apo = y_orb[index]
vx_apo = vx_orb[index]
vy_apo = vy_orb[index]
ic = [x_apo, vx_apo, y_apo,vy_apo]
t_circ, vars_circ = solve_ode(rocket_equations, t_span1, 50000, ic, args=params, method="RK4")


x_circ = vars_circ[:,0]
vx_circ = vars_circ[:,1]
y_circ = vars_circ[:,2]
vy_circ = vars_circ[:,3]

# I might write a function for these transformations later on
r_circ = np.sqrt(x_circ**2 + y_circ**2)
theta_circ = np.arctan2(y_circ, x_circ)
vr_circ = (x_circ*vx_circ + y_circ*vy_circ)/r_circ
vtheta_circ = (-y_circ*vx_circ + x_circ*vy_circ)/r_circ

# by the way the reason I switch coordinates all the time is because it is easier to solve the equations of motion in cartesian coordinates
# but it makes more sense to analzye in terms polar coordinates in orbits


print("altitude", r_circ[-1] - 6371000.0) # we started at 150 km of altitude and then traced the ellipse until the apogee, which has an altitude more than 1000 km! we are in a pretty high orbit right now!!!

# Plot r_circ, theta_circ, vr_circ, vtheta_circ vs time
plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
plt.plot(t_circ, r_circ)
plt.xlabel('Time (s)')
plt.ylabel('r (m)')
plt.title('Radius vs Time')


plt.subplot(2,2,2)
plt.plot(t_circ, theta_circ)
plt.xlabel('Time (s)')
plt.ylabel('theta (rad)')
plt.title('Theta vs Time')

plt.subplot(2,2,3)
plt.plot(t_circ, vr_circ)
plt.xlabel('Time (s)')
plt.ylabel('v_r (m/s)')
plt.title('Radial Velocity vs Time')

plt.subplot(2,2,4)
plt.plot(t_circ, vtheta_circ)
plt.xlabel('Time (s)')
plt.ylabel('v_theta (m/s)')
plt.title('Tangential Velocity vs Time')

plt.tight_layout()
plt.show()


# Earth's gravity is still vastly dominant at this altitude (about 1000 km), so we don't have to worry about any other body right now

# now let us see for ourselves that this orbit is indeed circular

ic2 =[x_circ[-1], vx_circ[-1], y_circ[-1], vy_circ[-1]]


tf = 3600 * 24 * 1 #1 day
t_span_final = [0.0, tf] 
nsteps_final = 100000



t_final, vars_final = solve_ode(orbit_equations, t_span_final, nsteps_final, ic2, args=params_orbit, method="RK4")



x_final_arr = vars_final[:, 0]
vx_final_arr = vars_final[:, 1]
y_final_arr = vars_final[:, 2]
vy_final_arr = vars_final[:, 3]

r_final_arr = np.sqrt(x_final_arr**2 + y_final_arr**2)

# Plot Y vs X
plt.figure()
plt.plot(x_final_arr, y_final_arr)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Final Orbit Trajectory (Y vs X)')
plt.grid(True)
plt.axis('equal')  # Ensure equal scaling for proper orbit shape
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')

# Plot R vs Time
plt.figure()
plt.plot(t_final, r_final_arr)
plt.xlabel('Time (s)')
plt.ylabel('r (m)')
plt.title('Radius vs Time for Final Orbit')
plt.grid(True)

plt.show()

#Now we integrate a movie creation for the elliptical to circular transition
# We will combine x_orb,y_orb (elliptical phase) and x_circ,y_circ (circularization burn phase) 
# and final circular orbit into one sequence for a simple animation. There is a lag but it is fine.
import matplotlib.animation as animation



step = 1000 
x_total = np.concatenate([x_orb[:index+1], x_circ, x_final_arr])
y_total = np.concatenate([y_orb[:index+1], y_circ, y_final_arr])

t_total = np.concatenate([t_orbit[:index+1],
                          t_circ + t_orbit[index],
                          t_final + t_orbit[index] + t_circ[-1]])

frame_indices = range(0, len(t_total), step)

fig, ax = plt.subplots(figsize=(6,5))
ax.set_aspect('equal', 'box')
ax.set_xlim(min(x_total)-50000, max(x_total)+50000)
ax.set_ylim(min(y_total)-50000, max(y_total)+50000)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Orbit Transition: Elliptical to Circular')

rocket_plot, = ax.plot([], [], 'o', color='green', label='Rocket')
rocket_path, = ax.plot([], [], '-', color='green', alpha=0.5)
ax.legend()

def init_anim():
    rocket_plot.set_data([], [])
    rocket_path.set_data([], [])
    return rocket_plot, rocket_path

def update_anim(frame):
    i = frame
    rocket_plot.set_data([x_total[i]], [y_total[i]])
    rocket_path.set_data(x_total[:i], y_total[:i])
    return rocket_plot, rocket_path

ani = animation.FuncAnimation(fig, update_anim, frames=frame_indices, init_func=init_anim, interval=20, blit=False, repeat=False)

Writer = animation.FFMpegWriter
writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

ani.save('orbit_transition.mp4', writer=writer, dpi=100)
plt.close(fig)
print("Animation 'orbit_transition.mp4' saved successfully.")


#  We see that we have circular orbit with a radius of about 7456.646 km
#  We saw that we have circular orbit with a radius of about 7456.646 km
# At this stage the mass the fuel left in stage 2 is 121610
# As a reminder we assumed a total mass of 400,000 kg
# STAGE 1: We burned 200,000 kg fuel and ejected 20,000 kg dry mass
#STAGE 2: Of the 160,000 kg available fuel we have 121610 kg remaining and a 20000kg dry mass that is for simplicity the core of the rocket
# The two stage model is for simplicity.

# In our next step we will try to achieve a Hoffman transfer/transfers to an orbit around Mars, but before that let us integrate our rocket to this heliocentric frame(sun's reference frame)






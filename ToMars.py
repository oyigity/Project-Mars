import matplotlib.pyplot as plt
import numpy as np
import os
import ctypes
import math
from pykep import lambert_problem, MU_SUN,AU
from ctypes import *
from numpy.ctypeslib import ndpointer
from numba import cfunc, types
from OdeSolver import solve_ode
from launch import rocket_equations
from CoordinateTransformations import ellipse_to_xy, xy_to_ellipse

#  We saw that we have circular orbit with a radius of about 7456.646 km
# At this stage the mass the fuel left in stage 2 is 21610 kg
# As a reminder we assumed a total mass of 400,000 kg
# STAGE 1: We burned 200,000 kg fuel and ejected 20,000 kg dry mass
#STAGE 2: Of the 160,000 kg available fuel we have 121610 kg remaining and a 20000kg dry mass that is for simplicity the core of the rocket
# The two stage model is for simplicity.

#We took January 1,2000 as our epoch, data was collected from NASA database

# We start with a Keplter solver that has an algorithm based on first order approximation Taylor series. This is the Newton root finding algorithm
def solve_kepler(M, e, tol=1e-12):
    E = M if e < 0.8 else math.pi
    while True:
        f = E - e*math.sin(E) - M
        fprime = 1 - e*math.cos(E)
        E_new = E - f/fprime
        if abs(E_new - E) < tol:
            return E_new
        E = E_new

# Extracting the position of planets from their ellyptic orbit parameters. The data was collected from NASA.  The goal is to use Mean anomaly to find the eccentricity anomaly and then compute the true anomaly
def position_from_elements(a, e, omega, M0, t, GM=4*math.pi**2, t0=0.0):
    #Mean Angular Frequency:
    n = math.sqrt(GM/a**3)
    M = M0 + n*(t - t0) # Mean Anomaly
    E = solve_kepler(M, e)
    # The equation for true anomaly in terms of the eccentricity anomaly below is well known and can be derived from the cartesian coordinate expressions in terms of ellitpical coordinate
    nu = 2*math.atan2(math.sqrt(1+e)*math.sin(E/2), math.sqrt(1-e)*math.cos(E/2))
    theta = (nu + omega) % (2*math.pi) # True anomaly+ the argument of perihelion, we had this logic in the solar system homework
    x,vx,y,vy = ellipse_to_xy(a, e, theta, omega, GM)
    return x, vx, y, vy

GM_sun = 4*math.pi**2 

# Earth orbital elements:
a_earth = 1.000
e_earth = 0.0167
omega_earth_deg = 114.20783
M0_earth_deg = 357.51716
omega_earth = math.radians(omega_earth_deg)
M0_earth = math.radians(M0_earth_deg)

# Mars orbital elements:
a_mars = 1.523679
e_mars = 0.0934
omega_mars_deg = 286.502
M0_mars_deg = 19.41248
omega_mars = math.radians(omega_mars_deg)
M0_mars = math.radians(M0_mars_deg)

T = 1.0 # 1 year
steps = 1000
times = [i*(T/steps) for i in range(steps+1)]

earth_x = []
earth_y = []
mars_x = []
mars_y = []

# We can plot Earth and Mars' orbit within a year, by incrementing time and solving for position using the position_from elements functiona and appending each step to an array
for t in times:
    ex,vx_e,ey,vy_e = position_from_elements(a_earth, e_earth, omega_earth, M0_earth, t, GM_sun)
    mx,vx_m,my,vy_m = position_from_elements(a_mars, e_mars, omega_mars, M0_mars, t, GM_sun)
    earth_x.append(ex)
    earth_y.append(ey)
    mars_x.append(mx)
    mars_y.append(my)

plt.figure()
plt.plot(earth_x, earth_y, label='Earth')
plt.plot(mars_x, mars_y, label='Mars')
plt.xlabel('X (AU)')
plt.ylabel('Y (AU)')
plt.title('Earth & Mars Elliptical Orbits after J2000 Epoch')
plt.grid(True)
plt.axis('equal')
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.legend()
plt.show()

#The parameters associated with the Rocket
rocket_radius_km = 7456.646
rocket_radius_m = rocket_radius_km * 1000.0
GM_earth = 3.986004418e14  # m^3/s^2
omega = math.sqrt(GM_earth / rocket_radius_m**3) #rad/s
# Now note that we will start by escaping earth's sphere of influence
# orbital velocity of rocket around earth wrt earth:
v_circular = math.sqrt(GM_earth / rocket_radius_m)

# The velocity needed to escape earth
v_escape = v_circular * math.sqrt(2.0)

print("At radius:", rocket_radius_km, "km above Earth's center")
print("Circular Orbit Velocity (m/s):", v_circular)
print("Escape Velocity (m/s):", v_escape)




#conversion factors:
m_to_AU = 1.0 / (1.495978707e11) # METERS TO AU
s_to_yr = 1.0 / (3.15576e7) # Seconds toyear
# 1 m/s to AU/yr:
m_s_to_au_yr = (3.15576e7 / 1.495978707e11) #from m_s to au_yr

# We can get the rocket position wrt Earth by putting a time value in years
# We do appropriate unit conversion to switch to the AU units
def rocket_position_around_earth(t):
    # t in years
    t_sec = t / s_to_yr
    x_rE_m = rocket_radius_m * math.cos(omega * t_sec)
    y_rE_m = - rocket_radius_m * math.sin(omega * t_sec)

    vx_rE_m_s = - rocket_radius_m * omega * math.sin(omega * t_sec)
    vy_rE_m_s =   rocket_radius_m * omega * math.cos(omega * t_sec)

    x_rE = x_rE_m * m_to_AU
    y_rE = y_rE_m * m_to_AU
    vx_rE = vx_rE_m_s * m_s_to_au_yr
    vy_rE = vy_rE_m_s * m_s_to_au_yr

    return x_rE, vx_rE, y_rE, vy_rE


# Note that the rocket velocity and position was relative to earth



# We want mars to lead earth by 40-45 degrees before initiating the transfer. According to literature, this is optimal value for transfer.
lead_angle_rad = math.radians(42.0) # target lead angle
t_depart = None # start with none and update 

#Iterate over the trajectory of Earth and Mars to find when Mars leads earth by 42 degrees

for t in times:
    Ex_,VxE_,Ey_,VyE_ = position_from_elements(a_earth, e_earth, omega_earth, M0_earth, t, GM_sun)
    Mx_,VxM_,My_,VyM_ = position_from_elements(a_mars, e_mars, omega_mars, M0_mars, t, GM_sun)
    theta_earth = math.atan2(Ey_,Ex_)
    theta_mars = math.atan2(My_,Mx_)
    angle_diff = (theta_mars - theta_earth) % (2*math.pi)

    if abs(angle_diff - lead_angle_rad) < math.radians(0.5):
        t_depart = t
        break;

if t_depart is None:
    print("Could not find a time within 1 year where Mars leads Earth by ~44°.")
else:
    print("Found departure time: t_depart =", t_depart, "years")

# We have found the departure time, but we will just make a small edit to find when the velocity of Rocket is alligned with the velocity of Earth

Ex,VxE,Ey,VyE = position_from_elements(a_earth, e_earth, omega_earth, M0_earth, t_depart, GM_sun)
#Earth's velocity angle
thetaE = math.atan2(VyE,VxE)

# Below we devised an algorithm to find the time when the velocity of the Earth and the rocket are aligned
best_t = t_depart
min_angle_diff = 1e9  
time_step = 1e-5  
search_window = 0.0005  # The orbital period of rocket around the earth is about 0.002 years so this search window should suffice
start_time = t_depart - search_window
end_time = t_depart + search_window

for t_test in np.arange(start_time, end_time, time_step):
    # Get the parameters for Earth and compute the angle of velocty
    Ex_test, VxE_test, Ey_test, VyE_test = position_from_elements(a_earth, e_earth, omega_earth, M0_earth, t_test, GM_sun)
    thetaE_test = math.atan2(VyE_test, VxE_test)
    # Do the swame for the rocket, but add the velocity of Earth to it as the rocket_positiin_around_earth gives velocity with respect to earth
    x_rE_test, vx_rE_test, y_rE_test, vy_rE_test = rocket_position_around_earth(t_test)
    vx_rocket_test = VxE_test + vx_rE_test
    vy_rocket_test = VyE_test + vy_rE_test
    #We compute the direction of Rocket's velocity
    thetaR_test = math.atan2(vy_rocket_test, vx_rocket_test)
    #We then compute the angle difference and also a dot product to check they are not alligned oppositely
    angle_diff = abs((thetaR_test - thetaE_test) % (2*math.pi))
    if angle_diff > math.pi:
        angle_diff = 2*math.pi - angle_diff
    dot_product = vx_rocket_test * VxE_test + vy_rocket_test * VyE_test #Quadnrant check

    if angle_diff < min_angle_diff and dot_product > 0:
        min_angle_diff = angle_diff
        best_t = t_test
        if angle_diff < math.radians(0.08):  # Its good enough for us if the difference is 0.08
            break


t_depart = best_t
t_arrival = t_depart + 0.96 
print("Final angle difference at departure (degrees):", math.degrees(min_angle_diff))

Ex,VxE ,Ey,VyE = position_from_elements(a_earth, e_earth, omega_earth, M0_earth, t_depart, GM_sun)
#A key point: before solving the Lambert problem and computing a trajectory for transfer 
#We have to escape earth's influenvce of gravity, this is because the Lambert problem ignores Earth's gravitation, which is quite significant at a LEO orbit, so we can't just move from the LEO to Mars using a lambert solver

#Earth's velocity angle at departure
earth_angle = math.atan2(VyE, VxE)
delta_v_escape = v_escape - v_circular
# its bs




#We assume an instaneous burn for simplicity (otherwise we could lose the allignment we aimed and these processes are fast anyways)
# the mass of fuel that needs to be burned is 41875 kg(calculated from rocket equation), which we have available
# based on ISP value 800, delta v = 2757, and m0 = 141000



# The time it takes to escape earth's SOI with such velocity will be around 3.5 days based on my computation
# The sphere of influence is about 900000 km and let us assume an average speed about 3-4 km/s then, the time to reach SOI will be approximately 3.5 days
# but lets just say its 8 days. Its fine, in fact, better if we go beyond 900000 kms
#We will solve an ODE to see where the rocket will be 8 days later and with what speed
# We will solve in Earth frame

x_0 = rocket_radius_m  # Rocket along an arbitrary x axis (This is a new coordinate system with earth at center don't confuse with the heliocentric one  )
y_0 = 0.0
vy_0 = v_circular + delta_v_escape
vx_0 = 0
Ic = [x_0, vx_0, y_0, vy_0]


days = 8 
t_final = days * 24 * 3600 #seconds
t_span = [0.0, t_final]
nsteps = 10000  # number of steps


def f_earth_only(t, vars, params, dfdt):
    GM_earth = params[0]
    x = vars[0]
    vx = vars[1]
    y = vars[2]
    vy = vars[3]
    r3 = (x**2 + y**2)**1.5
    ax = -GM_earth * x / r3
    ay = -GM_earth * y / r3
    dfdt[0] = vx
    dfdt[1] = ax
    dfdt[2] = vy
    dfdt[3] = ay

params_earth = [GM_earth]
t_out, vars_out = solve_ode(f_earth_only, t_span, nsteps, Ic, method="RK4", args=params_earth)
x_r = vars_out[:,0]
vx_r = vars_out[:,1]
y_r = vars_out[:,2]
vy_r = vars_out[:,3]
altitude_escape = np.sqrt(x_r[-1]**2 + y_r[-1]**2) - 6378000
vyRocket = vy_r[-1]
vxRocket = vx_r[-1] 
xRocket = x_r[-1]
yRocket = y_r[-1]

print("After performing the escape, we are at an altitude (km) of ", altitude_escape)

# And we escaped the SOI (well approximately) 
# Note that the arbitary axis y that we set the rocket's motion wrt earth at is alligned with earth's direction motion, so is in the direction arctan(VyE,VxE)


vxRocketAU = vxRocket * m_s_to_au_yr
vyRocketAU = vyRocket * m_s_to_au_yr
xRocketAU = xRocket * m_to_AU
yRocketAU = yRocket * m_to_AU

#Also keep in mind that Earth has moved for 8 days too which is about 0.0219 years
# The angle between Earth and Mars has changed slightly but its ok in literature it said somewhere between 40-45 degrees was ideal
t_depart = t_depart + 0.0219 # we change our departure time by 8 days(remember 8 days was the time it took for us to escape earth)
Ex,VxE ,Ey,VyE = position_from_elements(a_earth, e_earth, omega_earth, M0_earth, t_depart, GM_sun)

# coordinate tranforms (rotation) including reference frame shift to heliocentric
angle = earth_angle # in the heliocentric frame the direction of earth motion
VyRocketHelio = (vyRocketAU * np.sin(angle) + vxRocketAU* np.sin(angle - np.pi/2)) + VyE
VxRocketHelio = (vyRocketAU * np.cos(angle) + vxRocketAU* np.cos(angle - np.pi/2)) + VxE

yRocketHelio = (yRocketAU * np.sin(angle) + xRocketAU* np.sin(angle - np.pi/2)) + Ey
xRocketHelio = (yRocketAU * np.cos(angle) + xRocketAU* np.cos(angle - np.pi/2)) + Ex

# It is a bit tricky but we have switched from the coordinates in Earth's frame (our choice was also rotated wrt to the one we use in sun's frame) to sun's frame performing both a boost and a rotation, if you will 
# now that we have "escaped" earth we can use the lambert solver to solve for the path that will be taken by

Mx,VxM,My,VyM = position_from_elements(a_mars, e_mars, omega_mars, M0_mars, t_arrival, GM_sun)
year_to_sec = 3.15576e7 # as lambert solver takes SI

dt = (t_arrival - t_depart)*year_to_sec

# Lambert transfer from Rocket position to Mars
r1 = np.array([xRocketHelio*AU, yRocketHelio*AU, 0.0]) 
r2 = np.array([Mx*AU, My*AU, 0.0])

l = lambert_problem(r1, r2, dt, MU_SUN)
v1 = l.get_v1() # in m/s
v2 = l.get_v2() 


V_rocket_initial = np.array([VxRocketHelio, VyRocketHelio, 0.0])
V_needed_ms = np.array(v1).flatten()
V_needed_AU = m_s_to_au_yr * V_needed_ms
delv_toreach_mars = V_needed_AU - V_rocket_initial
delv_toreach_mars_mpersec = (V_needed_AU - V_rocket_initial)/ m_s_to_au_yr
mag_delv_mars = np.linalg.norm(delv_toreach_mars_mpersec) # about 3095 meters per second is the change in velocity needed to go to mars
# once again I will assume that we can make such a velocity change instantaneously
# so: 



Isp = 800

def required_fuel(m0, Isp, g0, delta_v):
    # Compute final mass after delta_v
    mf = m0 * math.exp(-delta_v/(Isp*g0))
    m_fuel = m0 - mf
    return m_fuel, mf

m0 = 141610.0 - 41875  # Current mass of rocket is the one we were at when we were orbitting around earth minus the mass of the fuel burned to escape earth
g0 = 9.81              


m_fuel_needed, mf = required_fuel(m0, Isp, g0, mag_delv_mars)
print("By the time we reach to Mars' location we will have a final mass of", mf )
# the direction we will burn the fuel will be determined by delv_toreach_mars's direction but we won't worry about that and just assume we burn in the right direction instantenously and reach the desired velocity
# this - "angle = math.atan2(delv_toreach_mars[1], delv_toreach_mars[0]" would determine the direction but we dont need 

# Applying the instanteonus impulse we will achieve the needed velocity

v_rocket_vector = V_needed_AU # in AU

# We will now simulate the trajectory, taking into account forces due Earth, Mars, Jupiter, and Sun, seeing how the unaccounted bodies(all except sun) perturb the expected trajectory
#For jupiter I have taken the same epoch J2000 as with Mars and Earth and we assume everything is co-planar once again
# When I set GM_e, GM_m, GM_j, to 0 rocket reaches mars exactly
#Parameters for Jupiter which I will model as circular with the radius being a_j

w_j = 2*math.pi/11.86
a_j =  5.2044 
angle_j0 = 1.755 
t0 = t_depart

GM_s = 4*math.pi**2
GM_e = 1.19e-4
GM_m = 1.3e-5
GM_j = 0.0377

params_mars = np.array([GM_s, GM_e, GM_m, GM_j,
                        a_earth, e_earth, omega_earth, M0_earth,
                        a_mars, e_mars, omega_mars, M0_mars,
                        a_j, w_j, angle_j0, t0], dtype=np.double)

def f(t, vars, params_mars, dfdt):

    #params = [GM_s, GM_e, GM_m, GM_j, a_e, w_e, a_m, w_m, a_j, w_j, angle_j0, t0]
    #
    # angle_j0 = Jupiter's angle at J2000 (rad)
    # t0   = time offset, so actual time since J2000 is T = t0 + t

    GM_s = params_mars[0]
    GM_e = params_mars[1]
    GM_m = params_mars[2]
    GM_j = params_mars[3]
    a_earth = params_mars[4]
    e_earth = params_mars[5]
    omega_earth = params_mars[6]
    M0_earth = params_mars[7]
    a_mars = params_mars[8]
    e_mars = params_mars[9]
    omega_mars = params_mars[10]
    M0_mars = params_mars[11]
    a_j = params_mars[12]
    w_j = params_mars[13]
    angle_j0 = params_mars[14]
    t0 = params_mars[15]

    T = t0 + t
    x = vars[0]
    vx = vars[1]
    y = vars[2]
    vy = vars[3]

    Ex, VxE, Ey, VyE = position_from_elements(a_earth, e_earth, omega_earth, M0_earth, T, GM=GM_s)
    Mx, VxM, My, VyM = position_from_elements(a_mars, e_mars, omega_mars, M0_mars, T, GM=GM_s)

    x_j = a_j * math.cos(w_j * T + angle_j0)
    y_j = a_j * math.sin(w_j * T + angle_j0)

    r_s3 = (x**2 + y**2)**1.5
    r_e3 = ((x - Ex)**2 + (y - Ey)**2)**1.5
    r_m3 = ((x - Mx)**2 + (y - My)**2)**1.5
    r_j3 = ((x - x_j)**2 + (y - y_j)**2)**1.5

    dfdt[0] = vx
    dfdt[2] = vy

    ax = (-GM_s * x / r_s3
          - GM_e*(x - Ex)/r_e3
          - GM_m*(x - Mx)/r_m3
          - GM_j*(x - x_j)/r_j3)

    ay = (-GM_s * y / r_s3
          - GM_e*(y - Ey)/r_e3
          - GM_m*(y - My)/r_m3
          - GM_j*(y - y_j)/r_j3)

    dfdt[1] = ax
    dfdt[3] = ay

#Initial conditions after final burn:
x_0 = xRocketHelio
y_0 = yRocketHelio
vx_0 = v_rocket_vector[0]
vy_0 = v_rocket_vector[1]

Ic = [x_0, vx_0, y_0, vy_0]
t_span = [0.0, (t_arrival - t_depart)] # running a little longer to see if it is better considering the gravitational perturbations from planets
nsteps = 100000

t_out, vars_out = solve_ode(f, t_span, nsteps, Ic, method="RK4", args=params_mars)

x_r = vars_out[:,0]
y_r = vars_out[:,2]

t_arr_full = t0 + t_out
earth_x_arr = []
earth_y_arr = []
mars_x_arr = []
mars_y_arr = []
jup_x_arr = []
jup_y_arr = []

for T_ in t_arr_full:
    Ex_, VxE_, Ey_, VyE_ = position_from_elements(a_earth, e_earth, omega_earth, M0_earth, T_, GM=GM_s)
    Mx_, VxM_, My_, VyM_ = position_from_elements(a_mars, e_mars, omega_mars, M0_mars, T_, GM=GM_s)
    earth_x_arr.append(Ex_)
    earth_y_arr.append(Ey_)
    mars_x_arr.append(Mx_)
    mars_y_arr.append(My_)

    x_jup = a_j * math.cos(w_j * T_ + angle_j0)
    y_jup = a_j * math.sin(w_j * T_ + angle_j0)
    jup_x_arr.append(x_jup)
    jup_y_arr.append(y_jup)

plt.figure()
plt.plot(earth_x_arr, earth_y_arr, label='Earth')
plt.plot(mars_x_arr, mars_y_arr, label='Mars')
plt.plot(x_r, y_r, label='Rocket')
plt.xlabel('X (AU)')
plt.ylabel('Y (AU)')
plt.title('Interplanetary Trajectory')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

import matplotlib.animation as animation


# Time to simulate the trajectoryy!!!

step = 1000  # already chosen to speed up rendering
frame_indices = range(0, len(t_arr_full), step)

# We determine plot limits including Jupiter too
x_min = min(min(earth_x_arr), min(mars_x_arr), min(jup_x_arr), min(x_r))
x_max = max(max(earth_x_arr), max(mars_x_arr), max(jup_x_arr), max(x_r))
y_min = min(min(earth_y_arr), min(mars_y_arr), min(jup_y_arr), min(y_r))
y_max = max(max(earth_y_arr), max(mars_y_arr), max(jup_y_arr), max(y_r))

fig, ax = plt.subplots(figsize=(6,5))
ax.set_aspect('equal', 'box')
ax.set_xlim(x_min - 0.5, x_max + 0.5)
ax.set_ylim(y_min - 0.5, y_max + 0.5)
ax.set_xlabel('X (AU)')
ax.set_ylabel('Y (AU)')
ax.set_title('Interplanetary Trajectory (Animation)')

sun_plot, = ax.plot([0], [0], 'o', color='yellow', markersize=10, label='Sun')
earth_plot, = ax.plot([], [], 'o', color='blue', label='Earth')
mars_plot, = ax.plot([], [], 'o', color='red', label='Mars')
jupiter_plot, = ax.plot([], [], 'o', color='orange', label='Jupiter')
rocket_plot, = ax.plot([], [], 'o', color='green', label='Rocket')
rocket_path, = ax.plot([], [], '-', color='green', alpha=0.5)

ax.legend()

def init():
    earth_plot.set_data([], [])
    mars_plot.set_data([], [])
    jupiter_plot.set_data([], [])
    rocket_plot.set_data([], [])
    rocket_path.set_data([], [])
    return earth_plot, mars_plot, jupiter_plot, rocket_plot, rocket_path

def update(frame):
    i = frame
    earth_plot.set_data([earth_x_arr[i]], [earth_y_arr[i]])
    mars_plot.set_data([mars_x_arr[i]], [mars_y_arr[i]])
    jupiter_plot.set_data([jup_x_arr[i]], [jup_y_arr[i]])
    rocket_plot.set_data([x_r[i]], [y_r[i]])
    rocket_path.set_data(x_r[:i], y_r[:i])
    return earth_plot, mars_plot, jupiter_plot, rocket_plot, rocket_path

ani = animation.FuncAnimation(fig, update, frames=frame_indices, init_func=init, interval=20, blit=False, repeat=False)

Writer = animation.FFMpegWriter

writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

ani.save('trajectory.mp4', writer=writer, dpi=100)
plt.close(fig)
print("saved to 'trajectory.mp4'.")

# If you look at the trajectory with perturbations giving "small" kicks at the right time can help fall into mars. When its at its closest distance I believe if you decelerate the rocket, you will fall to mars
# But we will not simulate the landing.
# final mass remaining
print("at the end of the transfer simulation the final mass remaining:", mf)


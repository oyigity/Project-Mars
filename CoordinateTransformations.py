#Yigit Yazgan

# This code is directly taken from Assignment 6 question 2, and is used because it is proven to work
import math


def ellipse_to_xy(a, e, theta, theta_E, GM_S= 4* math.pi **2):

    # Angular momentum is given by
    h = math.sqrt(GM_S * a * (1 - e**2))

    cos_theta_diff = math.cos(theta - theta_E)
    r = a * (1 - e**2) / (1 + e * cos_theta_diff)

    x = r * math.cos(theta)
    y = r * math.sin(theta)
    # The magnitude of velocity is given  by:
    v = math.sqrt(GM_S * (2 / r - 1 / a))

    sin_alpha_minus_theta = h / (r * v)

    # the value must be within 1 and -1 so

    sin_alpha_minus_theta = max(-1.0, min(1.0, sin_alpha_minus_theta))

    sin_theta_diff = math.sin(theta - theta_E)

    # the radial velocity is given by:

    vr = (GM_S / h) * e * sin_theta_diff # the radial velocity is important in determining what quadrant will arcsin of alpha minus theta will give

    # that is :

    if vr >= 0:
        # particle is moving away from the Sun
        alpha_minus_theta = math.asin(sin_alpha_minus_theta)
    else:
        # Particle is moving towards the Sun
        alpha_minus_theta = math.pi - math.asin(sin_alpha_minus_theta)

    alpha = theta + alpha_minus_theta

    # to ensure alpha is between 0 and 2Pi we can use the remainder logic

    alpha = alpha % (2 * math.pi)

    vx = v * math.cos(alpha)
    vy = v * math.sin(alpha)

    return x,vx,y,vy


def xy_to_ellipse(x,vx,y,vy, GM_S= 4* math.pi **2):
    # let us express the position as a vector
    r_vec = [x, y]
    # same with velocity
    v_vec = [vx, vy]
    # getting the magnitude with math.hypot
    r = math.hypot(x, y)
    v = math.hypot(vx, vy)

     # angular momentum in cartesian coordinates is given by:

    h = x * vy - y * vx 

    # the energy density is given by 

    u  =  0.5 * v**2 - GM_S / r
    # variable a can be given in terms of u

    a = -GM_S / (2 * u)
    # using the angular momentum equation we can give a
            
    e = math.sqrt(1 - (h**2) / (GM_S * a))

    theta_absolute = math.atan2(y, x) % (2 * math.pi) # this is relative to the x axis ( so the value of actual theta when thetaE = 0)

    # to get information about the location of the periapsis we can use the concept of the eccentricity vector
    ex = ( (v**2 - GM_S / r) * x - (x * vx + y * vy) * vx ) / GM_S
    ey = ( (v**2 - GM_S / r) * y - (x * vx + y * vy) * vy ) / GM_S
    e_vec_magnitude = math.hypot(ex, ey)
    # using the concept of eccentricity vector allows us not to deal with the sign ambiguities. We do not get any relative angle from arc cos. Instead we compute thetaE and theta separately!
        
    # QUICK CHECK IF e = e_vec_magnitude

    
            
     # we can now use the components of eccentricity to determine thetaE

    theta_E = math.atan(ey/ex) % (2 * math.pi)

    if e > 1e-5: # for numerical stability theta_E should be set to 0 (any arbitrary value would work)for small enough eccentricity. 1e-5 was determined through trial and error and understanding the numerical instability in our system
    # this is no probem as for small eccenricity, orbit gets closer to circular and it the variable theta_E loses its meaning.
        # Eccentricity vector is well-defined
        theta_E = math.atan2(ey, ex) % (2 * math.pi)
    else:
        # Circular orbit: set theta_E to zero
        theta_E = 0.0

    

    return a, e, theta_absolute, theta_E
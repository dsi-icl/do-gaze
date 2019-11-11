import numpy as np
import math

kinect_position = (x, y, z)

# Information about the GDO geometry
# Vision span is 313 degrees
r = 3
theta_node_1 = 3*313*pi/(180*16)
theta_node_2 = 313*pi/(180*4)


# Solving the quadratic equation related to the gaze position on the GDO
a = w[0]**2 + w[1]**2
b = 2*(w[0]*joint[0] + w[1]*joint[1])
c = joint[0]**2 + joint[1]**2 - r**2

d = b**2 - 4*a*c

if d<0:
    print("There has been an error")
    m = None
elif d == 0:
    m = -b/(2*a)
else:
    m_1 = (-b - math.sqrt(d)) / (2*a)
    m_2 = (-b + math.sqrt(d)) / (2*a)
    


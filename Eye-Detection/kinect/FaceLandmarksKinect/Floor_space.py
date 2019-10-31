from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np
import cv2

import ctypes
import _ctypes
import sys

import math

class Floor:
        
    # Automatically connects to the Kinect V2.
	# If autoFloorPlane is true, the Kinect grabs
    # the first available floor clip plane from the Kinect body.
    def __init__(self, bodyFrame, autoFloorPlane = True):
        self.floorPlane = np.array([bodyFrame.floor_clip_plane.x,
                                   bodyFrame.floor_clip_plane.y,
                                   bodyFrame.floor_clip_plane.z,
                                   bodyFrame.floor_clip_plane.w])
        if autoFloorPlane:
            self.kinect_tilt = math.atan(self.floorPlane[2]/self.floorPlane[1])
            self.R = np.array([[1,0,0], [0, math.cos(self.kinect_tilt), math.sin(self.kinect_tilt)], [0, -math.sin(self.kinect_tilt), math.cos(self.kinect_tilt)]])
        else:
            # By default, the floor space is the same as the camera space
            print("We have a floor detection problem")

    def point_to_transform_space(self, point):
        return np.dot(self.R, point)

    def face_direction(self):
        v = np.array([0,0,-1])
        u = np.array([self.floorPlane[0], self.floorPlane[1], self.floorPlane[2]])
        s = self.floorPlane[3]
        vprime = 2*np.dot(u,v)*u + (s*s - np.dot(u,u))*v + 2*s*np.cross(u,v)
        if (vprime[0] == 0 and vprime[1] == 0 and vprime[2] == 0):
            return None
        else:
            return vprime



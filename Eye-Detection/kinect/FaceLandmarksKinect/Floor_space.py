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

 # Gets the first detected floor plane from the body frame, freezing the kinect
    # until it is available.
    def _get_kinect_tilt(self, bodyFrame):

        # Save floor plane as [a,b,c,d], where ax+by+cz+d=0
        floorPlane = np.array([bodyFrame.floor_clip_plane.x,
                                   bodyFrame.floor_clip_plane.y,
                                   bodyFrame.floor_clip_plane.z,
                                   bodyFrame.floor_clip_plane.w])
		
        return math.atan(floorPlane[2]/floorPlane[1])
        
    # Automatically connects to the Kinect V2.
	# If autoFloorPlane is true, the Kinect grabs
    # the first available floor clip plane from the Kinect body.
    def __init__(self, bodyFrame, autoFloorPlane = True):
        # Get a reference to the Kinect sensor
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                      PyKinectV2.FrameSourceTypes_Depth |
                                                      PyKinectV2.FrameSourceTypes_Body)
        if autoFloorPlane:
            self.kinect_tilt = self._get_kinect_tilt(bodyFrame)
        else:
            # By default, the floor space is the same as the camera space
            print("We have a floor detection problem")

    def point_to_transform_space(self, point):
        theta = self.kinect_tilt
        R = np.array([[1,0,0], [0, math.cos(theta), math.sin(theta)], [0, -math.sin(theta), math.cos(theta)]])
        return np.dot(R, point)



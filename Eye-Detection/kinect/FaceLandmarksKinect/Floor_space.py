from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import numpy as np
import cv2

import ctypes
import _ctypes
import sys

import math
 

# Handles transforming between spaces
class SpaceTransform:
    # All vectors and planes of this class are normalized.
    # All the following np arrays pertain 
    # to the transformed space:
    # 	xy_plane: the normalized xy-plane
    # 	x_axis: vector in x_axis direction
    def __init__(self,
                 xy_plane = np.array([0,0,1,0]),
                 x_axis =  np.array([1,0,0])):

        # Plane origin is where the origin is in the untransformed space
        self.plane_origin = np.array([0,0,0])
        
        self.xy_plane = xy_plane
        self.x_axis =  x_axis/np.linalg.norm(x_axis)

        # Y-axis is automatically calculated from normal and x-axis
        self.y_axis = np.cross(self.xy_plane[0:3], self.x_axis)

    # Creates the transform space from 3 points:
    # origin: the new origin
    # on_x_axis: any point, except the origin, along the x-axis
    # y_pos:  any point in the positive y direction
    def create_from_3_points(self, origin, on_x_axis, y_pos):
        # Set the untranformed origin
        self.plane_origin = origin

        # Find the x-axis vector, normalize it
        self.x_axis = on_x_axis-origin
        self.x_axis = self.x_axis/np.linalg.norm(self.x_axis)
                       
        # Find the normal, normalize it
        n = np.cross(self.x_axis,y_pos-origin)  
        n = n/np.linalg.norm(n)

        # Find the xy plane
        self.xy_plane = np.array([n[0],n[1],n[2], -np.dot(n,origin)])

        #Find the new y-axis
        self.y_axis = np.cross(n, self.x_axis)

    # Transforms a point such that it can be referenced from the
    # coordinate system described by x-axis, in_y, and xy_plane.
    # A 1x3 numpy array is returned, or if the value is not finite,
    # None is returned
    def point_to_transform_space(self, point):
        # extract the norm of the ground plane,
        # The magnitude of normal is already 1 as per kinect docs.
        normal = self.xy_plane[0:3]

        # Find the distance from the point to the plane
        distancePP = np.dot(normal,point)+self.xy_plane[3]

        if (not math.isnan(distancePP)):
            # Find the position closest to the point on the xy plane
            planePos = point - distancePP*normal

            # Distance from the origin to the point's position on the plane 
            posVector = planePos - self.plane_origin

            # Calculate the floor position
            floorPos = np.array([np.dot(self.x_axis, posVector),
                                 np.dot(self.y_axis, posVector),
                                 distancePP])
            return floorPos
        else:
            return None

class Floor:

 # Gets the first detected floor plane from the body frame, freezing the kinect
    # until it is available.
    def _get_floor_plane_from_body_frame(self, bodyFrame):

        # Save floor plane as [a,b,c,d], where ax+by+cz+d=0
        floorPlane = np.array([bodyFrame.floor_clip_plane.x,
                                   bodyFrame.floor_clip_plane.y,
                                   bodyFrame.floor_clip_plane.z,
                                   bodyFrame.floor_clip_plane.w])
		
		# Determine the floor x-axis: [1,0,-a/c]
        floorXaxis = np.array([1,0,-floorPlane[0]/floorPlane[2]])
		
        floorTransform = SpaceTransform(floorPlane, floorXaxis)
		
        return floorTransform
        
    # Automatically connects to the Kinect V2.
	# If autoFloorPlane is true, the Kinect grabs
    # the first available floor clip plane from the Kinect body.
    def __init__(self, bodyFrame, autoFloorPlane = True):
        # Get a reference to the Kinect sensor
        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color |
                                                      PyKinectV2.FrameSourceTypes_Depth |
                                                      PyKinectV2.FrameSourceTypes_Body)
        if autoFloorPlane:
            self.floorTransform = self._get_floor_plane_from_body_frame(bodyFrame)
        else:
            # By default, the floor space is the same as the camera space
            self.floorTransform = SpaceTransform()



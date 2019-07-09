import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from acquisitionKinect import AcquisitionKinect
from frame import Frame
import face_alignment
from skimage import io
from rotate.rotation import Rotation3d
import json
import websocket

ws = websocket.WebSocket()
ws.connect("wss://gdo-gaze.dsi.ic.ac.uk")

"""
Functions
"""

"""
Rotations
"""
def rotation_x(v, alpha):
    R = np.array([[1, 0, 0], [0, np.cos(alpha), np.sin(alpha)], [0, -np.sin(alpha), np.cos(alpha)]])
    return np.dot(R, v)

"""
Norm of a vector
"""
def normv(v):
    return(v / np.linalg.norm(v))


if __name__ == '__main__':

	kinect = AcquisitionKinect()
	frame = Frame()
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cuda")


	while True:
		kinect.get_frame(frame)
		image = kinect._frameRGB
		frameDepth = kinect._frameDepth

		

		#OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

		preds = fa.get_landmarks(image)[-1]

		x_ = preds[45,:] - preds[36,:]
		y_ = preds[27,:] - preds[51,:]
		w_ = np.cross(x_, y_)

		if w_[2] < 0:
			w_ = w_*(-1)

		left_eye = (preds[45,:] + preds[42,:])//2
		right_eye = (preds[39,:] + preds[36,:])//2
		
		end_line_left = list(map(int, (left_eye + 300 * w_)))
		end_line_right = list(map(int, (right_eye + 300 * w_)))
		line_left = np.array([left_eye,end_line_left])
		line_right = np.array([right_eye, end_line_right])

		#cv2.line(image, (left_eye[0], left_eye[1]), (end_line_left[0], end_line_left[1]),
		#(255, 0, 0), 2)
		#cv2.line(image, (right_eye[0], right_eye[1]), (end_line_right[0], end_line_right[1]),
		#(255, 0, 0), 2)

		#cv2.circle(image, (right_eye[0], right_eye[1]), 2, (255,0,0), 0)
		cv2.circle(image, (left_eye[0], left_eye[1]), 2, (0,255,0), 0)
		#cv2.circle(image, (preds[27,0], preds[27,1]), 2, (0,0,255), 0)
		#cv2.circle(image, (preds[8,0], preds[8,1]), 2, (0,255,255), 0)


		scale = np.array([512/1920, 424/1080])

		depth = np.zeros([424,512,3])

		for i in range(424):
			for j in range(512):
				depth[i,j] = (frameDepth[i,j],frameDepth[i,j],255)

		"""
		Points of interest from 2D picture to 3D space coordinates
		Projection of the gaze on the screens
		
		"""
		# w,x,y,z are coordinates on the depth frame whereas w_d, x_d, y_d, z_d are the depth values of these coordinates
		w, x, y, z = np.array([preds[36,0]*scale[0], preds[36,1]*scale[1]]), np.array([preds[45,0]*scale[0], preds[45,1]*scale[1]]), np.array([preds[27,0]*scale[0], preds[27,1]*scale[1]]), np.array([preds[51,0]*scale[0], preds[51,1]*scale[1]])
		w_d, x_d, y_d, z_d = frameDepth[int(w[1]), int(w[0])], frameDepth[int(x[1]), int(x[0])], frameDepth[int(y[1]), int(y[0])], frameDepth[int(z[1]), int(z[0])]

		right_eye = np.array([right_eye[0]*scale[0], right_eye[1]*scale[1]])
		right_eye_d = frameDepth[int(right_eye[1]), int(right_eye[0])]

		left_eye = np.array([left_eye[0]*scale[0], left_eye[1]*scale[1]])
		left_eye_d = frameDepth[int(left_eye[1]), int(left_eye[0])]

		depthpoints = np.array([w,x,y,z,right_eye, left_eye])
		depths = np.array([w_d,x_d,y_d,z_d,right_eye_d, left_eye_d])
		face_land = kinect.acquireCameraSpace(depthpoints, depths)

		x_0 = face_land[0:3]
		x_1 = face_land[3:6]
		y_0 = face_land[6:9]
		y_1 = face_land[9:12]

		right_eye_s = face_land[12:15]
		left_eye_s = face_land[15:18]

		

		x_s = normv(x_1 - x_0)
		y_s = normv(y_0 - y_1)
		z_s = np.cross(x_s, y_s)

		#Correction of the director vector
		#z_s = rotation_x(z_s,0.3)
		

		if z_s[2] > 0:
			z_s = z_s * (-1)

		k = - right_eye_s[2] / (z_s[2])

		cible = right_eye_s + k*z_s

		print("cible", cible)

		data_point = {
			"x": cible[0],
			"y": cible[1],
			"z": cible[2]
		}

		message = json.dumps(data_point, separators=(',', ':'))

		ws.send(message)	

		if not image is None:
			cv2.imshow("Output-Keypoints",image)

		key = cv2.waitKey(1)
		if key == 27:
			ws.close()
			break
		

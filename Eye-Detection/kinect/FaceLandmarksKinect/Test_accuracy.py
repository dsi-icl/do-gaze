import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from acquisitionKinect import AcquisitionKinect
from Floor_space import Floor
from frame import Frame
import face_alignment
from skimage import io
import json
import math
import websocket
import pandas as pd

ws = websocket.WebSocket()
ws.connect("wss://gdo-gaze.dsi.ic.ac.uk")

class Mov_av:
	def __init__(self):
		self.p0 = pd.DataFrame([], columns=['x', 'y', 'z'])
		self.p1 = pd.DataFrame([], columns=['x', 'y', 'z'])
		self.p2 = pd.DataFrame([], columns=['x', 'y', 'z'])
		self.p3 = pd.DataFrame([], columns=['x', 'y', 'z'])
		self.p4 = pd.DataFrame([], columns=['x', 'y', 'z'])
		self.p5 = pd.DataFrame([], columns=['x', 'y', 'z'])

	"""
	Dataframes of moving average positions for each face
	"""
	
	def associate(self, nb, x, y, z):
		if nb == '0':
			print('Coucou0')
			self.p0 = self.p0.append({'x': x, 'y':y, 'z':z}, ignore_index=True) 
		if nb == '1':
			print('Coucou1')
			self.p1 = self.p1.append({'x': x, 'y':y, 'z':z}, ignore_index=True) 
		if nb == '2':
			print('Coucou2')
			self.p2 = self.p2.append({'x': x, 'y':y, 'z':z}, ignore_index=True) 
		if nb == '3':
			print('Coucou3')
			self.p3 = self.p3.append({'x': x, 'y':y, 'z':z}, ignore_index=True) 
		if nb == '4':
			print('Coucou4')
			self.p4 = self.p4.append({'x': x, 'y':y, 'z':z}, ignore_index=True) 
		if nb == '5':
			print('Coucou5')
			self.p5 = self.p5.append({'x': x, 'y':y, 'z':z}, ignore_index=True) 
	
	def get_mean(self, msg):
		if len(self.p0) == 3:
			msg.loc['p0'] = self.p0.mean()
			self.p0.drop([0], inplace=True)
			self.p0.reset_index(drop=True)
		if len(self.p1) == 3:
			msg.loc['p1'] = self.p1.mean()
			self.p1.drop([0], inplace=True)
			self.p1.reset_index(drop=True)
		if len(self.p2) == 3:
			msg.loc['p2'] = self.p2.mean()
			self.p2.drop([0], inplace=True)
			self.p2.reset_index(drop=True)
		if len(self.p3) == 3:
			msg.loc['p3'] = self.p3.mean()
			self.p3.drop([0], inplace=True)
			self.p3.reset_index(drop=True)
		if len(self.p4) == 3:
			msg.loc['p4'] = self.p4.mean()
			self.p4.drop([0], inplace=True)
			self.p4.reset_index(drop=True)
		if len(self.p5) == 3:
			msg.loc['p5'] = self.p5.mean()
			self.p5.drop([0], inplace=True)
			self.p5.reset_index(drop=True)
	
		return msg


# msg will contain all the gazes to send on the server
mov_ave = Mov_av()
"""
Compute minimal euclidean distance to link a skeleton to a face
"""

def face_number(list_skeleton, nose_s):
	min_ = None
	nb_skel = len(list_skeleton)
	for i in range(nb_skel):
		distance = np.linalg.norm(list_skeleton[i] - nose_s)
		if min_ is None:
			min_ = i
			distance_m = distance
		elif distance < distance_m:
			min_ = i
			distance_m = distance
	return min_, distance_m

"""
Check same length for list of bodies and list of faces
"""
def remove_dop(df):
    last = df.iterrows()
    df2 = df
    for _, row in last:
        last2 = df2.iterrows()
        for ind2, row2 in last2:
            if row[3] == row2[3]:
                if row[4] < row2[4]:
                    df2.drop(ind2, axis=0, inplace=True)
    return df2

if __name__ == '__main__':

	kinect = AcquisitionKinect()
	frame = Frame()
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda")
	msg = pd.DataFrame([], index=['p0', 'p1', 'p2', 'p3', 'p4', 'p5'], columns=['x', 'y', 'z'])


	while True:
		data_cible = pd.DataFrame([], columns=["x", "y", "z", "number", "distance"])
		kinect.get_frame(frame)
		kinect.get_color_frame()
		image = kinect._frameRGB
		frameDepth = kinect._frameDepth
		kinect.get_eye_camera_space_coord()
		joint = kinect.joint_points3D
		print("List of bodies", joint)
		CameraPoints = kinect.cameraPoints
		floor = Floor(kinect._bodies)


		#OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

		# Add movement sensor here (ie when the head doesn't move, don't use get_landmarks)
		# Use moving average

		preds = fa.get_landmarks(image)
		nb_detected = len(preds)

		try:
			assert len(joint) == nb_detected
		except:
			print("Error between the number of faces detected and the number of skeletons")

		for k in range(nb_detected):
			# draw all faces
			# for i in range(68):
			# 	cv2.circle(image, (preds[k][i,0], preds[k][i,1]), 3, (255, 0, 0), -1)
			
			# The right eye is defined by being the centor of two landmarks
			# right_eye = (preds[k][45,:] + preds[k][42,:])//2

			# 
			nose_s = np.array([CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][0], CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][1], CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][2]])
			face_nb, distance = face_number(joint, nose_s)
			
			x_0_f = np.array([CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][0], CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][1], CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][2]])
			x_0 = floor.point_to_transform_space(np.array([CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][0], CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][1], CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][2]]))
			x_1 = floor.point_to_transform_space(np.array([CameraPoints[int(preds[k][45,1]), int(preds[k][45,0])][0], CameraPoints[int(preds[k][45,1]), int(preds[k][45,0])][1], CameraPoints[int(preds[k][45,1]), int(preds[k][45,0])][2]]))
			x_1_2 = floor.point_to_transform_space(np.array([CameraPoints[int(preds[k][42,1]), int(preds[k][42,0])][0], CameraPoints[int(preds[k][42,1]), int(preds[k][42,0])][1], CameraPoints[int(preds[k][42,1]), int(preds[k][42,0])][2]]))
			y_0 = floor.point_to_transform_space(np.array([CameraPoints[int(preds[k][51,1]), int(preds[k][51,0])][0], CameraPoints[int(preds[k][51,1]), int(preds[k][51,0])][1], CameraPoints[int(preds[k][51,1]), int(preds[k][51,0])][2]]))
			y_1 = floor.point_to_transform_space(np.array([CameraPoints[int(preds[k][27,1]), int(preds[k][27,0])][0], CameraPoints[int(preds[k][27,1]), int(preds[k][27,0])][1], CameraPoints[int(preds[k][27,1]), int(preds[k][27,0])][2]]))

			print('Before transformation', x_0_f, 'After transformation', x_0)
			print("Mapper says here", x_0, x_1)
			x_s = x_1 - x_0
			x_s = x_s/(np.linalg.norm(x_s))
			y_s = y_1 - y_0
			y_s = y_s/(np.linalg.norm(y_s))
			z_s = np.cross(x_s, y_s)

			left_eye_s = (x_1+x_1_2)//2

			if z_s[2] > 0:
				z_s = z_s * (-1)

			k = - left_eye_s[2]/z_s[2]

			case = len(joint)
			if case > 0:
				print("Your face is here", joint)

			cible = left_eye_s + k*z_s

			print("cible", cible)

			data_cible = data_cible.append({"x":cible[0], "y":cible[1], "z":cible[2], "number":str(face_nb), "distance":distance}, ignore_index=True)

		data_cible.dropna(inplace=True)
		data_copy = data_cible.set_index('number')
		try:
			data_copy.to_json(orient='index')
		except ValueError:
			remove_dop(data_cible)
		for ind, val in data_cible.iterrows():
			print(val['number'], val['x'], val['y'], val['z'])
			mov_ave.associate(val['number'], val['x'], val['y'], val['z'])
		print("p0", mov_ave.p0)

		msg = mov_ave.get_mean(msg)
		print("msg", msg)
		message = msg.to_json(orient='index')
		

		ws.send(message)

		
		if not image is None:
			cv2.imshow("Output-Keypoints",image)

		key = cv2.waitKey(1)
		if key == 27:
			ws.close()
			break
		

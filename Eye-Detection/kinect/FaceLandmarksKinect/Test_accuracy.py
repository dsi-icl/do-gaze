import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from acquisitionKinect import AcquisitionKinect
from frame import Frame
import face_alignment
from skimage import io
import json
import websocket
import pandas as pd

ws = websocket.WebSocket()
ws.connect("wss://gdo-gaze.dsi.ic.ac.uk")


if __name__ == '__main__':

	kinect = AcquisitionKinect()
	frame = Frame()
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda")


	while True:
		data_cible = pd.DataFrame([], columns=["x", "y", "z"])
		kinect.get_frame(frame)
		kinect.get_color_frame()
		image = kinect._frameRGB
		frameDepth = kinect._frameDepth
		kinect.get_eye_camera_space_coord()
		joint = kinect.joint_points3D
		CameraPoints = kinect.cameraPoints


		#OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
		# scale = np.array([512/1920, 424/1080])

		# image = cv2.resize(image, None, fx = scale[0], fy = scale[1])

		print(image.shape)

		preds = fa.get_landmarks(image)
		nb_detected = len(preds)

		for k in range(nb_detected):
			for i in range(68):
				cv2.circle(image, (preds[k][i,0], preds[k][i,1]), 3, (255, 0, 0), -1)
			left_eye = (preds[k][45,:] + preds[k][42,:])//2
			cv2.circle(image, (left_eye[0], left_eye[1]), 2, (0,255,0), 0)
			print("Mapper says here", CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])], CameraPoints[int(preds[k][49,1]), int(preds[k][49,0])])
			x_0 = np.array([CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][0], CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][1], CameraPoints[int(preds[k][36,1]), int(preds[k][36,0])][2]])
			x_1 = np.array([CameraPoints[int(preds[k][45,1]), int(preds[k][45,0])][0], CameraPoints[int(preds[k][45,1]), int(preds[k][45,0])][1], CameraPoints[int(preds[k][45,1]), int(preds[k][45,0])][2]])
			x_1_2 = np.array([CameraPoints[int(preds[k][42,1]), int(preds[k][42,0])][0], CameraPoints[int(preds[k][42,1]), int(preds[k][42,0])][1], CameraPoints[int(preds[k][42,1]), int(preds[k][42,0])][2]])
			y_0 = np.array([CameraPoints[int(preds[k][51,1]), int(preds[k][51,0])][0], CameraPoints[int(preds[k][51,1]), int(preds[k][51,0])][1], CameraPoints[int(preds[k][51,1]), int(preds[k][51,0])][2]])
			y_1 = np.array([CameraPoints[int(preds[k][27,1]), int(preds[k][27,0])][0], CameraPoints[int(preds[k][27,1]), int(preds[k][27,0])][1], CameraPoints[int(preds[k][27,1]), int(preds[k][27,0])][2]])

			x_s = x_1 - x_0
			x_s = x_s/(np.linalg.norm(x_s))
			y_s = y_1 - y_0
			y_s = y_s/(np.linalg.norm(y_s))
			print(x_s, y_s)
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
			
			data_point = {
				"x": cible[0],
				"y": cible[1],
				"z": cible[2]
			}

			data_cible = data_cible.append({"x":cible[0], "y":cible[1], "z":cible[2]}, ignore_index=True)

		data_cible.set_index([['p'+str(i) for i in range(nb_detected)]], inplace=True)
		data_cible.dropna(inplace=True)
		message = data_cible.to_json(orient='index')

		ws.send(message)

		
		if not image is None:
			cv2.imshow("Output-Keypoints",image)

		key = cv2.waitKey(1)
		if key == 27:
			ws.close()
			break
		

import cv2
#import dlib
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

"""
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../../shape_predictor_68_face_landmarks.dat')
"""

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
		kinect.get_depth_frame()
		kinect.get_color_frame()
		kinect.get_eye_camera_space_coord()
		image = kinect._frameRGB
		frameDepth = kinect._frameDepth

		#print("###################################################################")
		#print(frameDepth)
		#print("###################################################################")
		#print(np.amax(frameDepth))
		#print("###################################################################")
		#print(len(frameDepth[0]))
		#print("###################################################################")
		#print(len(frameDepth))
		#OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		"""
		faces = detector(gray)

		for face in faces:
			landmarks = predictor(gray, face)

			for n in range(68):
				x = landmarks.part(n).x
				y = landmarks.part(n).y
				cv2.circle(image, (x, y), 4, (255, 0, 0), -1)
		"""
		preds = fa.get_landmarks(image)[-1]

		h = int(np.linalg.norm(preds[45,:] - preds[36,:]))
		v = int(np.linalg.norm(preds[27,:] - preds[51,:]))


		w, x, y, z = preds[36,:], preds[45,:], preds[27,:], preds[51,:]
		wp, xp, yp, zp = [-h//2, 0, 0], [h//2, 0, 0], [0, 100, 0], [0, 100 + v, 0]

		"""
		Projection of the gaze on the screens
		"""

		#rot = Rotation3d(w, x, y, z, wp, xp, yp, zp)
		#R, t = rot.rigid_transform_3D()

		#R_i = np.linalg.inv(R)

		#dir_vector = kinect.joint_points3D
		#dir_vector_norm = np.linalg.norm(dir_vector)
		#dir_vector = np.dot(R_i, dir_vector/dir_vector_norm)

		#k = -kinect.joint_points3D[2]/(dir_vector)

		#cible = kinect.joint_points3D + k*dir_vector

		#print("Draw a point here !!", cible)

		


		x = preds[45,:] - preds[36,:]
		y = preds[27,:] - preds[51,:]

		u = normv(x)
		v = normv(y)
		w = np.cross(u,v)

		if w[2] < 0:
			w = w * (-1)

		corrected_w = rotation_x(w, 0.3)

		left_eye = (preds[45,:] + preds[42,:])//2
		right_eye = (preds[39,:] + preds[36,:])//2
		end_line_left = list(map(int, (left_eye + 300 * corrected_w)))
		end_line_right = list(map(int, (right_eye + 300 * corrected_w)))
		line_left = np.array([left_eye,end_line_left])
		line_right = np.array([right_eye, end_line_right])

		cv2.line(image, (left_eye[0], left_eye[1]), (end_line_left[0], end_line_left[1]),
		(255, 0, 0), 2)
		cv2.line(image, (right_eye[0], right_eye[1]), (end_line_right[0], end_line_right[1]),
		(255, 0, 0), 2)
		scale = np.array([512/1920, 424/1080])

		#right_eye_depth = np.array([int(preds[42,0]*scale[0]), int(preds[42,1]*scale[1])])
		#print(frameDepth[right_eye_depth[1], right_eye_depth[0]])

		depth = np.zeros([424,512,3])

		for i in range(424):
			for j in range(512):
				depth[i,j] = (frameDepth[i,j],frameDepth[i,j],255)

		left_eye_d = np.array([int(left_eye[0]*scale[0]), int(left_eye[1]*scale[1])])
		right_eye_d = np.array([int(right_eye[0]*scale[0]), int(right_eye[1]*scale[1])])
		end_line_left_d = np.array([int(end_line_left[0]*scale[0]), int(end_line_left[1]*scale[1])])
		end_line_right_d = np.array([int(end_line_right[0]*scale[0]), int(end_line_right[1]*scale[1])])

		cv2.line(depth, (left_eye_d[0], left_eye_d[1]), (end_line_left_d[0], end_line_left_d[1]),
		(255, 0, 0), 2)
		cv2.line(depth, (right_eye_d[0], right_eye_d[1]), (end_line_right_d[0], end_line_right_d[1]),
		(255, 0, 0), 2)

		"""
		Points of interest from 2D picture to 3D space coordinates
		"""
		# w,x,y,z are coordinates on the depth frame whereas w_d, x_d, y_d, z_d are the depth values of these coordinates
		w, x, y, z = np.array([preds[36,0]*scale[0], preds[36,1]*scale[1]]), np.array([preds[45,0]*scale[0], preds[45,1]*scale[1]]), np.array([preds[27,0]*scale[0], preds[27,1]*scale[1]]), np.array([preds[51,0]*scale[0], preds[51,1]*scale[1]])
		w_d, x_d, y_d, z_d = frameDepth[int(w[1]), int(w[0])], frameDepth[int(x[1]), int(x[0])], frameDepth[int(y[1]), int(y[0])], frameDepth[int(z[1]), int(z[0])]

		right_eye = np.array([right_eye[0]*scale[0], right_eye[1]*scale[1]])
		right_eye_d = frameDepth[int(right_eye[1]), int(right_eye[0])]

		depthpoints = np.array([w,x,y,z,right_eye])
		depths = np.array([w_d,x_d,y_d,z_d,right_eye_d])
		face_land = kinect.acquireCameraSpace(depthpoints, depths)

		x_0 = face_land[0:3]
		x_1 = face_land[3:6]
		y_0 = face_land[6:9]
		y_1 = face_land[9:12]

		right_eye_s = face_land[12:15]

		

		x_s = normv(x_1 - x_0)
		y_s = normv(y_1 - y_0)
		z_s = np.cross(x_s, y_s)

		#Correction of the director vector
		z_s = rotation_x(z_s,0.3)
		

		if z_s[2] > 0:
			z_s = z_s * (-1)

		k = - right_eye_s[2] / (z_s[2])

		cible = right_eye_s + k*z_s

		data_point = {
			"x": cible[0],
			"y": cible[1],
			"z": cible[2]
		}
		with open("cible.json", "w") as write_file:
			json.dump(data_point, write_file)

		print("Draw a point here:", cible)



		#print(kinect.acquireCameraSpace(depthpoints, depths))
		#image2 = cv2.resize(image,(512,424), interpolation = cv2.INTER_AREA)
		#overlay = depth

		#cv2.addWeighted(overlay, alpha, image2, 0.2 , 0.8, output)
		

		if not image is None:
			cv2.imshow("Output-Keypoints",image)

		#np.savetxt("depth.csv", frameDepth, delimiter=",")

		key = cv2.waitKey(1)
		if key == 27:
		   break
		

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

"""
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../../shape_predictor_68_face_landmarks.dat')
"""

"""
Functions
"""
def normv(v):
    return(v / np.sqrt(np.sum(v**2)))


if __name__ == '__main__':

	kinect = AcquisitionKinect()
	frame = Frame()
	fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cuda")


	while True:
		kinect.get_frame(frame)
		kinect.get_depth_frame()
		kinect.get_color_frame()
		image = kinect._frameRGB
		frameDepth = kinect._frameDepth
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

		x = preds[45,:] - preds[36,:]
		y = preds[27,:] - preds[51,:]


		u = normv(x)
		v = normv(y)
		w = np.cross(u,v)

		if w[2] < 0:
			w = w * (-1)

		left_eye = (preds[45,:] + preds[42,:])//2
		right_eye = (preds[39,:] + preds[36,:])//2
		end_line_left = left_eye + 300 * w
		end_line_right = right_eye + 300 * w
		line_left = np.array([left_eye,end_line_left])
		line_right = np.array([right_eye, end_line_right])

		cv2.line(image, (left_eye[0], left_eye[1]), (end_line_left[0], end_line_left[1]),
		(255, 0, 0), 2)
		cv2.line(image, (right_eye[0], right_eye[1]), (end_line_right[0], end_line_right[1]),
		(255, 0, 0), 2)
		scale = np.array([512/1920, 424/1080])

		#right_eye_depth = np.array([int(preds[42,0]*scale[0]), int(preds[42,1]*scale[1])])
		#print(frameDepth[right_eye_depth[1], right_eye_depth[0]])

		depth = np.array([[(0,0,0) for i in range(512)] for j in range(424)])

		for i in range(424):
			for j in range(512):
				depth[i,j] = (frameDepth[i,j],frameDepth[i,j],255)

		image2 = cv2.resize(image,(512,424), interpolation = cv2.INTER_AREA)
		overlay = depth
		
		cv2.addWeighted(overlay, alpha, image2, 0.2 , 0.8, output)
		

		if not image is None:
			cv2.imshow("Output-Keypoints",output)

		key = cv2.waitKey(1)
		if key == 27:
		   break
		

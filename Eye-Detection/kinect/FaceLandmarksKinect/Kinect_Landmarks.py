import cv2
import dlib
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from acquisitionKinect import AcquisitionKinect
from frame import Frame

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../../shape_predictor_68_face_landmarks.dat')

if __name__ == '__main__':

	kinect = AcquisitionKinect()
	frame = Frame()

	while True:
		kinect.get_frame(frame)
		kinect.get_color_frame()
		image = kinect._frameRGB
		#OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		faces = detector(gray)

		for face in faces:
			landmarks = predictor(gray, face)

			for n in range(68):
				x = landmarks.part(n).x
				y = landmarks.part(n).y
				cv2.circle(image, (x, y), 4, (255, 0, 0), -1)



		if not image is None:
			cv2.imshow("Output-Keypoints",image)

		key = cv2.waitKey(1)
		if key == 27:
		   break

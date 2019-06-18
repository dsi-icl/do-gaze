import cv2
import numpy as np
from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
from acquisitionKinect import AcquisitionKinect
from frame import Frame


if __name__ == '__main__':

	kinect = AcquisitionKinect()
	frame = Frame()


	while True:
		kinect.get_frame(frame)
		kinect.get_color_frame()
		print(kinect._kinect.color_frame_desc.HorizontalFieldOfView, kinect._kinect.color_frame_desc.VerticalFieldOfView)
		image = kinect._frameRGB

		#OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
		#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		

		if not image is None:
			cv2.imshow("Output-Keypoints",image)

		key = cv2.waitKey(1)
		if key == 27:
		   break
		

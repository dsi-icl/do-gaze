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
import time

ws = websocket.WebSocket()
ws.connect("wss://gdo-gaze.dsi.ic.ac.uk")


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

def remove_dop_np(arrn):
    copy = arrn
    n = len(arrn)
    for i in range(n-1):
        pers = arrn[i][3]
        dist = arrn[i][4]
        for j in range(i+1, n):
            testeur = arrn[j]
            if pers == testeur[3]:
                if dist < testeur[4]:
                    copy = np.delete(copy, j, 0)
                else:
                    copy = np.delete(copy, i, 0)
    return copy

"""
Create face plan
"""
def face_plan(CP, face):
    x_0_pre = CP[int(face[36,1]), int(face[36,0])]
    x_1_pre = CP[int(face[45,1]), int(face[45,0])]
    x_1_2_pre = CP[int(face[42,1]), int(face[42,0])]
    y_0_pre = CP[int(face[51,1]), int(face[51,0])]
    y_1_pre = CP[int(face[27,1]), int(face[27,0])]

    x_0 = floor.point_to_transform_space(np.array([x_0_pre[0], x_0_pre[1], x_0_pre[2]]))
    x_1 = floor.point_to_transform_space(np.array([x_1_pre[0], x_1_pre[1], x_1_pre[2]]))
    x_1_2= floor.point_to_transform_space(np.array([x_1_2_pre[0], x_1_2_pre[1], x_1_2_pre[2]]))
    y_0 = floor.point_to_transform_space(np.array([y_0_pre[0], y_0_pre[1], y_0_pre[2]]))
    y_1 = floor.point_to_transform_space(np.array([y_1_pre[0], y_1_pre[1], y_1_pre[2]]))

    x_s = x_1 - x_0
    x_s = x_s/(np.linalg.norm(x_s))
    y_s = y_1 - y_0
    y_s = y_s/(np.linalg.norm(y_s))
    z_s = np.cross(x_s, y_s)

    left_eye_s = (x_1+x_1_2)//2

    if z_s[2] > 0:
        z_s = z_s * (-1)

    k = - left_eye_s[2]/z_s[2]
    cible = left_eye_s + k*z_s

    return(cible)


if __name__ == '__main__':

    kinect = AcquisitionKinect()
    frame = Frame()
    time.sleep(2)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device="cuda")
    compteur = 0
    timer = True
    timerB = time.time()
    while timer:
        timerB1 = time.time()
        timeA = timerB1 - timerB
        if timeA > 5:
            data_cible = np.array([[0,0,0,0,0]])
            kinect.get_frame(frame)
            kinect.get_color_frame()
            image = kinect._frameRGB
            frameDepth = kinect._frameDepth
            kinect.get_eye_camera_space_coord()
            joint = kinect.joint_points3D
            # print("List of bodies", joint)
            CameraPoints = kinect.cameraPoints
            floor = Floor(kinect._bodies)


            # OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Add movement sensor here (ie when the head doesn't move, don't use get_landmarks)
            # Use moving average

            if compteur == 2:
                test = time.time()
                face_2 = np.asarray(fa.get_landmarks(image))
                print("pb", time.time() - test)
                if face_2 != []:
                    nb_detected = len(face_2)
                    compteur += 1
                    preds = (face_0+face_1+face_2)/3
                    try:
                        assert len(joint) == nb_detected
                    except:
                        print("Error between the number of faces detected and the number of skeletons")

            elif compteur == 1:
                test = time.time()
                face_1 = np.asarray(fa.get_landmarks(image))
                print("pb", time.time() - test)
                if face_1 != []:
                    nb_detected = len(face_1)
                    compteur += 1
                    try:
                        assert len(joint) == nb_detected
                    except:
                        print("Error between the number of faces detected and the number of skeletons")

            elif compteur == 0:
                test = time.time()
                face_0 = np.asarray(fa.get_landmarks(image))
                print("pb", time.time() - test)
                if face_0 != []:
                    nb_detected = len(face_0)
                    compteur += 1
                    try:
                        assert len(joint) == nb_detected
                    except:
                        print("Error between the number of faces detected and the number of skeletons")

            elif compteur == 3:
                face_0 = face_1
                face_1 = face_2
                compteur -= 1

                for k in range(nb_detected):
                    face = preds[k]
                    # draw all faces
                    # for i in range(68):
                    # 	cv2.circle(image, (preds[k][i,0], preds[k][i,1]), 3, (255, 0, 0), -1)
                    
                    # The right eye is defined by being the centor of two landmarks
                    # right_eye = (preds[k][45,:] + preds[k][42,:])//2

                    # 
                    nose_s = np.array([CameraPoints[int(face[36,1]), int(face[36,0])][0], CameraPoints[int(face[36,1]), int(face[36,0])][1], CameraPoints[int(face[36,1]), int(face[36,0])][2]])
                    face_nb, distance = face_number(joint, nose_s)
                    
                    cible = face_plan(CameraPoints, face)

                    data_cible = np.append(data_cible, [[cible[0], cible[1], cible[2], face_nb, distance]], axis=0)

                data_cible = np.delete(data_cible, 0, 0)
                data_cible = data_cible[~np.isnan(data_cible).any(axis=1)]
                data_cible = remove_dop_np(data_cible)
                data_cible = data_cible[data_cible[:,3].argsort()]

                message = {}
                for i in range(len(data_cible)):
                    message['{0}'.format(str(i))] = {'x':data_cible[i][0], 'y':data_cible[i][1], 'z':data_cible[i][2]}

                print("message", message)
                message = json.dumps(message)
                ws.send(message)

            timerE = time.time() - timerB

            if timerE > 90:
                timer = False

            key = cv2.waitKey(1)
            if key == 27:
                ws.close()
                break
                

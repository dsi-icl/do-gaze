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
Solving the quadratic equation related to the gaze position on the GDO
"""

def solve(a, b, c):
    d = b**2 - 4*a*c

    if d<0:
        print("There has been an error")
        return None
    elif d == 0:
        return -b/(2*a)
    else:
        m_1 = (-b - math.sqrt(d)) / (2*a)
        m_2 = (-b + math.sqrt(d)) / (2*a)
        return m_1, m_2

"""
Create face plan
"""
def face_plan(CP, face, kinect_p, r, rotation_matrix):
    x_0_pre = CP[int(face[36,1]), int(face[36,0])]
    x_1_pre = CP[int(face[45,1]), int(face[45,0])]
    x_1_2_pre = CP[int(face[42,1]), int(face[42,0])]
    y_0_pre = CP[int(face[51,1]), int(face[51,0])]
    y_1_pre = CP[int(face[27,1]), int(face[27,0])]

    x_0_pre = floor.point_to_transform_space(np.array([x_0_pre[0], x_0_pre[1], x_0_pre[2]]))
    x_1_pre = floor.point_to_transform_space(np.array([x_1_pre[0], x_1_pre[1], x_1_pre[2]]))
    x_1_2_pre = floor.point_to_transform_space(np.array([x_1_2_pre[0], x_1_2_pre[1], x_1_2_pre[2]]))
    y_0_pre = floor.point_to_transform_space(np.array([y_0_pre[0], y_0_pre[1], y_0_pre[2]]))
    x_1_pre = floor.point_to_transform_space(np.array([y_1_pre[0], y_1_pre[1], y_1_pre[2]]))

    x_0 = np.dot(floor.point_to_transform_space(np.array([x_0_pre[0], x_0_pre[2], x_0_pre[1]])), rotation_matrix) + kinect_p
    x_1 = np.dot(floor.point_to_transform_space(np.array([x_1_pre[0], x_1_pre[2], x_1_pre[1]])), rotation_matrix) + kinect_p
    x_1_2= np.dot(floor.point_to_transform_space(np.array([x_1_2_pre[0], x_1_2_pre[2], x_1_2_pre[1]])), rotation_matrix) + kinect_p
    y_0 = np.dot(floor.point_to_transform_space(np.array([y_0_pre[0], y_0_pre[2], y_0_pre[1]])), rotation_matrix) + kinect_p
    y_1 = np.dot(floor.point_to_transform_space(np.array([y_1_pre[0], y_1_pre[2], y_1_pre[1]])), rotation_matrix) + kinect_p

    print("head position according to face_alignment", y_0)
    x_s = x_1 - x_0
    x_s = x_s/(np.linalg.norm(x_s))
    y_s = y_1 - y_0
    y_s = y_s/(np.linalg.norm(y_s))
    z_s = np.cross(x_s, y_s)

    left_eye_s = (x_1+x_1_2)//2

    a = z_s[0]**2 + z_s[1]**2
    b = 2*(z_s[0]*left_eye_s[0] + z_s[1]*left_eye_s[1])
    c = left_eye_s[0]**2 + left_eye_s[1]**2 - r**2
    k = solve(a, b, c)
    if len(k) == 2:
        print("k face alignment equals 2")
        test = left_eye_s + k[0]*z_s
        if test[1] > 0:
            sol = k[0]
            print("is it negative", test[1])
        else:
            sol = k[1]
            print("is it negative", left_eye_s + k[1]*z_s)
    else:
        sol = k
    cible = left_eye_s + sol*z_s

    return(cible)
        

if __name__ == '__main__':

    kinect_position = (2.63, 1.44, 2.515)

    # Information about the GDO geometry
    # Vision span is 313 degrees
    r = 3
    theta_node_1 = 1.02
    theta_node_2 = 1.37
    theta_kinect = 0.5
    R = np.array([[math.cos(theta_kinect), -math.sin(theta_kinect), 0], \
                [math.sin(theta_kinect), math.cos(theta_kinect), 0], \
                [0, 0, 1]])

    kinect = AcquisitionKinect()
    frame = Frame()
    time.sleep(2)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False, device="cuda")
    compteur = 0
    timer = True
    timerB = time.time()
    faces = [0 for i in range(3)]
    while timer:
        timerB1 = time.time()
        timeA = timerB1 - timerB
        if timeA > 15:
            data_cible = np.array([[0,0,0,0,0,0,0,0]])
            kinect.get_frame(frame)
            kinect.get_color_frame()
            image = kinect._frameRGB
            frameDepth = kinect._frameDepth
            kinect.get_eye_camera_space_coord()
            joint = kinect.joint_points3D
            # print("List of bodies", joint)
            CameraPoints = kinect.cameraPoints
            floor = Floor(kinect._bodies)
            kinect_direction = floor.face_direction()

            # OpenCv uses RGB image, kinect returns type RGBA, remove extra dim.
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Add movement sensor here (ie when the head doesn't move, don't use get_landmarks)
            # Use moving average

            if compteur == 2:
                test = time.time()
                faces[2] = np.asarray(fa.get_landmarks(image))
                print("pb", time.time() - test)
                if faces[2] != []:
                    nb_detected = len(faces[2])
                    compteur += 1
                    try:
                        assert len(joint) == nb_detected
                    except:
                        print("Error between the number of faces detected and the number of skeletons")
                        compteur -= 1
                    try:
                        preds = (faces[0]+faces[1]+faces[2])/3
                    except:
                        print("Average not possible due to a change in the number of people detected")
                        compteur = 0

            elif compteur == 3:
                faces[0] = faces[1]
                faces[1] = faces[2]
                compteur -= 1

                for k in range(nb_detected):
                    face = preds[k]
                    # draw all faces
                    # for i in range(68):
                    # 	cv2.circle(image, (preds[k][i,0], preds[k][i,1]), 3, (255, 0, 0), -1)
                    
                    # The right eye is defined by being the centor of two landmarks
                    # right_eye = (preds[k][45,:] + preds[k][42,:])//2

                    nose_s = np.array([CameraPoints[int(face[36,1]), int(face[36,0])][0], CameraPoints[int(face[36,1]), int(face[36,0])][1], CameraPoints[int(face[36,1]), int(face[36,0])][2]])
                    face_nb, distance = face_number(joint, nose_s)
                    # getting the coefficient that we will use for the gaze estimation (using only the kinect) 
                    try:
                        len(kinect_direction)
                        body = np.array([joint[face_nb][0], joint[face_nb][1], joint[face_nb][2]])
                        trans = floor.point_to_transform_space(body)
                        body = np.array([trans[0], trans[2], trans[1]])
                        print("head position according to kinect preprocessed", trans)
                        body = np.dot(body, R) + kinect_position
                        print("head position according to kinect", body)
                        kinect_direction = np.dot(kinect_direction, R)
                        a = kinect_direction[0]**2 + kinect_direction[1]**2
                        b = 2*(kinect_direction[0]*body[0] + kinect_direction[1]*body[1])
                        c = body[0]**2 + body[1]**2 - r**2
                        k = solve(a, b, c)
                        if len(k) == 2:
                            test = body + k[0]*kinect_direction
                            print("k kinect equals 2")
                            if test[1] > 0:
                                sol = k[0]
                                print("is it negative", test[1])
                            else:
                                sol = k[1]
                                print("is it negative", body + k[1]*kinect_direction)
                        else:
                            sol = k
                        cible_k = body + sol*kinect_direction
                    except TypeError:
                        print("problem with the kinect detection")

                    # Still have to figure out which m to take
                    
                    cible = face_plan(CameraPoints, face, kinect_position, r, R)

                    data_cible = np.append(data_cible, [[cible[0], cible[1], cible[2], face_nb, distance, cible_k[0], cible_k[1], cible_k[2]]], axis=0)
                    

                data_cible = np.delete(data_cible, 0, 0)
                data_cible = data_cible[~np.isnan(data_cible).any(axis=1)]
                data_cible = remove_dop_np(data_cible)
                data_cible = data_cible[data_cible[:,3].argsort()]

                message = {}
                for i in range(len(data_cible)):
                    message['{0}'.format(str(i))] = {'x':data_cible[i][0], 'y':data_cible[i][1], 'z':data_cible[i][2], 'x_k':data_cible[i][5], 'y_k':data_cible[i][6], 'z_k':data_cible[i][7]}

                print("message", message)
                message = json.dumps(message)
                ws.send(message)

            else:
                test = time.time()
                faces[compteur] = np.asarray(fa.get_landmarks(image))
                print("pb", time.time() - test)
                if faces[compteur] != []:
                    nb_detected = len(faces[compteur])
                    compteur += 1
                    try:
                        assert len(joint) == nb_detected
                    except:
                        print("Error between the number of faces detected and the number of skeletons")
                        compteur -= 1
            timerE = time.time() - timerB

            if timerE > 90:
                timer = False

            key = cv2.waitKey(1)
            if key == 27:
                ws.close()
                break
                

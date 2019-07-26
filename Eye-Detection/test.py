import cv2
import numpy as np
#import dlib
import face_alignment
import time
import sys

algo = sys.argv[0]

departureTime = time.time()
print("Time0", time.time() - departureTime)
cap = cv2.VideoCapture(0)
print("Width:", cap.get(3), "Height:", cap.get(4))
print("Time0.1", time.time() - departureTime)
if (algo == 0):
    detector = dlib.get_frontal_face_detector()
    print("Time0.2", time.time() - departureTime)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("Time0.3", time.time() - departureTime)

else:
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda")

while True:
    _, frame = cap.read()
    
    
    if (algo == 0):
        #print("Time1", time.time() - departureTime)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print("Time2", time.time() - departureTime)
        faces = detector(gray)
        #print("Time3", time.time() - departureTime)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            #print("Time3.1", time.time() - departureTime)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            landmarks = predictor(gray, face)
            #print("Time3.2", time.time() - departureTime)

            for n in range(68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
                
            print("Time3.3", time.time() - departureTime)

        eyes_position = frame[landmarks.part(21).y:landmarks.part(28).y,landmarks.part(36).x:landmarks.part(45).x]
        print("Time4", time.time() - departureTime)

    else:
        startTime = time.time()
        preds = fa.get_landmarks(frame)[-1]
        print("get_lmrks_frm_img", time.time() - startTime)
        for i in range(68):
            cv2.circle(frame, (preds[i,0], preds[i,1]), 3, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break


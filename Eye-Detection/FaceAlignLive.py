import cv2
import numpy as np
import face_alignment
from skimage import io
from rotation import Rotation


"""
Functions
"""
def normv(v):
    return(v / np.sqrt(np.sum(v**2)))

"""
Script
"""
cap = cv2.VideoCapture(0)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cuda")


while True:
    _,frame = cap.read()

    preds = fa.get_landmarks(frame)[-1]

    x = preds[45,:] - preds[36,:]
    y = preds[27,:] - preds[51,:]


    u = normv(x)
    v = normv(y)
    w = np.cross(u,v)

    """
    rot = Rotation(u, v, w)
    angles = rot.find_angles()

    print(angles)
    """
    
    if w[2] < 0:
        w = w * (-1)

    left_eye = (preds[45,:] + preds[42,:])//2
    right_eye = (preds[39,:] + preds[36,:])//2
    end_line_left = left_eye + 300 * w
    end_line_right = right_eye + 300 * w
    line_left = np.array([left_eye,end_line_left])
    line_right = np.array([right_eye, end_line_right])

    """
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
            """
    cv2.line(frame, (left_eye[0], left_eye[1]), (end_line_left[0], end_line_left[1]),
    (255, 0, 0), 2)
    cv2.line(frame, (right_eye[0], right_eye[1]), (end_line_right[0], end_line_right[1]),
    (255, 0, 0), 2)

    #eyes_position = frame[landmarks.part(21).y:landmarks.part(28).y,landmarks.part(36).x:landmarks.part(39).x]

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

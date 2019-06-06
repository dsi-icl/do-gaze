import cv2
import numpy as np

img = cv2.imread('../Andrinirina/all.png',0)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,100,100,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
print(circles)
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

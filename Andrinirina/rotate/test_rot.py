import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/im2.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[881, 663],[1024,637],[935,641],[935, 739]])
pts2 = np.float32([[100,100],[260,100],[180,200],[180,304]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(360,360))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
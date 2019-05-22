import cv2
import dlib
import numpy as np
import sys
import pandas as pd

path = sys.argv[1]

img = cv2.imread(path, 0)[200:290,115:200]
height, width = img.shape

step_h = height//3
step_w = width//5

print(height, width)
#level = pd.DataFrame(columns = ['Mean_Value'])
for i in range(15):
    tile = img[step_h*(i%3):step_h*(i%3 + 1),step_w*(i%5):step_w*(i%5 + 1)]
    #level = level.append({'Mean_Value':tile.mean(axis = 0).mean(axis = 0)}, ignore_index = True)

#level.plot.bar()

while True:
    cv2.imshow('test', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

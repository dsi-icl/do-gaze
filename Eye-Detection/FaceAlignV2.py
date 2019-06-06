import cv2
import numpy as np
import face_alignment
from skimage import io
import pandas as pd


from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

Eye_matrix = pd.read_csv('Eye_matrix.csv')

X = Eye_matrix.drop('Position', axis=1)
y = Eye_matrix[['Position']]

class Image:

    def __init__(self, left_eye, right_eye):
        self.left_eye = left_eye
        self.right_eye = right_eye

    for weights in ['uniform', 'distance']:
        clf = KNeighborsClassifier(3, weights=weights)
        clf.fit(X, y)
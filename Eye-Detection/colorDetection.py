# import the necessary packages
import numpy as np
import sys
import dlib
import cv2
import glob
import os

# construct the argument parse and parse the arguments
faces_folder_path = sys.argv[1]

win = dlib.image_window()
# define the list of boundaries
boundaries = [
    ([150, 150, 150], [255, 255, 255])
]

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    print("Processing file: {}".format(f))
    image = dlib.load_rgb_image(f)
    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask = mask)

        # show the images
        win.set_image(output)
    dlib.hit_enter_to_continue()

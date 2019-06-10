import time

import cv2
import mss
import numpy


with mss.mss() as sct:
    """
    # Part of the screen to capture
    monitor = sct.monitors[1]

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        img = cv2.circle(img,(447,63), 3, (0,0,255), -1) 

        # Display the picture
        cv2.imshow("OpenCV/Numpy normal", img)
        cv2.namedWindow("OpenCV/Numpy normal",cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("OpenCV/Numpy normal", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(3000) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break"""

    #All the monitors in 1 screenshot
    filename = sct.shot(mon=-1, output='fullscreen.png')
    print(filename)

    #1 screenshot per monitor
    for filename in sct.save():
    print(filename)
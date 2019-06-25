#import time

import cv2
import mss
import numpy
import win32gui
import win32ui
from ctypes import windll
from PIL import Image

hwnd = win32gui.GetActiveWindow()

# Change the line below depending on whether you want the whole window
# or just the client area. 
#left, top, right, bot = win32gui.GetClientRect(hwnd)
left, top, right, bot = win32gui.GetWindowRect(hwnd)
w = right - left
h = bot - top

hwndDC = win32gui.GetWindowDC(hwnd)
mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
saveDC = mfcDC.CreateCompatibleDC()

saveBitMap = win32ui.CreateBitmap()
saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)

saveDC.SelectObject(saveBitMap)

# Change the line below depending on whether you want the whole window
# or just the client area. 
#result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
print(result)

bmpinfo = saveBitMap.GetInfo()
bmpstr = saveBitMap.GetBitmapBits(True)

im = Image.frombuffer(
    'RGB',
    (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
    bmpstr, 'raw', 'BGRX', 0, 1)

win32gui.DeleteObject(saveBitMap.GetHandle())
saveDC.DeleteDC()
mfcDC.DeleteDC()
win32gui.ReleaseDC(hwnd, hwndDC)

if result == 1:
    #PrintWindow Succeeded
    im.save("test.png")

"""
with mss.mss() as sct:
    # Part of the screen to capture
    monitor = sct.monitors[1]

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        img = cv2.circle(img,(447,0), 3, (0,0,255), -1) 

        # Display the picture
        cv2.imshow("OpenCV/Numpy normal", img)
        cv2.namedWindow("OpenCV/Numpy normal",cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("OpenCV/Numpy normal", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("fps: {}".format(1 / (time.time() - last_time)))

        # Press "q" to quit
        if cv2.waitKey(50) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
            
            
    

    #All the monitors in 1 screenshot
    filename = sct.shot(mon=-1, output='fullscreen.png')
    print(filename)

    #1 screenshot per monitor
    for filename in sct.save():
        print(filename)
        """
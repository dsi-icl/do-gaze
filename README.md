# GDO-GAZE Project

**GDO-GAZE** is a gaze estimation software using a kinect (hopefully multiple kinects soon) and a face recognition library. It is developped in Python and requires a GPUn for optimal performance.


## Installation

### Get the GDO-GAZE source code
```bash
git clone https://github.com/dsi-icl/gdo-gaze.git
```

### Requirements
This project was developped using [Anaconda](https://www.anaconda.com/distribution/).

* Python 3.5+ or Python 2.7
* Linux, Windows or macOS
* pytorch (>=1.0)
* [Face_alignment](https://github.com/1adrianb/face-alignment) library by Adrian Bulat
* [Websocket client](https://pypi.org/project/websocket_client/)
* Opencv
* Pykinect2:
    1. pip install comtypes
    2. Install the [Kinect for Windows SDK v2](https://www.microsoft.com/en-gb/download/details.aspx?id=44561)
    3. pip install pykinect2

:exclamation: Use pip in the following directory 
* Mac: ```.../anaconda3/lib/python3.7/site-packages/``` 
* Windows: ```.../anaconda3/lib/sites-packages/```


## Launching the software
Open a terminal in ```Eye-Detection/FaceLandmarksKinect/```:

```bash
python benchmark.py
```


## Contributors
Development of this project was done by [Andrianirina Rakotoharisoa](https://github.com/Wowygrimm) under the supervision of Dr Ovidiu Șerban and Florian Guitton.
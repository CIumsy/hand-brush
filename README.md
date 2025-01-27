# Hand Brush 🖌️
Machine learning paint program that uses your hand as a brush!


```

Note:

1. Demos (gifs) will take a bit of time to load coz they are heavy 🧸

2. You are required to install the required libraries using the requirements.txt file.

3. To run the notebooks locally you are required to install Jupyter-notebook
  OR Jupyter-lab.

4. Although python2 is deprecated, If you have both python2 and python3 installed 
  replace python with python3 for running the application and demos.
```

<br>

## Prerequisites
- Python 3.7 or higher
- A working webcam

## Installation
1. Clone the repository:
   ```bash
   cd <repository-directory>
   git clone https://github.com/lorforlinux/hand-brush.git

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt


##  Draw using your hand
---
This application uses a webcam and hand gestures to control mouse movements and clicks in real time. It leverages MediaPipe for hand tracking, OpenCV for video processing, and PyAutoGUI for mouse control.

### Features
- **Move Mouse Cursor:** Use your index finger to control the cursor position.
- **Left Click:** Close your fist.
- **Right Click:** Open your hand and flex your fingers.
- **Smooth Cursor Movement:** Includes smoothing and thresholds for responsive control.

### Run the script
    python cursor.py
    
    


###  Run the script


###  Demo of the program!
---
For hand traking my application here is using the pretrained ``SSD with MobilenetV1` model from [EvilPort2](https://github.com/EvilPort2)'s
hand tracking [repository](https://github.com/EvilPort2/Hand-Tracking). The paint toolbox uses the code from [acl21](https://github.com/acl21)'s Webcam Paint OpenCV [repository](https://github.com/acl21/Webcam_Paint_OpenCV). Hand Brush program is truly a combination of those two repositories and i highly recommend you to check out their repositories 🦔


```
# for running hand brush program

python handbrush.py
```


Watch the demo with updated tool box [here](https://www.youtube.com/watch?v=Pnr-YD98XYo&feature=youtu.be).


![paint program demo][paint]

<br>


### Hand detection demo!
---

This demo uses [EvilPort2](https://github.com/EvilPort2)'s
hand tracking [model](https://github.com/EvilPort2/Hand-Tracking) but you can also try the [victordibia](https://github.com/victordibia)'s hand tracking [model](https://github.com/victordibia/handtracking). I tried both and got better result from the first one.

```
# for running hand detection demo

python handdetect.py

OR

python detect_single_threaded.py
```

![hand detect demo][detect]


<br>


### Hand pointer demo!
---
This demo uses the [akshaybahadur21](https://github.com/akshaybahadur21)'s Hand Movement Tracking [code](https://github.com/akshaybahadur21/HandMovementTracking) for creating the pointer trail.


```
# for running hand pointer demo

python handpointer.py
```


![pointer demo][pointer]


<br>

### Contour demo!
---
This one is just for fun and learning [How can I find contours inside ROI using opencv and Python?](https://stackoverflow.com/questions/42004652/how-can-i-find-contours-inside-roi-using-opencv-and-python).

```
# for running contour demo

python handcontour.py
```

![pointer demo][contour]


<br>

### Jupyter Notebooks
---
If you want to learn how does it work internally, the best way is to follow the Jupyter Notebooks i have included in this repository.

| Notebook | Purpose
| --- | --- |
|[detecthand.ipynb](./detecthand.ipynb)| This notebook will give you the base code for all the other notebooks and for the application itself.|
| [gethand.ipynb](./gethand.ipynb)| Here you'll learn to extract the detected hand as separate image data.|
|[centeroid.ipynb](./centeroid.ipynb)| We will calculate the centeroid point in this notebook.|
|[countour.ipynb](./contour.ipynb)| We will find the contours in the detected hand by taking it as ROI (Region Of Interest)|

![Jupyter Notebooks][notebooks]



<br>

`If you find any issue in the code OR want any help please create an issue in the issue tracker :)`

<br>


## References
1. https://github.com/EvilPort2/Hand-Tracking
2. https://github.com/acl21/Webcam_Paint_OpenCV
3. https://github.com/akshaybahadur21/HandMovementTracking
4. https://github.com/victordibia/handtracking




[detect]: ./assets/detect.gif "hand detect demo"
[paint]: ./assets/paint.gif "paint program demo"
[contour]: ./assets/contour.gif "contour demo"
[pointer]: ./assets/pointer.gif "pointer demo"
[notebooks]: ./assets/notebooks.png "contour demo"


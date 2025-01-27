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


##  Control the cursor using your hand
---
This program uses a webcam and hand gestures to control mouse movements and clicks in real time. It leverages MediaPipe for hand tracking, OpenCV for video processing, and PyAutoGUI for mouse control.

### Features
- **Move Mouse Cursor:** Use your index finger to control the cursor position.
- **Left Click:** Close your fist.
- **Right Click:** Open your hand and flex your fingers.
- **Smooth Cursor Movement:** Includes smoothing and thresholds for responsive control.

### Run the script
    python cursor.py
#### Terminate the script using CTRL+C   


#### Note: You can also run cursorlite.py to only control the cursor without the buttons.

  
    


##  Draw using your hands
---
This program allows users to draw on a virtual canvas using hand gestures detected via a webcam. It leverages MediaPipe for real-time hand tracking and OpenCV for rendering the canvas. Users can draw freehand, toggle straight-line mode, erase, and customize brush settings and colors.

### Features
- **Freehand Drawing**: Use your hand to draw lines on the canvas.
- **Customizable Brush**: Change brush size and color.
- **Clear Canvas**: Clear the entire drawing canvas with a single command.
- **On-Screen Tutorial**: Toggle an interactive tutorial for controls.

### Run the script
    python handdraw.py
#### Terminate the program by pressing 'q' on your keyboard.

<img src="canvas.jpg" alt="Canvas" style="display: block; margin: 10px auto; width: 400px;">

### Controls
-   q: Quit the program.
-   e: Clear the canvas.
-   r: Change brush color to Red.
-   g: Change brush color to Green.
-   b: Change brush color to Blue.
-   k: Change brush color to Black.
-   y: Change brush color to Cyan.
-   +: Increase brush size.
-   -: Decrease brush size.
-   t: Show or hide the tutorial.


##  Demo of the program!
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


## Hand detection demo!
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


## Hand pointer demo!
---
This demo uses the [akshaybahadur21](https://github.com/akshaybahadur21)'s Hand Movement Tracking [code](https://github.com/akshaybahadur21/HandMovementTracking) for creating the pointer trail.


```
# for running hand pointer demo

python handpointer.py
```


![pointer demo][pointer]


<br>

## Contour demo!
---
This one is just for fun and learning [How can I find contours inside ROI using opencv and Python?](https://stackoverflow.com/questions/42004652/how-can-i-find-contours-inside-roi-using-opencv-and-python).

```
# for running contour demo

python handcontour.py
```

![pointer demo][contour]


<br>

## Jupyter Notebooks
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


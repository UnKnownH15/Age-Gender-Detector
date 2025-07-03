# Age-Gender-Detector
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-pytest%20✓-brightgreen)](https://pytest.org/)

This is a final project for HarvardX CS50P course ([CS50's Introduction to Programming with Python](https://cs50.harvard.edu/python/2022/)).

## Features

- Face detection using pre-trained OpenCV DNN model
- Age group prediction (0–100+ years)
- Gender prediction (Male/Female)
- Works in real time with input from webcam

This is a final project for the HarvardX CS50P course, built using Python and OpenCV's DNN module. It performs real-time face detection, followed by age and gender prediction using pre-trained deep learning models.

You can run the program using an image, a video file, or your computer’s webcam.

How It Works
- After detecting faces from an input frame (image, video, or live webcam), the program:
- Extracts each face from the frame
- Uses a deep learning model to predict gender
- Uses another model to predict age group
- Draws a green rectangle around the face
- Annotates the image with predicted gender and age group



## Installation
It's recommended to first create a python virtual environment and then install the requirements with the following command:
```bash
pip install -r requirements.txt
```

### To run it with Image or Video
```bash
python project.py --image path/to/image_or_video
```
### To run it with default webcam
```bash
python project.py
```
### To run with specifc webcam
```bash
python project.py --camera replace_with_camera_index
```
### To exit
Just Press **Esc** Key



## Acknowledgments
I would like to express my deepest appreciation to David Malan and the whole CS50 team for entertaining lectures and helpful problem sets.

 


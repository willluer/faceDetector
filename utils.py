import cv2
import numpy as np
import time

# Input: path to image
# Output: img array, img width, img height
def readImageFromFile(path):
	print("[INFO] Reading image from file.")
	frame = cv2.imread(path)
	dims = np.shape(frame)
	return frame,dims[0],dims[1]

# Input: video source
# Output: img array, img width, img height
def readVideoCapture(path):
	camera = cv2.VideoCapture(path)
	time.sleep(1)
	print("[INFO] Reading image from webcam. Say Cheese!")
	ret, frame = camera.read()
	if not ret:
		print("[ERROR] Unable to read image from webcam.")
		raise IOError("Unable to read frame")
	dims = np.shape(frame)
	return frame,dims[0],dims[1]

# Converts normalized bounding box to usable form
def boxConvert(boxArrays,w,h):
	box = boxArrays[0][0]
	y1 = int(box[0]*w)
	x1 = int(box[1]*h)
	y2 = int(box[2]*w)
	x2 = int(box[3]*h)
	return x1,y1,x2,y2

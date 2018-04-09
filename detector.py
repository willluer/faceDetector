import cv2
import numpy as np
import time
import argparse
from os.path import exists
import tensorflow as tf
from TensorflowFaceDetector import TensorflowFaceDetector
from utils import readImageFromFile,readVideoCapture, boxConvert

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

class DetectionManager(object):
	def __init__(self,source,type="image"):

		if source == 0 or type == "video":
			self.readImage = readVideoCapture
		else:
			self.readImage = readImageFromFile

		self.faceDetector = TensorflowFaceDetector(PATH_TO_CKPT)
		self.threshold = 0.6
	# Output bounding box for still coordinates
	def run(self,source):

		# Read image
		img,width,height = self.readImage(source)

		# Send image to FaceDetector
		print("[INFO] Calculating Bounding Box")
		(boxes, scores, classes, num_detections) = self.faceDetector.findBoundingBox(img)

		if scores[0][0] > self.threshold:
			boundingBox = boxConvert(boxes,width,height)
			return True,boundingBox,img
		else:
			return False,None,None

if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("-s", "--source", required=True, help="Enter the path to your image or enter 0 to use your webcam.")
	args = vars(ap.parse_args())
	source = args["source"]

	print("[INFO] Source supplied: ", source)

	if source == "0":
		source = int(source)
	else:
		if not exists(source):
			raise IOError('Supplied path does not exists')


	# Get bounding box and image
	flag,bb,frame = DetectionManager(source).run(source)
	if flag:
		x1,y1,x2,y2 = bb

		# Draw bounding box on image
		cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 2)

		# Display image
		cv2.imshow("Shiseido | Face Detector",frame)
		print("[RESULTS] Face found")
		print("[USAGE] Press any button to close the window")
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	else:
		print("[RESULTS] No face detected")

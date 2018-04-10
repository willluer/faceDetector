import cv2
import numpy as np
import time
import argparse
from os.path import exists
from detector import DetectionManager

'''
Facial Tracking for video source

Takes in a bounding box and video source and tracks the box in corresponding frames.
INPUT: Bounding box of form [x1,y1,x2,y2], video source
OUTPUT: Visualization of Medainflow tracking algorithm

Usage:
python3 detectAndTrack.py -s/--source [path]
[path] is either a path to a file or 0 to use local webcam

'''
class trackingManager(object):
	def __init__(self,bb,source):
		self.source = cv2.VideoCapture(source)
		self.windowName = "Shiseido | Face Detector | Tracking"
		self.tracker = cv2.TrackerMedianFlow_create()
		self.previous = None
		self.current = None

		#Convert bounding box to usable form: (x1, y1, xSpan, ySpan)
		x1,y1,x2,y2 = bb
		bb = (bb[0],bb[1],bb[2]-bb[0],bb[3]-bb[1])
		self.boundingBox = bb

	def run(self,bb,source):
		time.sleep(1)
		ret, prev = self.source.read()
		width,height,_ = np.shape(prev)

		if not ret:
			print("[ERROR] Unable to read from source")

		# Equalize image
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
		prev = clahe.apply(prev)
		self.previous = prev

		ok = self.tracker.init(self.previous, self.boundingBox)

		while True:
			# Start timer
			timer = cv2.getTickCount()

			# Read new frame
			ret, frame = self.source.read()

			if not ret:
				print("[INFO] Video stream has ended")
				break

			# Equalize
			curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			curr = clahe.apply(curr)
			self.current = curr

			# Update Tracker
			ok, bbox = self.tracker.update(curr)

			# Draw bounding box
			if ok:
				self.boundingBox = bbox
				# Tracking success
				p1 = (int(bbox[0]), int(bbox[1]))
				p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
				cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
			else :
				# Tracking failure
				cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

			# Calculate Frames per second (FPS)
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

			# Display FPS on frame
			cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
			cv2.imshow(self.windowName, frame)

			k = cv2.waitKey(1)
			self.previous = curr
			if k == 27 :
				break


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


	# Start Detection and tracking if specified
	flag,bb,frame = DetectionManager(source,"video").run(source)
	if flag:
		print("[INFO] Initiating MedianFlow tracker")
		print("[USAGE] Press ESC to end the stream")
		trackingManager(bb,source).run(bb,source)
	else:
		print("[RESULTS] No face detected")

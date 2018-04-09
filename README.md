# Face Detection Project
# William Luer

# Dependencies:
   - Tensorflow
   - OpenCV
   - Numpy

## Description
Face detection using Python and Tensorflow. I created a face detection algorithm that is capable of calculating a bounding box from a local image or from your local webcam (detector.py). I also implemented a facial tracking algorithm that uses the bounding box found in detector.py and passes it to a MedianFlow tracking algorithm (detectAndTrack.py). A sample photo and video are provided in the data folder. The photo may be used as input to detector.py and the vidoe may be used as input to detecAndTrack.py.

### Usage for face detection on supplied image path:
python3 detector.py -s/--source path_to_image
ex.) python3 detector.py -s data/sampleFace.jpg

#### Usage for face detection from webcam:
python3 detector.py -s/--source 0
ex.) python3 detector.py -s 0

### Usage for face detection and tracking on supplied video path:
python3 detectAndTrack.py -s/--source <path>
ex.) python3 detectAndTrack.py -s data/trackingSample.avi

### Usage for face detection from webcam:
python3 detector.py -s/--source 0
ex.) python3 detectAndTrack.py -s 0


### File Structure:
faceDetector:
  +--- data/
  |  +--- sampleFace.jpg
  |  +--- trackingSample.avi
  |
  +--- model/
  |  +--- frozen_inference_graph_face.pb
  |
  +--- protos
  +--- detectAndTrack.py
  +--- detector.py
  +--- TensorFlowFaceDetector.py
  +--- utils.py
  +--- README.md

# References
Pre-trained tensorflow model was found online via the provided link.
https://github.com/yeephycho/tensorflow-face-detection/tree/master/model
https://github.com/yeephycho/tensorflow-face-detection/tree/master/protos

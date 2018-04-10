# Face Detection Project
William Luer - 4/9/2018

## Dependencies:
   - Tensorflow
   - OpenCV
   - Numpy

## Description:
Face detection algorithm using Python and Tensorflow. I created a face detection algorithm that calculates a bounding box from a local image or from your local webcam (detector.py). Additionally, I implemented a facial tracking algorithm that uses the bounding box found in detector.py and passes it to a MedianFlow tracking algorithm (detectAndTrack.py).


## Usage:

1. **Face detection on supplied image path:** <br>
  python3 detector.py -s/--source [path]<br>
  ex.) python3 detector.py -s data/sampleFace.jpg

2. **Face detection from webcam:**<br>
  python3 detector.py -s/--source 0<br>
  ex.) python3 detector.py -s 0

3. **Face detection/tracking on supplied video path:**<br>
  python3 detectAndTrack.py -s/--source [path]<br>
  ex.) python3 detectAndTrack.py -s data/trackingSample.avi

4. **Face detection/Tracking from webcam:**<br>
  python3 detector.py -s/--source 0<br>
  ex.) python3 detectAndTrack.py -s 0


### File Structure:

|── data/
|&emsp;&emsp;&ensp;|── sampleFace.jpg
|&emsp;&emsp;&ensp;|── trackingSample.avi
|── results/<br>
|&emsp;&emsp;&ensp;|── detectorResults.png<br>
|&emsp;&emsp;&ensp;|── detectAndTrackResults.avi<br>
|── model/<br>
|&emsp;&emsp;&ensp;|── frozen_inference_graph_face.pb<br>
|── protos/<br>
|&emsp;&emsp;&ensp;|── face_label_map.pbtxt<br>
|&emsp;&emsp;&ensp;|── string_int_label_map_pb2.py<br>
|── detectAndTrack.py<br>
|── detector.py<br>
|── TensorflowFaceDetector.py<br>
|── utils.py<br>
|── README.md<br>



## Sample Data
A sample photo and video are provided in the data folder. The photo may be used as input to detector.py and the video may be used as input to detectAndTrack.py.

## Results
The results of detector.py and detectAndTrack.py run on the sample data provided can be found in the results folder.


# References
Pre-trained tensorflow model was found online via the provided link. <br>
https://github.com/yeephycho/tensorflow-face-detection/tree/master/model <br>
https://github.com/yeephycho/tensorflow-face-detection/tree/master/protos

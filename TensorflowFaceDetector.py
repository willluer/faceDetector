import tensorflow as tf
import cv2
import numpy as np
import time

class TensorflowFaceDetector(object):
	def __init__(self, PATH_TO_CKPT):
		self.detection_graph = tf.Graph()
		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')


		with self.detection_graph.as_default():
			config = tf.ConfigProto()
			config.gpu_options.allow_growth = True
			with tf.Session(graph=self.detection_graph, config=config) as self.sess:
				self.windowNotSet = True

    # INPUT: still image
    # OUTPUT: bounding box coordinates
	def findBoundingBox(self,vis):
		# Run image through tensorflow model
		image_np = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
		image_np_expanded = np.expand_dims(image_np, axis=0)
		image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
		# Each box represents a part of the image where a particular object was detected.
		boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
		# Each score represent how level of confidence for each of the objects.
		# Score is shown on the result image, together with the class label.
		scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
		classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
		num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
		# Actual detection.
		start_time = time.time()
		(boxes, scores, classes, num_detections) = self.sess.run(
			[boxes, scores, classes, num_detections],
			feed_dict={image_tensor: image_np_expanded})
		elapsed_time = time.time() - start_time
		print('[INFO] Inference time cost: {}'.format(elapsed_time))

		return (boxes, scores, classes, num_detections)

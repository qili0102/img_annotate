#need the "faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz" downloaded 
#from the modelzoo and decompressed first

import pyautogui
import time

import numpy as np
import os
import six
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
import collections
# import pyscreenshot as ImageGrab

img_path = 'my_screenshot.jpg'

def focus():
	# size = pyautogui.locateOnScreen('screenpng/focus.png')
	# pyautogui.click(size[0], size[1])
	pyautogui.click(600, 118)

def get_screenshot():
	im = pyautogui.screenshot(img_path)

#choose labels-->default as cars
def choose_labels():
	# list_e = list(pyautogui.locateAllOnScreen('screenpng/labels.png'))
	# for l in list_e:
	# 	pyautogui.click(l[0]-960, l[1]-750)
	# 	print(l[0], l[1])
	#Gauss
	global count
	rounds = int((count+4) * (count+5) / 2)
	for f in range(1,7):
		pyautogui.typewrite(['\t'])
		time.sleep(0.5)
	for f in range(rounds):
		pyautogui.typewrite([' ', ' '])
		pyautogui.typewrite(['\t'])
		time.sleep(0.5)
	# x = 1244
	# y = 455
	# while y < 742:
	# 	pyautogui.click(x, y)
	# 	y = y + 45
	# 	pyautogui.typewrite(['enter'])
	# 	time.sleep(1)
	# pyautogui.click(269, 176)
	# pyautogui.hotkey('command', 'down')
	# time.sleep(1)
	# y = 680
	# while y > 124:
	# 	pyautogui.click(x, y)
	# 	y = y - 45
	# 	pyautogui.typewrite(['enter'])
	# 	time.sleep(1)
	# pyautogui.click(269, 176)
	# pyautogui.hotkey('command', 'up')
	# time.sleep(1)

#helper code
def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#draw bounding boxes using mouse
def draw_bounding_box(ymin, xmin, ymax, xmax):
	im_width, im_height = pyautogui.size()
	(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
	if right-left < 1000:
		pyautogui.moveTo(left+2, top-2, duration=0.25);
		print("move",pyautogui.position())
		time.sleep(2)
		pyautogui.dragRel(right-left, bottom-top+4, duration=0.5);
		time.sleep(2)
		print("drag",pyautogui.position())
		pyautogui.click(269, 176)
		time.sleep(2)
		print("click",pyautogui.position())
		global count
		count = count+1

def draw_boxes_array(boxes, classes, scores, category_index, instance_masks=None, keypoints=None, max_boxes_to_draw=20, min_score_thresh=.5, agnostic_mode=False):
	print("drw array")
	box_to_display_str_map = collections.defaultdict(list)
	print("box_to_display_str_map")
	box_to_color_map = collections.defaultdict(str)
	print("box_to_color_map")
	box_to_instance_masks_map = {}
	print("box_to_instance_masks_map")
	box_to_keypoints_map = collections.defaultdict(list)
	print("box_to_keypoints_map")
	if not max_boxes_to_draw:
		max_boxes_to_draw = boxes.shape[0]
		print("max_boxes_to_draw")
	for i in range(min(max_boxes_to_draw, boxes.shape[0])):
		if scores is None or scores[i] > min_score_thresh:
			box = tuple(boxes[i].tolist())
			print("box1")
			if instance_masks is not None:
				box_to_instance_masks_map[box] = instance_masks[i]
				print("box_to_instance_masks_map[box]")
			if keypoints is not None:
				box_to_keypoints_map[box].extend(keypoints[i])
				print("box_to_keypoints_map[box]")
			if scores is None:
				box_to_color_map[box] = 'black'
				print("box_to_color_map[box]")
			else:
				if not agnostic_mode:
					if classes[i] in category_index.keys():
						class_name = category_index[classes[i]]['name']
					else:
						class_name = 'N/A'
					display_str = '{}: {}%'.format( class_name, int(100*scores[i]))
				else:
					display_str = 'score: {}%'.format(int(100 * scores[i]))
				box_to_display_str_map[box].append(display_str)
				if agnostic_mode:
					box_to_color_map[box] = 'DarkOrange'
				else:
					box_to_color_map[box] = 'DarkOrange'

	for box, color in six.iteritems(box_to_color_map):
		ymin, xmin, ymax, xmax = box
		print(ymin, xmin, ymax, xmax)
		draw_bounding_box(ymin, xmin, ymax, xmax)

def load_models():
	MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017'
	# MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
	# MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
	# MODEL_NAME = 'faster_rcnn_resnet101_coco_11_06_2017'
	# MODEL_NAME = 'rfcn_resnet101_coco_11_06_2017'
	# Path to frozen detection graph. This is the actual model that is used for the object detection.
	PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
	# List of the strings that is used to add correct label for each box.
	PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

	NUM_CLASSES = 90
	#load the model
	global detection_graph
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
    #loading label map
	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	global category_index
	category_index = label_map_util.create_category_index(categories)
	#start detection
def get_recognize():
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			image = Image.open(img_path)
			rgb_im = image.convert('RGB')
			# the array based representation of the image will be used later in order to prepare the
      		# result image with boxes and labels on it.
			image_np = load_image_into_numpy_array(rgb_im)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# Actual detection.
			(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
			# vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), category_index, use_normalized_coordinates=True, line_thickness=8)
			draw_boxes_array(np.squeeze(boxes), np.squeeze(classes), np.squeeze(scores), category_index)
			os.remove(img_path)

def zoom_in():
	pyautogui.click(373, 118)
	time.sleep(1)

def zoom_out():
	pyautogui.hotkey('command', 'up')
	pyautogui.click(352, 118)
	time.sleep(1)

def next_frame():
	pyautogui.hotkey('command', 'up')
	pyautogui.typewrite(['right'])
	time.sleep(1)

global i
i = 0
load_models()
while i <= 464:
	global count
	count = 0
	focus()
	zoom_in()
	get_screenshot()
	get_recognize()
	zoom_out()
	choose_labels()
	focus()
	time.sleep(1)
	next_frame()
	time.sleep(2)
	i+=1





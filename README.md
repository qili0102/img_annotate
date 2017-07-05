Installation: 

1. Install TensorFlow: https://www.tensorflow.org/install/
							
2. Install Pyautogui module in python3
							
3. Pull entire tensorflow/models repo to python3.6/site-packages/tensorflow/

Then in terminal: 

	cd models/object_detection

	python3 setup.py install

The rest installation steps of object detection API is in here: 
https://github.com/tensorflow/models/tree/master/object_detection

Except when adding libraries to PYTHONPATH:

We need to add two lines instead of one:

	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
	export PYTHONPATH=$PYTHONPATH:~/tensorflow/lib/python3.6/site-packages/tensorflow/models/slim
              
4. download all the available frozen models from model zoo, or just one:
              https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md
              Decompress them under the current object detection model folder.
              

Run: 

Put this test_obj.py under the current object detection model folder. 
     
Uncomment the model of your choosing start from line 168 (Only can apply one model at a time). (Actually, I found the faster_rcnn_inception_resnet_v2_atrous model is the most accurate one, but it took longer to do the object detection. It will be faster if integrates with GPU.)
     
Leave your annotation website open (http://reactor.ctre.iastate.edu/nvidiacity/) to the biggest extent without affecting the terminal window.
     
In terminal: 

	python3 test_obj.py
     
This script will start annotate and select the first label "Car" then move to the next image. If you want to stop the scrpt, just open the terminal and press "crl+z".

Note: Have to choose an image without any annotation. Or you can tinker with the choose label function in the script.

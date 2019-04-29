# MiniProject
Machine Learning mini project

Repo Instruction
contains 5 files :
	main.py : main python file that contains the model

	input.py : python file that takes doodle as input

	test.py : python file that takes input from input.py and predict the doodle

	keras-3.h5 : saved keras model

	fc.txt : list of classes used in our model
	
	Doodle classifier.ipynb : Creats model, google collab file


Doodle Classifier

Doodle It' is a machine learning project where the game asks users to draw a doodle, then the
game tries to guess what it is. We teach the model to recognize drawings with the help of machine
learning. We used a CNN to recognize drawings of different types. The CNN will be trained on the
dataset. We trained the model on GPU for free on Google Colab using Keras then run it on the
browser directly using TensorFlow.js. We used keras with tensorflow backend.

Getting Started

Prerequisites
You need to have following libraries in order to run this project.
Tensorflow, Keras, Numpy, PIL, Tkinter, OpenCV, Scikit learn, Matplotlib

Installing 
-First run the main.py to train the data
	OR
-Directly use keras.h5 which is the saved trained model
-Run input.py which opens the canvas for you to draw in. The input from the input.py is then send to test.py that predicts what the drawing is.

Data Used
For this project we used the datasets from
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap
where the datasets were all in preprocessed in matrix numpy form

Model Used
For this project we used CNN model.

Result
![alt text](https://github.com/ElinaBaral/MiniProject/edit/master/img1.png)
![alt text](https://github.com/ElinaBaral/MiniProject/edit/master/img2.png)
![alt text](https://github.com/ElinaBaral/MiniProject/edit/master/img3.png)

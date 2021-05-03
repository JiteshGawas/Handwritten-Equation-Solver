from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
from keras.preprocessing.image import save_img 
#for matrix math
import numpy as np
#for importing our keras model
import keras.models
#for regular expressions, saves time dealing with string data
import re
import tensorflow as tf
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import * 

from io import BytesIO
import base64
import json

from cnn import ConvolutionalNeuralNetwork
#initalize our flask app
app = Flask(__name__)





@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index_new.html")



@app.route('/predict',methods=['POST'])
def predict():
	print("inside app.py predict")
	# encoding image to base 64
	operation = BytesIO(base64.urlsafe_b64decode(request.form['operation']))
	CNN = ConvolutionalNeuralNetwork()
	operation = CNN.predict(operation)


	return json.dumps({
		'operation': operation,
		'solution': eval(operation)
	})

	

if __name__ == "__main__":
	#decide what port to run the app in
	#run the app locally on the givn port
	app.run(debug = True)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
print("Jitesh")
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 22:50:49 2020

@author: sharv
"""
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# # Model saved 
MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path)
    img = img.resize((28, 28))
    img = np.invert(img.convert('L')).ravel()
    x = img.reshape(1, 28, 28, 1)
    x = x / 255.0

    preds = model.predict([x])
    preds = np.argmax(preds, axis=1)
    if preds[0] == 0:
        preds = "The number is 0"
    elif preds[0] == 1:
        preds = "The number is 1"
    elif preds[0] == 2:
        preds = "The number is 2"
    elif preds[0] == 3:
        preds = "The number is 3"
    elif preds[0] == 4:
        preds = "The number is 4"
    elif preds[0] == 5:
        preds = "The number is 5"
    elif preds[0] == 6:
        preds = "The number is 6"
    elif preds[0] == 7:
        preds = "The number is 7"
    elif preds[0] == 8:
        preds = "The number is 8"
    elif preds[0] == 9:
        preds = "The number is 9"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(os.path.abspath("__file__"))
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=False)
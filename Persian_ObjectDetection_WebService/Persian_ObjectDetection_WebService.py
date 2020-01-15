# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:47:34 2019
A Simple code for a Persian Object Recognition Web Service using the flask framework.
@author: Abdolkarim Saeedi
"""
import os.path
import cv2
import os
from Detector import Predict
from flask import Flask, request, render_template, send_from_directory

__author__ = 'Abdolkarim Saeedi'
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
    execution_path = target
    print(execution_path)
    image = Predict(os.path.join(execution_path, filename))
    print(image.shape)
    print('predicted')
    out_image = cv2.imwrite(os.path.join(execution_path,  "flask"+filename), image)
    print('wrote out the image')
    print('flask'+filename)
    return render_template("complete_display_image.html", image_name="flask"+filename)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


if __name__ == "__main__":
    app.run(port=4555, debug=True)

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:47:34 2019
A Simple code for a Persian Object Recognition Web Service using the flask framework.
@author: Abdolkarim Saeedi
"""
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
import arabic_reshaper
from bidi.algorithm import get_display


def Predict(imagePath):
    Confidence, Threshold, Width, Height = 0.01, 0.3, 416, 416
    color1 = np.random.randint(0,255, size = 255)
    color2 = np.random.randint(0,255, size = 255)
    color3 = np.random.randint(0,255, size = 255)
    font = ImageFont.truetype('Sahel.ttf', 40)
    labels = ["شخص","دوچرخه","ماشین","موتور","هواپیما","اتوبوس","قطار","کامیون","قایق","چراغ راهنمایی","شیر آتش نشانی","علامت توقف","پارک متر","نیمکت","پرنده","گربه","سگ","اسب","گوسفند","گاو","فیل","خرس","گورخر","زرافه","کوله پشتی","چتر","کیف دستی","کروات","چمدان","فریزبی","اسکی","برد اسکی","توپ","بادبادک","چوب بیسبال","دستکش بیسبال","اسکیت برد","تخته شنا","راکت تنیس","بطری","بطری شراب","فنجان","چنگال","چاقو","قاشق","ظرف","موز","سیب","ساندویچ","پرتقال","برکلی","هویج","هات داگ","پیتزا","دونات","کیک","صندلی","مبل","گیاه کاشتنی","تخت","میز نهارخوری","توالت","تلویزیون","لپتاپ","موس","ریموت","کیبورد","گوشی","مایکروویو","گاز","تستر","سینک","یخچال","کتاب","ساعت","گلدان","قیچی","عروسک خرسی","سشوار","مسواک"]
    trained_model_configuration = "yolov3.cfg"
    trained_model_weights = "yolov3.weights"
    model = cv2.dnn.readNetFromDarknet(trained_model_configuration, trained_model_weights)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    modelLayers = model.getLayerNames()
    findOutputLayers = lambda model : [modelLayers[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    def boxPlot(labelIdx, confidence, left, top, right, bottom, i):
        clr1 = color1[i]
        clr2 = color2[i]
        clr3 = color3[i]
        d = ImageDraw.Draw(picture)
        d.rectangle([(left, top), (right, bottom)], None, (clr1, clr2, clr3), 3)    
        if labels:
            assert(labelIdx < len(labels))
            reshaped_text = arabic_reshaper.reshape(labels[labelIdx])
            bidi_text = get_display(reshaped_text)
        d.text((left, top - 5), bidi_text, font = font, fill=(clr1,clr2,clr3))

    def boxDet(image, model_outputs):
        imHeight = image.shape[0]
        imWidth = image.shape[1]
        classIds = []
        confidences = []
        boxes = []
        for output in model_outputs:
            for prediction in output:
                scores = prediction[5:]
                labelIdx = np.argmax(scores)
                confidence = scores[labelIdx]
                if confidence > Confidence:
                    center_x = int(prediction[0] * imWidth)
                    center_y = int(prediction[1] * imHeight)
                    width = int(prediction[2] * imWidth)
                    height = int(prediction[3] * imHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(labelIdx)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, Confidence, Threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            boxPlot(classIds[i], confidences[i], left, top, left + width, top + height, i)
    if (imagePath):
        frame = cv2.imread(imagePath)
        #print(frame.shape)
        outputFile = imagePath[:-4]+'_Object_Detector.jpg'


        picture = Image.fromarray(frame)

        blob = cv2.dnn.blobFromImage(frame, 1/255, (Width, Height), [0,0,0], 1, crop=False)
        model.setInput(blob)
        model_outputs = model.forward(findOutputLayers(model))
        boxDet(frame, model_outputs)
        if (imagePath):
            #print('image')
            opencvimage = np.array(picture)
    #print('Done!')
    return opencvimage

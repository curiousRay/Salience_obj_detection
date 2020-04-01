# -*- coding:utf-8 -*-
"""
@author: Lihao Lei
@license: GPL-3.0
@contact: leilei199708@gmail.com
@file: detector.py
@time: 2020/3/25
@desc: create bounding box for salience region
"""
import args as args
import cv2
import numpy as np

image = cv2.imread('./single_salience_map/test2.png')

saliency = cv2.saliency.ObjectnessBING_create()

# compute the bounding box predictions used to indicate saliency
(success, saliencyMap) = saliency.computeSaliency(image)
numDetections = saliencyMap.shape[0]

# loop over the detections
for i in range(0, min(numDetections, args["max_detections"])):
    # extract the bounding box coordinates
    (startX, startY, endX, endY) = saliencyMap[i].flatten()

    # randomly generate a color for the object and draw it on the image
    output = image.copy()
    color = np.random.randint(0, 255, size=(3,))
    color = [int(c) for c in color]
    cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)
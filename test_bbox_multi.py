# -*- coding:utf-8 -*-
"""
@author: Lihao Lei
@license: GPL-3.0
@contact: leilei199708@gmail.com
@file: test_bbox_multi.py
@desc: 使用annotations对测试集进行标注
"""

import numpy as np
from PIL import Image, ImageDraw

root_dir = "./dataset/VisDrone2019-DET-train/"
annotations_dir = root_dir + "annotations/"
image_dir = root_dir + "images/"
profile_name = "0000002_00005_d_0000014"

classes = (
    'ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus',
    'motor', 'others')

with open(annotations_dir + profile_name + '.txt', 'r') as f:
    lines = f.readlines()
    boxes = []
    for idx in range(len(lines)):  # 遍历该图中所有对象标记
        box = np.zeros((len(lines), 5), dtype=np.uint16)
        line = lines[idx].split(',')  # [x,y,w,h,score,object_category,truncation,occlusion]
        x1 = int(line[0]) - 1  # Make pixel indexes 0-based
        y1 = int(line[1]) - 1  # X轴向右，Y轴向下
        x2 = x1 + int(line[2])
        y2 = y1 + int(line[3])
        label = classes[int(line[5])]
        boxes.append([x1, y1, x2, y2, label])

    # print(boxes)

img = Image.open(image_dir + profile_name + '.jpg')
draw = ImageDraw.Draw(img)
for ix in range(len(boxes)):
    draw.rectangle([boxes[ix][0], boxes[ix][1], boxes[ix][2], boxes[ix][3]], outline=(255, 255, 0))
    draw.text([boxes[ix][0], boxes[ix][1]], boxes[ix][4], (255, 0, 0))

img.save(root_dir + profile_name + '.png')

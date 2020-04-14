# -*- coding:utf-8 -*-
"""
@author: Lihao Lei
@license: GPL-3.0
@contact: leilei199708@gmail.com
@file: test_bbox_single.py
@desc:
"""
import numpy as np
import single_salience_map.pySaliencyMap as pySaliencyMap
from PIL import Image, ImageDraw

class_tag = "bike1"

root_dir = "./dataset/UAV123_10fps/"
annotations_dir = root_dir + "anno/UAV123_10fps/"
image_dir = root_dir + "data_seq/UAV123_10fps/" + class_tag + "/"


with open(annotations_dir + class_tag + '.txt', 'r') as f:
    lines = f.readlines()

    boxes = []
    for idx in range(len(lines)):  # 遍历该文件夹下所有anno文件
        box = np.zeros((len(lines), 5), dtype=np.uint16)
        line = lines[idx].split(',')  # [x,y,w,h,score,object_category,truncation,occlusion]
        x1 = int(line[0]) - 1  # Make pixel indexes 0-based
        y1 = int(line[1]) - 1  # X轴向右，Y轴向下
        x2 = x1 + int(line[2])
        y2 = y1 + int(line[3])
        label = class_tag
        boxes.append([x1, y1, x2, y2, label])

    print(boxes)

img = Image.open(image_dir + '000001' + '.jpg')
draw = ImageDraw.Draw(img)
# for ix in range(len(boxes)):
# 仅绘制第一张
for ix in range(1):
    draw.rectangle([boxes[ix][0], boxes[ix][1], boxes[ix][2], boxes[ix][3]], outline=(255, 255, 0))
    draw.text([boxes[ix][0], boxes[ix][1]], boxes[ix][4], (255, 0, 0))

img.save(root_dir + '000001' + '.png')

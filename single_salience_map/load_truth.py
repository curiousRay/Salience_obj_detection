# -*- coding:utf-8 -*-
"""
@author: Lihao Lei
@license: GPL-3.0
@contact: leilei199708@gmail.com
@file: load_truth.py
@desc: 
"""
import pickle
import pySaliencyMapDefs as defs

def calc_diff(truth, bboxes):
    res = []
    return res

def gen_paths(num):
    return ["%s%06d%s" % (defs.IMG_DIR, x, ".jpg") for x in range(1, num + 1)]

def load(bboxes):
    f = open("temp/bboxes.pickle", "rb")
    raw = pickle.load(f)
    # {'../dataset/UAV123_10fps/data_seq/UAV123_10fps/bike1/000001.jpg': [[(1104, 510, 8, 8), (1104, 510, 8, 8),...]]
    f.close()
    res = []
    with open(defs.ANNO_FILE, 'r') as f:
        lines = f.readlines()
        for idx in range(len(lines)):  # 遍历所有图像帧
            line = lines[idx].split(',')  # [x,y,w,h]
            x1 = int(line[0]) - 1  # Make pixel indexes 0-based
            y1 = int(line[1]) - 1  # X轴向右，Y轴向下
            x2 = x1 + int(line[2])
            y2 = y1 + int(line[3])
            truth = [x1, y1, x2, y2]
            bboxes = raw.get(gen_paths(idx+1)[idx])[0]
            res.append(calc_diff(truth, bboxes))

    return res
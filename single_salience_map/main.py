# -*- coding:utf-8 -*-
"""
@author: Lihao Lei
@license: GPL-3.0
@contact: leilei199708@gmail.com
@file: bbox.py
@desc: main single object detection entrance
"""

import os
import cv2
import pySaliencyMap
import pySaliencyMapDefs as defs
import numpy as np
import pickle
from single_salience_map import load_truth, bbox
from imutils import paths
from multiprocessing import Pool
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import matplotlib
font = {'family': 'SimSun',
        'weight': 'regular',
        'size': '14'}
matplotlib.rc('font', **font)
import time


def process_images(payload):
    print("[INFO] starting process {}".format(payload["id"]))
    bboxes = {}

    for image_path in payload["input_paths"]:
        # read, cost 14 secs time
        # images = [cv2.imread(file) for file in glob.glob("../dataset/UAV123_10fps/data_seq/UAV123_10fps/bike1/*.jpg")]
        image = cv2.imread(image_path)
        # here influences structure of the output binary file

        l = bboxes.get(image_path, [])
        sm = pySaliencyMap.pySaliencyMap(image.shape[1], image.shape[0])  # img_width, img_height
        # saliency_map = sm.SMGetSM(image, [0.3, 0.3, 0.2, 0.2])  # computation

        print("hit!")
        # TODO: show graph, IMPORTANT TEST SWITCH
        # Adaboost学习器中的输入样本应为四个集合，每个集合代表一种特征，有1000个元素，每个元素代表框的中心点。
        # 以下为feed至四个弱分类器中样本值

        weights = [1, 0, 0, 0]
        saliency_map = sm.SMGetSM(image, weights)
        plt.subplot(2, 2, 1)
        plt.title('亮度显著图')
        plt.axis('off')
        plt.imshow(saliency_map, 'gray')
        # cv2.imshow('', saliency_map)

        weights = [0, 1, 0, 0]
        saliency_map = sm.SMGetSM(image, weights)
        plt.subplot(2, 2, 2)
        plt.title('颜色显著图')
        plt.axis('off')
        plt.imshow(saliency_map, 'gray')
        # cv2.imshow('', saliency_map)


        weights = [0, 0, 1, 0]
        saliency_map = sm.SMGetSM(image, weights)
        plt.subplot(2, 2, 3)
        plt.title('方向显著图')
        plt.axis('off')
        plt.imshow(saliency_map, 'gray')
        # cv2.imshow('', saliency_map)


        weights = [0, 0, 0, 1]
        saliency_map = sm.SMGetSM(image, weights)
        plt.subplot(2, 2, 4)
        plt.title('区域对比度显著图')
        plt.axis('off')
        plt.imshow(saliency_map, 'gray')
        # cv2.imshow('', saliency_map)

        weights = [.3, 0.3, 0.2, 0.2]
        saliency_map = sm.SMGetSM(image, weights)
        # cv2.imshow('', saliency_map)

        plt.subplots_adjust(wspace=0.1, hspace=0.28)
        plt.axis('off')

        # plt.show()

        saliency_map = sm.SMGetSM(image, [.3, 0.5, 0.1, 0.1])

        # 改这里

        l.append(bbox.bbox_rect(100, np.uint8(255 * saliency_map)))
        # saliency_map = sm.SMGetSM(image, [0, 1, 0, 0])
        # l.append(bbox.bbox_rect(100, np.uint8(255 * saliency_map)))
        # saliency_map = sm.SMGetSM(image, [0, 0, 1, 0])
        # l.append(bbox.bbox_rect(100, np.uint8(255 * saliency_map)))
        # saliency_map = sm.SMGetSM(image, [0, 0, 0, 1])
        # l.append(bbox.bbox_rect(100, np.uint8(255 * saliency_map)))

        bboxes[image_path] = l
        # print(int(time.time()))
    f = open(payload["output_path"], "wb")
    f.write(pickle.dumps(bboxes))
    f.close()


def chunk(list, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(list), n):
        # yield the current n-sized chunk to the calling function
        yield list[i: i + n]


def calc_center_dist(raw, truth):
    p1_x = raw[0] + raw[2] / 2
    p1_y = raw[1] + raw[3] / 2
    p2_x = (truth[0] + truth[2]) / 2
    p2_y = (truth[1] + truth[3]) / 2
    return np.sqrt(pow(p1_x - p2_x, 2) + pow(p1_y - p2_y, 2))


def calc_iou_avg(res):
    iou_sum = [0, 0, 0, 0]
    for idx, img in enumerate(res):
        iou_sum[0] = iou_sum[0] + res[idx][0][1]
        iou_sum[1] = iou_sum[1] + res[idx][1][1]
        iou_sum[2] = iou_sum[2] + res[idx][2][1]
        iou_sum[3] = iou_sum[3] + res[idx][3][1]
    return [avg/IMG_NUM for avg in iou_sum]


def gen_paths(num):
    return ["%s%06d%s" % (defs.IMG_DIR, x, ".jpg") for x in range(1, num + 1)]


def gen_bboxes():
    procs = PROCESS_NUM if PROCESS_NUM > 0 else os.cpu_count()
    img_num_per_proc = int(np.ceil(IMG_NUM / float(procs)))
    chunked_paths = list(chunk(gen_paths(IMG_NUM), img_num_per_proc))
    # print(chunked_paths[0])

    payloads = []
    for (i, imagePaths) in enumerate(chunked_paths):
        outputPath = os.path.sep.join(["temp", "proc_{}.pickle".format(i)])
        # construct a dictionary of data for the payload, then add it
        data = {
            "id": i,
            "input_paths": imagePaths,
            "output_path": outputPath
        }
        payloads.append(data)

    print("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process_images, payloads)
    # close the pool and wait for all processes to finish
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")

    # fetch multi-processed data and combine them
    bboxes = {}
    # loop over all pickle files in the output directory
    for p in paths.list_files("temp", validExts=".pickle", ):
        # load the contents of the dictionary
        data = pickle.loads(open(p, "rb").read())
        # print(data)  # {xxx.jpg:[[(), (), ...], [(), (), ...]], yyy.jpg:[[(), (), ...], [(), (), ...]]}
        # loop over the hashes and image paths in the dictionary
        for (tempPaths, tempH) in data.items():
            # grab all image paths with the current hash, add in the
            # image paths for the current pickle file, and then
            # update our hashes dictionary
            imagePaths = bboxes.get(tempPaths, [])
            imagePaths.extend(tempPaths)
            print(tempH)
            # bboxes[tempPaths] = [tempH[0], tempH[1], tempH[2], tempH[3]]
            bboxes[tempPaths] = [tempH[0]]
    # serialize the hashes dictionary to disk
    print("[INFO] serializing hashes...")
    # print(bboxes)
    f = open("temp/bboxes.pickle", "wb")
    f.write(pickle.dumps(bboxes))
    # print(bboxes)
    f.close()

    # print(bboxes)
    return bboxes


# main
if __name__ == '__main__':
    PROCESS_NUM = -1  # number of processes to be created, use -1 to deploy all cores
    IMG_NUM = len([lists for lists in os.listdir(defs.IMG_DIR)])

    # begin regressor loop
    bboxes = gen_bboxes()
    # print(bboxes)
    truths = load_truth.load(bboxes)
    # print('truth')
    # print(truths)
    values = list(bboxes.values())  # values为多张图片的bbox值，每个图片对应四个特征
    res = bbox.bbox_judge(values, truths)  # bbox.values() is an object
    print(res)
    truths = truths[0]
    # img = cv2.imread('000017.jpg')
    # cv2.imshow('test', img)
    # boundRect = res[0][0][0]
    # color = (255, 255, 255)#baise
    # color_truth = (255,255,0)#lanse
    #
    # #检测到的
    #
    # cv2.rectangle(img, (int(boundRect[0]), int(boundRect[1])), (int(boundRect[0] + boundRect[2]), int(boundRect[1] + boundRect[3])), color, 2)
    #
    # #真实的
    # cv2.rectangle(img, (int(truths[0]), int(truths[1])), (int(truths[2]), int(truths[3])), color_truth, 2)
    #
    # cv2.imshow('Contours', img)
    cv2.waitKey()

    # X = np.zeros([IMG_NUM, 4])
    #
    # for idx, img in enumerate(res):
    #     X[idx][0] = calc_center_dist(res[idx][0][0], truths[idx])
    #     X[idx][1] = calc_center_dist(res[idx][1][0], truths[idx])
    #     X[idx][2] = calc_center_dist(res[idx][2][0], truths[idx])
    #     X[idx][3] = calc_center_dist(res[idx][3][0], truths[idx])
    #
    # iou_avg = calc_iou_avg(res)
    # y = np.asarray([0, 0])
    # X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    # regr = AdaBoostRegressor(random_state=0, n_estimators=100)
    # regr.fit(X, y)
    #
    # print(regr.feature_importances_)

    # cv2.waitKey(0)
    # end regressor loop
    cv2.destroyAllWindows()

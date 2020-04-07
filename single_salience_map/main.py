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
import numpy as np
import pickle
import bbox
from imutils import paths
from multiprocessing import Pool
import time


def process_images(payload):
    print("[INFO] starting process {}".format(payload["id"]))
    hashes = {}

    for image_path in payload["input_paths"]:
        # read, cost 14 secs time
        # images = [cv2.imread(file) for file in glob.glob("../dataset/UAV123_10fps/data_seq/UAV123_10fps/bike1/*.jpg")]
        image = cv2.imread(image_path)
        # here influences structure of the output binary file

        l = hashes.get(image_path, [])
        sm = pySaliencyMap.pySaliencyMap(image.shape[1], image.shape[0])  # img_width, img_height
        saliency_map = sm.SMGetSM(image)
        res = bbox.bbox_rect(100, np.uint8(255 * saliency_map))
        l.append(res)
        hashes[image_path] = l
        # print(int(time.time()))

    f = open(payload["output_path"], "wb")
    f.write(pickle.dumps(hashes))
    f.close()
    # for img in images:
    #     # initialize
    #     sm = pySaliencyMap.pySaliencyMap(img.shape[1], img.shape[0])  # img_width, img_height
    #     # computation
    #     saliency_map = sm.SMGetSM(img)
    #
    #     print(bbox.bbox_rect(100, np.uint8(255 * saliency_map)))


def chunk(list, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(list), n):
        # yield the current n-sized chunk to the calling function
        yield list[i: i + n]


def gen_paths(num):
    return ["%s%06d%s" % (IMG_DIR, x, ".jpg") for x in range(1, num + 1)]


# main
if __name__ == '__main__':
    PROCESS_NUM = -1  # number of processes to be created
    IMG_DIR = "../dataset/UAV123_10fps/data_seq/UAV123_10fps/bike1/"

    IMG_NUM = len([lists for lists in os.listdir(IMG_DIR)])
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
    hashes = {}
    # loop over all pickle files in the output directory
    for p in paths.list_files("temp", validExts=".pickle", ):
        # load the contents of the dictionary
        data = pickle.loads(open(p, "rb").read())
        # loop over the hashes and image paths in the dictionary
        for (tempPaths, tempH) in data.items():
            # grab all image paths with the current hash, add in the
            # image paths for the current pickle file, and then
            # update our hashes dictionary
            imagePaths = hashes.get(tempPaths, [])
            imagePaths.extend(tempPaths)
            hashes[tempPaths] = tempH
    # serialize the hashes dictionary to disk
    print("[INFO] serializing hashes...")
    print(hashes)
    f = open("temp/bboxes.pickle", "wb")
    f.write(pickle.dumps(hashes))
    f.close()
    #    cv2.waitKey(0)
    cv2.destroyAllWindows()

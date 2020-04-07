# -------------------------------------------------------------------------------
# Name:        main
# Purpose:     Testing the package pySaliencyMap
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     April 24, 2014
# Copyright:   (c) Akisato Kimura 2014-
# Licence:     All rights reserved
# -------------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
import pySaliencyMap
import numpy as np
import pickle
import glob
import bbox
from imutils import paths
import os
from multiprocessing import Pool


def process_images(payload):
    print("[INFO] starting process {}".format(payload["id"]))
    hashes = {}

    for image_path in payload["input_paths"]:
        # read, cost 14 secs time
        # images = [cv2.imread(file) for file in glob.glob("../dataset/UAV123_10fps/data_seq/UAV123_10fps/bike1/*.jpg")]
        image = cv2.imread(image_path)
        # here influences structure of the output binary file
        h = dhash(image)
        h = convert_hash(h)
        l = hashes.get(image_path, [])
        l.append(h)
        hashes[image_path] = l
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


def dhash(image, hashSize=8):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def convert_hash(h):
    # convert the hash to NumPy's 64-bit float and then back to
    # Python's built in int
    return int(np.array(h, dtype="float64"))


def chunk(list, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(list), n):
        # yield the current n-sized chunk to the calling function
        yield list[i: i + n]


def gen_paths(num):
    return ["%s%06d%s" % (IMG_DIR, x, ".jpg") for x in range(1, num + 1)]


# main
if __name__ == '__main__':
    PROCESS_NUM = 2  # number of processes to be created
    IMG_DIR = "../dataset/UAV123_10fps/data_seq/UAV123_10fps/bike1/"

    procs = PROCESS_NUM if PROCESS_NUM > 0 else os.cpu_count()
    img_num_per_proc = int(np.ceil(1029 / float(procs)))

    chunked_paths = list(chunk(gen_paths(1029), img_num_per_proc))
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
    # print(hashes)
    print("[INFO] serializing hashes...")
    f = open("temp/bboxes.pickle", "wb")
    f.write(pickle.dumps(hashes))
    f.close()
    #    cv2.waitKey(0)
    cv2.destroyAllWindows()

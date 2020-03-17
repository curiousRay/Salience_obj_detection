# -*- coding:utf-8 -*-
"""
@author: Lihao Lei
@license: GPL-3.0
@contact: leilei199708@gmail.com
@file: image.py
@time: 2020/3/17
@desc: Initialization of images.
"""

import numpy as np
import os
import pylab

from matplotlib import pyplot
from matplotlib import image


class Image:

    def __init__(self, filename=None, label=None, mat=None):
        if filename is not None:
            self.imgName = filename
            self.img = image.imread(filename)

            if len(self.img.shape) == 3:
                self.img = self.img[:, :, 1]

        else:
            assert mat is not None
            self.img = mat

        self.label = label

        # self.stdImg  = Image._normalization(self.img)
        # self.iimg    = Image._integrateImg(self.stdImg)
        # self.vecImg  = self.iimg.transpose().flatten()
        self.vecImg = Image._integrate_img(Image._normalization(self.img)).transpose().flatten()

    @staticmethod
    def _integrate_img(img):

        assert img.__class__ == np.ndarray

        row, col = img.shape
        # @iImg is integrated image of normalized image @self.stdImg
        iImg = np.zeros((row, col))

        """
        for i in range(0, row):
            for j in range(0, col):
                if j == 0:
                    iImg[i][j] = image[i][j]
                else:
                    iImg[i][j] = iImg[i][j - 1] + image[i][j]

        for j in range(0, col):
            for i in range(1, row):
                iImg[i][j] += iImg[i - 1][j]
        """

        iImg = img.cumsum(axis=1).cumsum(axis=0)
        return iImg

    @staticmethod
    def _normalization(img):

        assert img.__class__ == np.ndarray

        row, col = img.shape

        # stdImg standardized image
        stdImg = np.zeros((row, col))
        """
            What image.sum() do is the same as the following code 
        but more faster than this.

        for i in range(self.Row):
            for j in range(self.Col):
                sigma += image[i][j]
        """
        # sigma = image.sum()

        meanVal = img.mean()
        stdVal = img.std()
        if stdVal == 0:
            stdVal = 1

        stdImg = (img - meanVal) / stdVal

        return stdImg

    @staticmethod
    def show(img=None):
        if img is None:
            return
        pyplot.matshow(img)
        pylab.show()


class ImageSet:
    def __init__(self, imgDir=None, label=None, sampleNum=None):

        assert isinstance(imgDir, str)

        self.imgDir = imgDir
        self.fileList = os.listdir(imgDir)
        self.fileList.sort()

        if sampleNum is None:
            self.sampleNum = len(self.fileList)
        else:
            self.sampleNum = sampleNum

        self.curFileIdx = self.sampleNum
        self.label = label

        self.images = [None for _ in range(self.sampleNum)]

        processed = -25.
        for i in range(self.sampleNum):
            self.images[i] = Image(imgDir + self.fileList[i], label)

            if i % (self.sampleNum / 25) == 0:
                processed += 25.
                print("Loading ", processed, "%")

        print("Loading 100 %\n")

    def read_next_img(self):
        img = Image(self.imgDir + self.fileList[self.curFileIdx], self.label)
        self.curFileIdx += 1
        return img

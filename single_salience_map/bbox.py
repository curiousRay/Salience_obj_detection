# -*- coding:utf-8 -*-
"""
@author: Lihao Lei
@license: GPL-3.0
@contact: leilei199708@gmail.com
@file: bbox.py
@desc: get bounding box info from saliency region
"""
import random as rng
import cv2


def bbox_rect(val, img):
    # https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
    threshold = val
    ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    canny_output = cv2.Canny(binary, threshold, threshold * 2)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    # black background
    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    drawing = img
    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                      (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        # cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

    # Show in a window, IMPORTANT TEST SWITCH
    # cv2.imshow('Contours', drawing)
    cv2.waitKey()  # 防止窗口闪退
    return boundRect


def bbox_judge(raw, truth):
    print(truth)
    print(raw)
    return raw
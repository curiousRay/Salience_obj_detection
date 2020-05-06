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

    # TODO: Show in a window, IMPORTANT TEST SWITCH
    # cv2.imshow('Contours', drawing)

    cv2.waitKey()  # forbid unexpected window quit
    # print(boundRect)
    return boundRect


def bbox_iou(boxes, truth):
    """
    computing IoU of single image
    boxes: (x0, y0, w, h) -> (y0, x0, y1, x1)
    truth: (x0, y0, x1, y1) -> (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # print(boxes)
    # print(truth)
    res = []
    rec2 = [truth[1], truth[0], truth[3], truth[2]]
    S_rec2 = (rec2[3] - rec2[1]) * (rec2[2] - rec2[0])
    for box in boxes:
        rec1 = [box[1], box[0], box[1] + box[3], box[0] + box[2]]
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])

        # # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            res.append(0)
            continue
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            res.append((intersect / (S_rec1 + S_rec2 - intersect)))

    iou = max(list(set(res)))  # delete dumplication and get max IOU
    best_box = boxes[res.index(iou)]  # filter the best box
    # print(best_box, iou)

    return [best_box, iou]


def bbox_judge(raw, truths):
    # print(raw)
    # print(truths)
    res = []
    for i in range(len(truths)):
        a = bbox_iou(raw[i][0], truths[i])
        b = bbox_iou(raw[i][1], truths[i])
        c = bbox_iou(raw[i][2], truths[i])
        d = bbox_iou(raw[i][3], truths[i])
        res.append([a, b, c, d])
    return res

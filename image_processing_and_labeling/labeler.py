import numpy as np
import cv2
import sys
import os
from matplotlib import pyplot as plt

def crop_minAreaRect(img, rect):
    # get the parameter of the small rectangle
    center = (int(rect[0][0]), int(rect[0][1]))
    size = (int(rect[1][0]), int(rect[1][1]))
    angle = rect[2]

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop

pk_space = cv2.imread('./'+sys.argv[1])
pk_space = cv2.cvtColor(pk_space, cv2.COLOR_BGR2GRAY)
image = cv2.imread('./cars.jpeg')
labels = cv2.imread('./labels.jpeg')
labels = cv2.cvtColor(labels, cv2.COLOR_BGR2RGB)

_, bin_img = cv2.threshold(pk_space,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
for cnt in cnts:
    perim = cv2.arcLength(cnt, True)
    if perim > 200:
        rect = cv2.minAreaRect(cnt)
        croped_image = crop_minAreaRect(image, rect)
        croped_label = crop_minAreaRect(labels, rect)

        rows, cols = int(croped_label.shape[0]/2), int(croped_label.shape[1]/2)
        label = 'Occupied' if croped_label[rows][cols][0] == 206 else 'Empty'

        #cv2.imwrite('teste.jpeg', croped_image)
        plt.imshow(croped_image)
        plt.axis('off')
        _ = plt.title("Label: " + label)
        plt.show()
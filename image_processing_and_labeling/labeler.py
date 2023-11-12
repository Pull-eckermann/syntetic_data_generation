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

try:
    lots = os.walk(sys.argv[1])
except:
    print("Error: Please specify the patth to the parking lot directory")
    exit(1)

for path, _, files in lots:
    for img in files:
        if 'image' in img:
            image = cv2.imread(path + '/' + img)

            for pk_space_path in files:
                if "mask" in pk_space_path:
                    # Reads the image containing the individual parking space
                    pk_space = cv2.imread(path + '/' + pk_space_path)
                    pk_space = cv2.cvtColor(pk_space, cv2.COLOR_BGR2RGB)
                    pk_space_gray = cv2.cvtColor(pk_space, cv2.COLOR_BGR2GRAY)

                    _, bin_img = cv2.threshold(pk_space_gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    for cnt in cnts:
                        perim = cv2.arcLength(cnt, True)
                        if perim > 200:
                            rect = cv2.minAreaRect(cnt)
                            croped_image = crop_minAreaRect(image, rect)
                            croped_label = crop_minAreaRect(pk_space, rect)

                            rows, cols = int(croped_label.shape[0]/2), int(croped_label.shape[1]/2)
                            label = 'Occupied' if croped_label[rows][cols][0] > 100 else 'Empty'

                            pk_space_name = pk_space_path.replace('mask','pk-space')
                            cv2.imwrite('labeled/{}/{}'.format(label, pk_space_name), croped_image)
                            #plt.imshow(croped_image)
                            #plt.axis('off')
                            #_ = plt.title("Label: " + label)
                            #plt.show()
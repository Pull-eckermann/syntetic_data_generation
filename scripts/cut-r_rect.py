import cv2
import numpy as np
import json
import os
import sys

def segment_and_rotate_space(img, segmentation):
    cnt = []
    cnt.append((int(segmentation[0]), int(segmentation[1])))
    cnt.append((int(segmentation[2]), int(segmentation[3])))
    cnt.append((int(segmentation[4]), int(segmentation[5])))
    cnt.append((int(segmentation[6]), int(segmentation[7])))

    rect = cv2.minAreaRect(np.asarray(cnt))

    # get the parameter of the small rectangle
    center = rect[0]
    w, h = rect[1]
    angle = rect[2]
    
    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, (int(w), int(h)), center)
    #img_crop = img_rot[int(center[1] - h/2):int(center[1] + h/2), int(center[0] - w/2):int(center[0] + w/2)]

    return img_crop

json_file = open(sys.argv[1])
data = json.load(json_file)

images = data['images']
annotations = data['annotations']

for image in images:
    try:
        imgId = image['id']
        imgPath = image['file_name']
        datelist = image["date"]
        date = str(datelist[0]) + '-' + str(datelist[1]) + '-' + str(datelist[2])
        subset = image["subset"]
    except:
        continue

    imgPathOccupied = "CNRPark-EXT/" + subset + "/" + date + '/Occupied'
    imgPathEmpty = "CNRPark-EXT/" + subset + "/" + date + '/Empty'
    os.makedirs(imgPathOccupied, exist_ok = True)
    os.makedirs(imgPathEmpty, exist_ok = True)
    img = cv2.imread(imgPath)

    for annotation in annotations:
        if annotation["image_id"] == imgId:
            annotId = annotation["id"]
            label = annotation["category_id"]
            segmentation = annotation['segmentation'][0]
            imgRect = segment_and_rotate_space(img, segmentation)
            imgName = "{}.jpg".format(annotId)
            if label == 0:
                cv2.imwrite(imgPathEmpty + '/' + imgName, imgRect)
            else:
                cv2.imwrite(imgPathOccupied + '/' + imgName, imgRect)


import cv2
import numpy as np
import json
import os
import sys

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

    imgPathOccupied = "PKlot/" + subset + "/" + date + '/Occupied'
    imgPathEmpty = "PKlot/" + subset + "/" + date + '/Empty'
    os.makedirs(imgPathOccupied, exist_ok = True)
    os.makedirs(imgPathEmpty, exist_ok = True)
    img = cv2.imread(imgPath)

    for annotation in annotations:
        if annotation["image_id"] == imgId:
            annotId = annotation["id"]
            label = annotation["category_id"]
            bbox = annotation['bbox']
            imgRect = img[bbox[1] : bbox[1] + bbox[3] , bbox[0] : bbox[0] + bbox[2]]
            imgName = "{}.jpg".format(annotId)
            if label == 0:
                cv2.imwrite(imgPathEmpty + '/' + imgName, imgRect)
            else:
                cv2.imwrite(imgPathOccupied + '/' + imgName, imgRect)


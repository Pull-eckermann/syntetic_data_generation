import cv2
import numpy as np
import json
import os
import sys

def segment_bbox(img, cnt):
    x,y,w,h = cv2.boundingRect(cnt)
    img_rect = img[y:y+h, x:x+w]

    return img_rect

def segment_and_rotate_space(img, cnt):

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

def detect_occupancy(cropped_img):
    occupied = np.asarray([0, 17, 255])
    empty = np.asarray([16, 255, 0])

    (h, w) = cropped_img.shape[:2]
    color_to_compare = cropped_img[h//2, w//2]

    return np.array_equal(color_to_compare, occupied)

os.makedirs("../../Datasets-synthetic/{}-synthetic/Occupied".format(sys.argv[2]), exist_ok = True)
os.makedirs("../../Datasets-synthetic/{}-synthetic/Empty".format(sys.argv[2]), exist_ok = True)
occupied = 0
empty = 0
cont = 1

Lots = os.walk(sys.argv[1])
for path, dir, files in Lots:
    if files:
        image = cv2.imread(path + '/' + files[0])
        print(path + '/' + files[0])
        pk_spaces = cv2.imread(path + '/' + files[2])
        pk_spaces_gray = cv2.cvtColor(pk_spaces, cv2.COLOR_BGR2GRAY)
        h, w = pk_spaces_gray.shape[:2]
        pk_spaces_gray = pk_spaces_gray[:h-1, :w-1]
        valid_bbox = np.asarray([[[100, 100]], [[w-100, 100]], [[w-100, h-100]], [[100, h-100]]])

        _, thresh = cv2.threshold(pk_spaces_gray,1,255,cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, 1, 2)

        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = float(M['m10']/M['m00'])
            cy = float(M['m01']/M['m00'])
            if cv2.pointPolygonTest(valid_bbox, (cx, cy), False) == -1:
                continue

            #img_rect = segment_and_rotate_space(image, cnt)
            img_rect = segment_bbox(image, cnt)
            if img_rect.shape[:2][0] < 50 or img_rect.shape[:2][1] < 50:
                continue

            segmentation_rect = segment_and_rotate_space(pk_spaces, cnt)
            if detect_occupancy(segmentation_rect) == True:
                cv2.imwrite('../../Datasets-synthetic/{}-synthetic/Occupied/{}-{}.jpeg'.format(sys.argv[2], sys.argv[2], cont), img_rect)
            else:
                cv2.imwrite('../../Datasets-synthetic/{}-synthetic/Empty/{}-{}.jpeg'.format(sys.argv[2], sys.argv[2], cont), img_rect)
            cont += 1            

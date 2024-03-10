import cv2
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET

def segment_and_rotate_space(img_path, space):
    img = cv2.imread(img_path)

    # get the parameter of the small rectangle
    angle = space.find('./rotatedRect/angle')
    angle = int(angle.attrib['d'])

    center = space.find('./rotatedRect/center')
    x = int(center.attrib['x'])
    y = int(center.attrib['y'])
    center = (x ,y)

    size = space.find('./rotatedRect/size')
    w = int(size.attrib['w'])
    h = int(size.attrib['h'])
    size = (w ,h)

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical, and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop

def getBoundingBox(img_path, space):
    img = cv2.imread(img_path)

    points = space.find('./contour')
    cnt = []
    for point in points:
        cnt.append((int(point.attrib['x']), int(point.attrib['y'])))

    x,y,w,h = cv2.boundingRect(np.asarray(cnt))

    img_crop = img[y:y + h , x:x + w]

    return img_crop

# Certifies that the paths was passed in command the line
if len(sys.argv) < 2:
    print("Please, provide the path to PKLot Dataset")
    exit(0)

if "PKLot" not in sys.argv[1]:
    print("Please, provide the path to PKLot Dataset")
    exit(0)

#Walks througt PKLot directory
Lots = os.walk(sys.argv[1])
for path, _, files in Lots:
    for img in files:
        if img.endswith('.jpg'):
            xml = img.replace('.jpg', '.xml')
            img_path = path + '/' + img #Path to the Parking Lot image
            xml_path = path + '/' + xml #Path to the respective xml file
            try:
                xmltree = ET.parse(xml_path)
            except:
                continue
            PKspaces = xmltree.findall('./space')
            for space in PKspaces:
                PLspace_img_rt = getBoundingBox(img_path, space)
                
                new_img = img.replace('.jpg','')
                tag = '#{}.jpg'.format(space.attrib['id'])
                new_img =  new_img + tag
                if len(space.attrib) > 1:
                    if space.attrib['occupied'] == '0':
                        new_path = path + '\Empty\\'
                        new_path = new_path.replace('PKLot', 'PKLot_divided')
                        new_img_path = new_path + new_img
                        os.makedirs(new_path, exist_ok = True)
                        cv2.imwrite(new_img_path, PLspace_img_rt)
                    else:
                        new_path = path + '\Occupied\\'
                        new_path = new_path.replace('PKLot', 'PKLot_divided')
                        new_img_path = new_path + new_img
                        os.makedirs(new_path, exist_ok = True)
                        cv2.imwrite(new_img_path, PLspace_img_rt)

import cv2
import numpy as np
import json
import os
import sys

os.makedirs("cam9-synthetic/Occupied", exist_ok = True)
os.makedirs("cam9-synthetic/Empty", exist_ok = True)
cont = 1

Lots = os.walk(sys.argv[1])
for path, dir, files in Lots:
    if files:
        image = cv2.imread(path + '/' + files[0])
        print(path + '/' + files[0])
        json_file = open(path + '/' + files[1])
        data = json.load(json_file)

        annotations = data['captures'][0]['annotations']
        for annotation in annotations:
            if annotation['id'] == 'bounding box':
                for bouding_box in annotation['values']:
                    x, y = bouding_box['origin']
                    w, h = bouding_box['dimension']
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    if w >= 80 and h >= 80:
                        img_rect = image[y:y+h, x:x+w]
                        name = path.split("/")[-1]

                        if bouding_box['labelName'] == 'Occupied':
                            cv2.imwrite('cam9-synthetic/Occupied/cam9-{}.jpeg'.format(cont), img_rect)
                        else:
                            cv2.imwrite('cam9-synthetic/Empty/cam9-{}.jpeg'.format(cont), img_rect)
                        cont += 1

            

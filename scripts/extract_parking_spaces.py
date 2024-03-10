import cv2
import numpy as np
import json
import os
import sys

os.makedirs("UFPR05-syntetic-realistic/Occupied", exist_ok = True)
os.makedirs("UFPR05-syntetic-realistic/Empty", exist_ok = True)

Lots = os.walk(sys.argv[1])
for path, dir, files in Lots:
    if files:
        image = cv2.imread(path + '/' + files[0])
        print(path + '/' + files[0])
        json_file = open(path + '/' + files[2])
        data = json.load(json_file)

        annotations = data['captures'][0]['annotations']
        for i in range(0, len(annotations)):
            annotation = annotations[i]
            if annotation['id'] == 'Vaga':
                for bouding_box in annotation['values']:
                    x, y = bouding_box['origin']
                    w, h = bouding_box['dimension']
                    x = int(x)
                    y = int(y)
                    w = int(w)
                    h = int(h)
                    if w >= 60 and h >= 60:
                        img_rect = image[y:y+h, x:x+w]
                        name = path.split("\\")[-1]

                        if bouding_box['labelName'] == 'Ocupada':
                            cv2.imwrite('UFPR05-syntetic-realistic\Occupied\{}_{}.jpeg'.format(name, bouding_box['instanceId']), img_rect)
                        else:
                            cv2.imwrite('UFPR05-syntetic-realistic\Empty\{}_{}.jpeg'.format(name, bouding_box['instanceId']), img_rect)

            

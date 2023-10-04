import cv2
import os
import sys
import csv
from localbinarypatterns import LocalBinaryPatterns
from matplotlib import pyplot as plt


# Certifies that the paths was passed in command the line
if len(sys.argv) < 3:
    print("Please, provide the path to PKLot Dataset and name for extracted features")
    exit(0)

#contruct object to extract the features
desc = LocalBinaryPatterns(8, 1)

shown = False
#Walks througt PKLot directory
Lots = os.walk(sys.argv[1])
for path, _, files in Lots:
    for img in files:
        if  img.endswith('.jpg'):
            image = path + '/' + img #Path to the Parking Lot image
            image = cv2.imread(image)
            image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp, hist = desc.describe(image_grey)
            if shown is False:
                shown = True
                plt.figure(figsize=(10,6))
                plt.subplot(1,2,1)
                plt.imshow(image)
                plt.subplot(1,2,2)
                plt.imshow(lbp, 'gray')
                plt.show()
                plt.plot(hist,color = "blue")
                plt.show()
            
            with open(sys.argv[2]+'.csv', 'a', newline='') as file:
                hist_class = list(hist)
                if "Empty" in path:
                    hist_class.append(0)
                else:
                    hist_class.append(1)
                writer = csv.writer(file)
                writer.writerow(hist_class)            

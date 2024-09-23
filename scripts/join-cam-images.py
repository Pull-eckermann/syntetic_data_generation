import os
import sys
import math

days = os.listdir(sys.argv[1])

os.makedirs(sys.argv[1] + "/All-days/Empty", exist_ok = True)
os.makedirs(sys.argv[1] + "/All-days/Occupied", exist_ok = True)
dataset = sys.argv[1] + "/All-days/"

for day in days:
    folder = sys.argv[1] + '/' + day
    lots = os.listdir(folder)
    for lot in lots:
        lot_path = folder + '/' + lot
        images = os.listdir(lot_path)
        for image in images:
            img_path = lot_path + '/' + image
            print("Coping " + img_path)
            os.system('cp "{}" {}'.format(img_path, dataset + lot))
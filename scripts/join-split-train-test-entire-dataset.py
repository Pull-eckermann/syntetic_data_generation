import os
import sys
import math

cams = os.listdir(sys.argv[1])
img_cont = 1

os.makedirs(sys.argv[1] + "/Test/Empty", exist_ok = True)
os.makedirs(sys.argv[1] + "/Train/Empty", exist_ok = True)
os.makedirs(sys.argv[1] + "/Test/Occupied", exist_ok = True)
os.makedirs(sys.argv[1] + "/Train/Occupied", exist_ok = True)

for cam in cams:
    days = os.listdir(sys.argv[1] + '/' + cam)
    n_days = len(days)
    cont = 1

    for day in days:
        if cont <= math.ceil(n_days * 0.7):
            dataset = sys.argv[1] + '/Train/'
        else:
            dataset = sys.argv[1] + '/Test/'

        folder = sys.argv[1] + '/' + cam + '/' + day
        lots = os.listdir(folder)
        for lot in lots:
            lot_path = folder + '/' + lot
            images = os.listdir(lot_path)
            for image in images:
                img_path = lot_path + '/' + image
                print("Coping " + img_path)
                os.system('cp {} {}'.format(img_path, dataset + lot + '/' + '{}.jpg'.format(img_cont)))
                img_cont += 1

        cont += 1
import os
import sys
import math

cams = os.listdir(sys.argv[1])

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
            dataset = sys.argv[1] + '\Train'
        else:
            dataset = sys.argv[1] + '\Test'

        folder = sys.argv[1] + '\\' + cam + '\\' + day
        lots = os.listdir(folder)
        for lot in lots:
            if lot == 'occupied':
                os.system('copy .\{} .\{}'.format(folder + '\\' + lot, dataset + '\Occupied'))
            else:
                os.system('copy .\{} .\{}'.format(folder + '\\' + lot, dataset + '\Empty'))
        cont += 1

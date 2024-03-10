import os
import sys
import math

os.makedirs("unified-CNRPark/Train/Occupied", exist_ok = True)
os.makedirs("unified-CNRPark/Train/Empty", exist_ok = True)
os.makedirs("unified-CNRPark/Test/Occupied", exist_ok = True)
os.makedirs("unified-CNRPark/Test/Empty", exist_ok = True)

#Walks througt PKLot directory
days = os.listdir(sys.argv[1])
n_days = len(days)
cont = 1

for day in days:
    if cont <= math.ceil(n_days * 0.7):
        dataset = 'unified-CNRPark\Train'
    else:
        dataset = 'unified-CNRPark\Test'

    Lots = os.walk(sys.argv[1] + '\\' + day)
    for path, _, files in Lots:
        if files:
            if 'occupied' in path:
                dataset_path = dataset + '\Occupied'
            elif 'empty' in path :
                dataset_path = dataset + '\Empty'
            for img in files:
                image = path + '\\' + img #Path to the Parking Lot image
                os.system('copy .\{} .\{}'.format(image, dataset_path))
    cont += 1

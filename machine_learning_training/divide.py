import os
import sys

os.system('rm -rf Train')
os.system('rm -rf Test') 
os.makedirs("Train/Occupied", exist_ok = True)
os.makedirs("Train/Empty", exist_ok = True)
os.makedirs("Test/Occupied", exist_ok = True)
os.makedirs("Test/Empty", exist_ok = True)

dataset = 'Test'
cont = 0

#Walks througt PKLot directory
Lots = os.walk(sys.argv[1])
for path, _, files in Lots:
    if files:
        if dataset == 'Train' and cont >= 2:
            dataset = 'Test'
            cont = 0
        elif cont >= 2:
            dataset = 'Train'
            cont = 0

        if 'Occupied' in path:
            dataset_path = dataset+'/Occupied'
        else:
            dataset_path = dataset+'/Empty'
            
        for img in files:
            image = path + '/' + img #Path to the Parking Lot image
            os.system('cp {} {}'.format(image, dataset_path))
        cont += 1

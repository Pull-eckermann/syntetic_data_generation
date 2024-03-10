import os
import sys
import math

days = os.listdir(sys.argv[1])

os.makedirs(sys.argv[1] + "/All-days/Empty", exist_ok = True)
os.makedirs(sys.argv[1] + "/All-days/Occupied", exist_ok = True)

for day in days:
    folder = sys.argv[1] + '\\' + day
    lots = os.listdir(folder)
    for lot in lots:
        os.system('copy .\{} .\{}'.format(folder + '\\' + lot, sys.argv[1] + '\All-days\\' + lot))
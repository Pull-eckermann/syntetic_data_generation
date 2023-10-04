import sys
import csv

max_elem = 0
min_elem = 999999

try:
    with open(sys.argv[1], 'r+', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            row.pop()
            row = list(map(int, row))
            aux_min = min(row)
            aux_max = max(row)
            if aux_max > max_elem:
                max_elem = aux_max
            if aux_min < min_elem:
                min_elem = aux_min

        print("Min parameter: {}".format(min_elem))
        print("max_elem parameter: {}".format(max_elem))
except OSError as error:
    print(error)
except IndexError as error:
    print(error)
    
try:
    with open(sys.argv[1], 'r+', newline='') as file:
        reader = csv.reader(file)    
        with open('nomalized_'+sys.argv[1], 'a', newline='') as filewriter:
            writer = csv.writer(filewriter)
            for row in reader:
                tag = row.pop()
                new_row = []
                for elem in row:
                    new_elem = (int(elem) - min_elem) / (max_elem - min_elem)
                    new_row.append(new_elem)
                    
                new_row.append(tag)
                writer.writerow(new_row)

except OSError as error:
    print(error)
except IndexError as error:
    print(error)


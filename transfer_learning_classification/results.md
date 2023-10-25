***With model parking_lot

QA_Auto@Venkomputer:~/faculdade/syntetic_data_generation/transfer_learning_classification$ time python3 classify_lot.py ../../CNRPark-Patches/A
Total accuracy: 0.9413860103626943

real    1m37,416s
user    10m8,595s
sys     2m6,381s

QA_Auto@Venkomputer:~/faculdade/syntetic_data_generation/transfer_learning_classification$ time python3 classify_lot.py real-data/Test/
Total accuracy: 0.9603197048049988

real    11m23,738s
user    69m20,105s
sys     11m45,840s
QA_Auto@Venkomputer:~/faculdade/syntetic_data_generation/transfer_learning_classification$ time python3 classify_lot.py ../../CNRPark-Patches/B
Total accuracy: 0.6153126798388644

real    1m31,327s
user    8m37,416s
sys     1m53,815s

QA_Auto@Venkomputer:~/faculdade/syntetic_data_generation/transfer_learning_classification$ time python3 classify_lot.py /home/venkopad/faculdade/PKLot/PKLotSegmented/PUC/Sunny/2012-11-20/
Total accuracy: 0.9738636363636364

real    1m7,525s
user    6m22,348s
sys     1m21,218s

QA_Auto@Venkomputer:~/faculdade/syntetic_data_generation/transfer_learning_classification$ time python3 classify_lot.py /home/venkopad/faculdade/PKLot/
Total accuracy: 0.9919871794871795

real    1m57,751s
user    10m57,271s
sys     2m23,889s

QA_Auto@Venkomputer:~/faculdade/syntetic_data_generation/transfer_learning_classification$ time python3 classify_lot.py real-data/Train/
Total accuracy: 0.9676142816581488

real    13m16,181s
user    77m32,269s
sys     12m53,017s

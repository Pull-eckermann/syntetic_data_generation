#!/bin/bash

for i in {1..3}
do
    echo "######################################################################### $i EXECUTION #########################################################################"
    echo ">>> Training PKlot v3 mixed"
    python3 retrain.py PKlot v3 -m
    #echo ">>> Training CNR v3 mixed"
    #python3 retrain.py CNR v3 -m
    #echo ">>> Training CNR v3"
    #python3 retrain.py CNR v3
    #echo ">>> Training PKlot v3"
    #python3 retrain.py PKlot v3
    #echo ">>> Training all-synthetic v3"
    #python3 retrain.py all-synthetic v3

    python3 run-tests.py
    echo "################################################################################################################################################################"
done

#!/bin/bash
#PBS -l ncpus=20
#PBS -N Analysis_clean 
#PBS -o Output/
#PBS -j oe
#PBS -m ae

cd ~/FlowerRecog
python3 Analysis_clean.py

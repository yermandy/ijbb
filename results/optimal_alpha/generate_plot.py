import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

def find_appropriate(qualities, appropriate, curves):
    for quality in qualities:
        quality = qualities[quality].tolist()
        if quality['label'] not in appropriate:
            continue
        quality['color'] = None
        if quality['label'] == 'CNN-FQ 24 epochs 1.5':
            quality['label'] = 'CNN-FQ | λ=1.5'
        if quality['label'] == 'L2-norm 1.75':
            quality['label'] = 'L2–norm | λ=1.75'
        if quality['label'] == 'retina detector score 0.5':
            quality['label'] = 'RetinaFace | λ=0.5'
        curves.append(quality)

qualities_1_5 = np.load(f"results/optimal_alpha/results_1.5.npz", allow_pickle=True)
qualities_0_5 = np.load(f"results/optimal_alpha/results_0.5.npz", allow_pickle=True)
qualities_1_75 = np.load(f"results/optimal_alpha/results_1.75.npz", allow_pickle=True)

curves = []
appropriate = ['L2-norm 1.75', 'CNN-FQ 24 epochs 1.5', 'retina detector score 0.5']

for quality in qualities_0_5:
    quality = qualities_0_5[quality].tolist()
    quality['color'] = None
    if quality['label'] == 'SE-ResNet-50':
        curves.append(quality)

find_appropriate(qualities_1_75, appropriate, curves)
find_appropriate(qualities_0_5, appropriate, curves)
find_appropriate(qualities_1_5, appropriate, curves)
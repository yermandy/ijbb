import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

def plot_tar_at_far(curves, errorbar=False, plot_SOTA=False):
    fig, ax = plt.subplots(1, 1, figsize=(8,5), tight_layout=True)
    for i, roc in enumerate(curves):
        far = roc['far']
        tar = roc['tar_mean']
        ax.plot(far, tar, lw=2, label=roc['label'], color=roc['color'], alpha=0.95)
        if errorbar:
            indecies = np.array([1,3,10,30,90,300,900])
            x = far[indecies] + np.linspace(0.000003 * (i+1), 0.00006 * (i+1), len(indecies))
            y = tar[indecies]
            e = roc['tar_std'][indecies]
            ax.errorbar(x, y, yerr=e, color=roc['color'], alpha=0.7, fmt=' ', capsize=2)
    if plot_SOTA:
        far = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        tar = [0.762, 0.863, 0.926, 0.963, 0.989, 1]
        ax.plot(far, tar, lw=2, label='SE-GV-4-g1', color='black', alpha=0.95)
    ax.set_xscale("log")
    ax.set_xlim(0.00001, 0.1)
    ax.set_ylim(0.65, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper left')    
    ax.set_xlabel('False Accept Rate')
    ax.set_ylabel('True Accept Rate')
    ax.grid(True, linestyle='dotted', which="both")
    fig1 = plt.gcf()
    plt.show()
    fig1.set_size_inches((8 * 1.5, 5 * 1.5))
    fig1.savefig('results/optimal_alpha/ijbb_optimal_lambda.png', dpi = 300, bbox_inches='tight')


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


find_appropriate(qualities_1_5, appropriate, curves)
find_appropriate(qualities_0_5, appropriate, curves)
find_appropriate(qualities_1_75, appropriate, curves)


for quality in qualities_0_5:
    quality = qualities_0_5[quality].tolist()
    quality['color'] = None
    if quality['label'] == 'SE-ResNet-50':
        curves.append(quality)

plot_tar_at_far(curves)
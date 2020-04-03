import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

def custom_plot(x, y, x_label, y_label, folder="plots", y_line=None, color="black", file_name="plot"):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, lw=2, color=color, tight_layout=True)
    if y_line is not None:
        plt.axvline(x=y_line, color='red')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(True, linestyle='dotted')
    os.makedirs(f'results/{folder}', exist_ok=True)
    plt.savefig(f'results/{folder}/{file_name}', dpi = 300)
    plt.close(fig)

def plot_auc_removed(auc, removed, quality_name, selected_removed, i='all'):
    custom_plot(removed, auc, "Removed", "AUC", quality_name, selected_removed, file_name=f'{i}_auc_removed_{quality_name}.png')

def plot_auc_th(auc, th, quality_name, best_threshold, i='all'):
    custom_plot(th, auc, "Thresholds", "AUC", quality_name, best_threshold, file_name=f'{i}_auc_thresholds_{quality_name}.png')

def plot_one_face_th(one_faces, th, quality_name, best_threshold, i='all'):
    custom_plot(th, one_faces, "Thresholds", "Sets with one face", quality_name, best_threshold, file_name=f'{i}_one_face_thresholds_{quality_name}.png')

def plot_points(points):
    fig, ax = plt.subplots(1, 1, figsize=(8,5), dpi=100, tight_layout=True)
    for point in points:
        ax.plot(point['x'], point['y'], 'o', lw=2, c=point['c'])
    ax.set_xscale("log")
    ax.set_xlim(0.0001, 0.01)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('False Accept Rate')
    ax.set_ylabel('True Accept Rate')
    ax.grid(True, linestyle='dotted', which="both")
    plt.show()

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
    fig1.set_size_inches((12, 7.5))
    fig1.savefig('results/tar@far.png', dpi = 300, bbox_inches='tight')
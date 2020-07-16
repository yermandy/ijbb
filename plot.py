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
    ax.set_xlabel('False Acceptance Rate')
    ax.set_ylabel('True Acceptance Rate')
    ax.grid(True, linestyle='dotted', which="both")
    plt.show()

def plot_roc(curves, plot_SOTA=False, fontsize=25):
    fig, ax = plt.subplots(1, 1, figsize=(8,5), tight_layout=True)
    for i, roc in enumerate(curves):
        far = roc['far']
        tar = roc['tar']
        color = roc['color'] if 'color' in roc else None
        ax.plot(far, tar, lw=2, label=roc['label'], color=color, alpha=0.95)
    if plot_SOTA:
        far = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        tar = [0.762, 0.863, 0.926, 0.963, 0.989, 1]
        ax.plot(far, tar, lw=2, label='SE-GV-4-g1', color='black', alpha=0.95)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, .01)
    ax.set_ylim(top=1)
    # ax.set_ylim(0.65, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='lower right')
    ax.set_xlabel('False Acceptance Rate')
    ax.set_ylabel('True Acceptance Rate')
    ax.grid(True, linestyle='dotted', which="both")
    fig1 = plt.gcf()
    plt.show()
    ax.set_xlabel('False Acceptance Rate', fontsize=fontsize)
    ax.set_ylabel('True Acceptance Rate', fontsize=fontsize)
    ax.legend(loc='lower right', fontsize=18)
    fig1.set_size_inches((8, 5))
    fig1.savefig('results/ROC.png', dpi = 300, bbox_inches='tight')

def plot_cmc(curves, fontsize=18):
    fig, ax = plt.subplots(1, 1, figsize=(8,5), tight_layout=True)
    for i, curve in enumerate(curves):
        cmc_prob = curve['cmc']
        cmc_rank = np.arange(1, len(cmc_prob) + 1)
        color = curve['color'] if 'color' in curve else None
        ax.plot(cmc_rank, cmc_prob, lw=2, label=curve['label'], color=color, alpha=0.95)
    ax.set_xscale("log")
    ax.set_xlim(1, 1e3)
    ax.set_ylim(top=1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='lower right')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Identification Rate')
    ax.grid(True, linestyle='dotted', which="both")
    fig1 = plt.gcf()
    plt.show()
    ax.set_xlabel('Rank', fontsize=fontsize)
    ax.set_ylabel('Identification Rate', fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    fig1.set_size_inches((8, 5))
    fig1.savefig('results/CMC.png', dpi = 300, bbox_inches='tight')

def plot_det(curves, fontsize=18):
    fig, ax = plt.subplots(1, 1, figsize=(8,5), tight_layout=True)
    for i, curve in enumerate(curves):
        fpir = curve['fpir']
        fnir = curve['fnir']
        color = curve['color'] if 'color' in curve else None
        ax.plot(fpir, fnir, lw=2, label=curve['label'], color=color, alpha=0.95)
    ax.set_xscale("log")
    ax.set_ylim(top=1, bottom=0)
    ax.set_xlim(1e-4, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='upper right')
    ax.set_xlabel('False Positive Identification Rate')
    ax.set_ylabel('False Negative Identification Rate')
    ax.grid(True, linestyle='dotted', which="both")
    fig1 = plt.gcf()
    plt.show()
    ax.set_xlabel('False Positive Identification Rate', fontsize=fontsize)
    ax.set_ylabel('False Negative Identification Rate', fontsize=fontsize)
    ax.legend(loc='upper right', fontsize=fontsize)
    fig1.set_size_inches((8, 5))
    fig1.savefig('results/DET.png', dpi = 300, bbox_inches='tight')


if __name__ == "__main__":
    # curves_dict = np.load(f"results/verification_results.npz", allow_pickle=True)
    # curves_dict = np.load(f"results/identification_results.npz", allow_pickle=True)
    curves_dict = np.load(f"results/curves/cov_forehead.npz", allow_pickle=True)

    curves = []

    ''' 
    # naive qualities
    curves_dict = dict(curves_dict)
    curves_dict['arr_3'] = curves_dict.pop('arr_3')
    curves_dict['arr_1'] = curves_dict.pop('arr_1')
    curves_dict['arr_2'] = curves_dict.pop('arr_2')
    curves_dict['arr_0'] = curves_dict.pop('arr_0')
    # '''

    for quality in curves_dict:
        quality = curves_dict[quality].tolist()
        # if quality['label'] == 'SE-ResNet-50':
        #     quality['label'] = 'Averaging'
        # elif 'L2-norm' in quality['label']:
        #     quality['label'] = f"WA + {quality['label']}"
        # elif 'CNN-FQ' in quality['label']:
        #     quality['label'] = f"WA + {quality['label']}"
        # elif 'RetinaFace' in quality['label']:
        #     quality['label'] = f"WA + {quality['label']}"

        curves.append(quality)

    # from results.optimal_alpha.generate_plot import curves

    plot_roc(curves)
    # plot_det(curves)
    # plot_cmc(curves)
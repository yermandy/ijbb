import numpy as np
import matplotlib.pyplot as plt

def plot_coviariates(qualities, labels, fig_name, save=True, show=True, fill=True, histtype='bar', bins=40, alpha=0.75):
    fig, axs = plt.subplots(1, 1, tight_layout=True)

    for i, quality in enumerate(qualities):
        axs.hist(quality, bins=bins, alpha=alpha, label=labels[i], density=True, fill=fill, histtype=histtype)

    axs.legend(loc='best', fontsize=16)

    axs.set_xlabel('Quality', fontsize=16)
    axs.set_ylabel('Density', fontsize=16)

    if show:
        plt.show()
    if save:
        fig.set_size_inches((8, 5))
        fig.savefig(f'results/covariates/{fig_name}.png', dpi=300, bbox_inches='tight')


def separate_qualities(cov, labels, quality):

    not_nan = np.flatnonzero(cov != 'NaN')

    quality = quality[not_nan]
    cov = cov[not_nan]

    qualities = []
    for label in labels:
        cov_i = np.flatnonzero(cov == label)
        qualities.append(quality[cov_i])

    return qualities

def separate_qualities_cont(cov, labels, quality):
    not_nan = np.flatnonzero(cov != 'NaN')

    quality = quality[not_nan]
    cov = cov[not_nan].astype(np.float32)
    cov = np.abs(cov)

    qualities = []
    for label in labels:
        cov_i = np.flatnonzero((cov >= label[0]) & (cov < label[1]))
        qualities.append(quality[cov_i])

    return qualities


if __name__ == "__main__":
    
    metadata = np.genfromtxt(f"resources/ijbb_faces_cov.csv", dtype=np.str, delimiter=',', skip_header=1)
    # quality = np.load('resources/features/norms_retina_ijbb_0.5.npy')
    quality = np.load('resources/qualities/ijbb_cnn_fq_score_model_24.npy')

    if len(np.flatnonzero(quality > 1)) > 0:
        min_qs = np.min(quality)
        max_qs = np.max(quality)
        quality = (quality - min_qs) / (max_qs - min_qs)


    '''
    # Forehead
    cov = metadata[:, 2] 
    qualities = separate_qualities(cov, ['1', '0'], quality)
    plot_coviariates(qualities, ['Forehead visible = 1', 'Forehead visible = 0'], 'cov_forehead', show=False)
    # '''

    # Nose / mouth 
    '''
    cov = metadata[:, 1] 
    qualities = separate_qualities(cov, ['1', '0'], quality)
    plot_coviariates(qualities, ['Nose/mouth visible = 1', 'Nose/mouth visible = 0'], 'cov_nose_mouth', show=False)
    # '''

    # Facial hair
    '''
    cov = metadata[:, 3] 
    qualities = separate_qualities(cov, ['0', '1', '2', '3'], quality)
    labels = ['Facial hair = 0', 'Facial hair = 1', 'Facial hair = 2', 'Facial hair = 3']
    plot_coviariates(qualities, labels, 'cov_facial_hair', show=False)
    # '''

    # Yaw
    '''
    cov = metadata[:, 8]
    labels = [(0, 15), (15, 30), (30, 45), (45, 90)]
    qualities = separate_qualities_cont(cov, labels, quality)
    labels = ['Yaw [0°, 15°]', 'Yaw [15°, 30°]', 'Yaw [30°, 45°]', 'Yaw [45°, 90°]']
    plot_coviariates(qualities, labels, 'cov_yaw', show=False)
    # '''

    # Roll
    '''
    cov = metadata[:, 9]
    labels = [(0, 15), (15, 65)]
    qualities = separate_qualities_cont(cov, labels, quality)
    labels = ['Roll [0°, 15°]', 'Roll [15°, 65°]']
    plot_coviariates(qualities, labels, 'cov_roll', show=False, bins=30)
    # '''
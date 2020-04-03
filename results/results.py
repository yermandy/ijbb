import numpy as np


qualities = np.load("./results/results.npz", allow_pickle=True)

for quality in qualities:
    quality = qualities[quality].tolist()

    tars = quality['tar_mean']
    stds = quality['tar_std']
    fars = quality['far']
        
    tars_at_fars = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    print(f"\n{quality['name']}")
    for t_at_f in tars_at_fars:
        tar = tars[np.searchsorted(fars, t_at_f)]
        std = stds[np.searchsorted(fars, t_at_f)]
        print(f'{t_at_f:.0E}: {tar:.3f}')
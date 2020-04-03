import numpy as np

tars_at_fars = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

highest_t_at_f = [0] * len(tars_at_fars)
highest_t_at_f_metric = [''] * len(tars_at_fars)

best_in_quality = {}

def print_qualities(qualities):
    print('–––––––––––––')
    for quality in qualities:
        quality = qualities[quality].tolist()

        q_name = quality['name']
        if q_name == 'SE-ResNet-50':
            continue

        q_name, l = q_name.split(" ")

        if q_name not in best_in_quality:
            best_in_quality[q_name] = {
                'tars_at_fars': [0] * len(tars_at_fars),
                'l': [''] * len(tars_at_fars)
            }

        tars = quality['tar_mean']
        stds = quality['tar_std']
        fars = quality['far']

        print(f"\n{q_name}")
        for i, t_at_f in enumerate(tars_at_fars):
            tar = tars[np.searchsorted(fars, t_at_f)]
            std = stds[np.searchsorted(fars, t_at_f)]
            
            if highest_t_at_f[i] < tar:
                highest_t_at_f[i] = tar
                highest_t_at_f_metric[i] = q_name
            
            BIQ = best_in_quality[q_name]
            if BIQ['tars_at_fars'][i] < tar:
                BIQ['tars_at_fars'][i] = tar
                BIQ['l'][i] = l

            print(f'{t_at_f:.0E}: {tar:.5f}')


alphas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
for alpha in alphas:
    qualities = np.load(f"results/optimal_alpha/results_{alpha}.npz", allow_pickle=True)
    print_qualities(qualities)



print("\nBest in each quality")
for k, v in best_in_quality.items():

    print(f"\n{k}")
    for i in range(len(v['tars_at_fars'])):
        print(f'{v["tars_at_fars"][i]:.5f} : {tars_at_fars[i]:.0E} : {v["l"][i]}')


print("\nThe best")
for i in range(len(highest_t_at_f)):
    print(f'{highest_t_at_f[i]:.5f} : {highest_t_at_f_metric[i]}')


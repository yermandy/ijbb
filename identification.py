import numpy as np
from json import load
from time import time
from plot import *
from config import *
from aggragation import Aggregation

class Identification(Aggregation):

    def __init__(self, template_faces_dict, features, quality_scores=None, method=None):
        super().__init__(
            template_faces_dict,
            features,
            quality_scores,
            method
        )

    def calc_distances(self, templates_features):
        gallery_S1 = np.genfromtxt("protocol/ijbb_1N_gallery_S1.csv", delimiter=",", dtype=np.int, skip_header=1)[:, [0,1]]
        gallery_S2 = np.genfromtxt("protocol/ijbb_1N_gallery_S2.csv", delimiter=",", dtype=np.int, skip_header=1)[:, [0,1]]
        gallery = np.concatenate((gallery_S1, gallery_S2), axis=0)
        probe = np.genfromtxt("protocol/ijbb_1N_probe_mixed.csv", delimiter=",", dtype=np.int, skip_header=1)[:, [0,1]]

        gallery_templates = np.unique(gallery, axis=0)
        probe_templates = np.unique(probe, axis=0)
        
        gallery_features = np.empty((len(gallery_templates), 256))
        gallery_identities = np.empty(len(gallery_templates))
        
        for i, (g, g_i) in enumerate(gallery_templates):
            gallery_features[i] = templates_features[g]
            gallery_identities[i] = g_i

        mated = []
        nonmated = []

        for p, p_i in probe_templates:

            t1 = templates_features[p]

            ## find mated features
            mated_idx = np.flatnonzero(p_i == gallery_identities)
            mated_features = gallery_features[mated_idx]

            ## find nonmated features
            mask = np.ones(len(gallery_templates), np.bool)
            mask[mated_idx] = False
            nonmated_features = gallery_features[mask]

            ## calculate mated and nonmated distances
            nonmated_d = 1 - np.dot(t1, nonmated_features.T)
            mated_d = 1 - np.dot(t1, mated_features.T)

            if len(nonmated_d) > 0: 
                ## find nonmated with min dist
                nonmated_d = np.min(nonmated_d)
                nonmated.append(nonmated_d)
            
            if len(mated_d) > 0: 
                ## find rank
                distances = 1 - np.dot(t1, gallery_features.T)
                sorted_idx = np.argsort(distances)
                sorted_identities = gallery_identities[sorted_idx]
                rank = np.flatnonzero(sorted_identities == p_i)[0] + 1
                
                ## find mated with max dist
                mated_d = np.max(mated_d)
                mated.append([mated_d, rank])
            
        return np.array(nonmated), np.array(mated)
    

    def calc_det(self, nonmated, mated, R=30):

        ranks = mated[:, 1]
        mated = mated[:, 0]

        min_d_nonmated = np.min(nonmated)
        max_d_nonmated = np.max(nonmated) + 1e-8

        min_d_mated = np.min(mated)
        max_d_mated = np.max(mated) + 1e-8

        min_d = min_d_mated if min_d_mated < min_d_nonmated else min_d_mated
        max_d = max_d_mated if max_d_mated > max_d_nonmated else max_d_nonmated
        thresholds = np.linspace(min_d, max_d, 500)

        fpirs = []
        fnirs = []
        for th in thresholds:

            fpir = len(nonmated[nonmated <= th]) 
            fnir = len(mated[(mated > th) | (ranks > R)]) 

            fpirs.append(fpir)
            fnirs.append(fnir)

        fpirs = np.array(fpirs) / nonmated.shape[0]
        fnirs = np.array(fnirs) / mated.shape[0]
        return fpirs, fnirs

    def calc_cmc(self, templates_features):
        gallery_S1 = np.genfromtxt("protocol/ijbb_1N_gallery_S1.csv", delimiter=",", dtype=np.int, skip_header=1)[:, [0,1]]
        gallery_S2 = np.genfromtxt("protocol/ijbb_1N_gallery_S2.csv", delimiter=",", dtype=np.int, skip_header=1)[:, [0,1]]
        gallery = np.concatenate((gallery_S1, gallery_S2), axis=0)
        probe = np.genfromtxt("protocol/ijbb_1N_probe_mixed.csv", delimiter=",", dtype=np.int, skip_header=1)[:, [0,1]]

        gallery_templates = np.unique(gallery, axis=0)
        probe_templates = np.unique(probe, axis=0)

        gallery_features = np.empty((len(gallery_templates), 256))
        gallery_identities = np.empty(len(gallery_templates))
        for i, (g, g_i) in enumerate(gallery_templates):
            gallery_features[i] = templates_features[g]
            gallery_identities[i] = g_i

        found_in_rank = []

        for p, p_i in probe_templates:

            t1 = templates_features[p]

            distances = 1 - np.dot(t1, gallery_features.T)

            sorted_idx = np.argsort(distances)

            sorted_identities = gallery_identities[sorted_idx]

            rank = np.flatnonzero(sorted_identities == p_i)[0] + 1
            found_in_rank.append(rank)
        
        cmc = [0]
        found_in_rank = np.array(found_in_rank)
        for rank in range(1, len(found_in_rank) + 1):
            
            found_n = np.flatnonzero(found_in_rank == rank).shape[0]
            
            prev = cmc[len(cmc) - 1]
            cmc.append(prev + found_n / len(found_in_rank))

        cmc.pop(0)

        return cmc


if __name__ == "__main__":
    
    template_faces_dict = load(open('resources/ijbb_templates_subjects.json', 'r'))
    template_faces_dict = {int(k) : np.array(v) for k, v in template_faces_dict.items()}
    features = np.load("resources/features/features.npy")    

    interp_fpir = np.linspace(1e-4, 1, 100000)
    qualities = new_aggregation

    curves = []
    show_results = True

    for quality in qualities.values():
        print(f'\n{quality["name"]}')
        # if append_calculated(quality, 'averaging', curves, f'results/averaging.npz'):
        #     continue

        # region load values from dict

        if 'file' in quality:
            quality_scores = np.load(f"resources/{quality['file']}")
            # Normalize qualities if not in [0, 1] range
            if len(np.flatnonzero(quality_scores > 1.0)) > 0:
                min_qs = np.min(quality_scores)
                max_qs = np.max(quality_scores)
                quality_scores = (quality_scores - min_qs) / (max_qs - min_qs)
        else:
            quality_scores = None

        features = np.load(f'resources/{quality["features"]}') if 'features' in quality else features

        pairs = np.load(f'resources/{quality["comparisons"]}')['comparisons'] if 'comparisons' in quality else pairs

        method = quality['method'] if 'method' in quality else None

        alpha = quality['alpha'] if 'alpha' in quality else None

        quality['color'] = quality['color'] if 'color' in quality else None

        # endregion

        start = time()
        identify = Identification(
            template_faces_dict=template_faces_dict,
            features=features,
            quality_scores=quality_scores,
            method=method
        )

        templates = identify.calc_templates()
        nonmated_d, mated_d = identify.calc_distances(templates)
        
        ## Calculate DET curve
        fpirs, fnirs = identify.calc_det(nonmated_d, mated_d)

        interp_fnir = np.interp(interp_fpir, fpirs, fnirs)
        print(f'Execution time: {time() - start:.2f} sec')
        
        print(f'AUC: {np.trapz(interp_fnir, interp_fpir):.5f}')

        quality['fpir'] = interp_fpir
        quality['fnir'] = interp_fnir

        ## Calculate CMC curve
        cmc = identify.calc_cmc(templates)
        quality['cmc_prob'] = cmc

        if show_results:
            print('TPIR@FPIR’s of:')
            fnir_at_frir = [1e-2, 1e-1]
            for f_at_f in fnir_at_frir:
                tpir = 1 - interp_fnir[np.searchsorted(interp_fpir, f_at_f)]
                print(f'\t{f_at_f:.0E}: {tpir:.3f}')
            
            print('CMC:')
            for i in [0, 4, 9]:
                print(f'\tRank-{i+1}: {cmc[i]:.3f}')

        curves.append(quality)

    np.savez_compressed("results/results.npz", *curves)

    plot_cmc(curves)
    plot_det(curves)
    

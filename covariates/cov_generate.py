import numpy as np
import json
from itertools import combinations


metadata = np.genfromtxt(f"resources/ijbb_faces_cov.csv", dtype=np.str, delimiter=',', skip_header=1)
faces = np.genfromtxt(f"resources/ijbb_faces.csv", dtype=np.str, delimiter=',')

template_faces_map = json.load(open('resources/ijbb_cov_templates_subjects.json', 'r'))
faces_templates = {v[0] : int(k) for k, v in template_faces_map.items()}


def create_pairs(from_identities):
    pairs = combinations(from_identities, 2)
    pairs = [*pairs]
    pairs = np.array(pairs)

    t1_i = pairs[:, 0]
    t2_i = pairs[:, 1]

    identities = faces[:, 6]

    labels = np.where(identities[t1_i] == identities[t2_i], 1, 0)

    impostor = len(np.flatnonzero(labels == 0))
    print('Impostor: ', impostor)
    print('Genuine: ', len(labels) - impostor)

    for i, (f1, f2) in enumerate(pairs):
        pairs[i] = [faces_templates[f1], faces_templates[f2]]

    return np.concatenate((pairs, labels[:, np.newaxis]), axis=1)

def save_pairs(cov, min_len, name):
    cov = np.random.choice(cov, min_len, replace=False)
    cov_pairs = create_pairs(cov)
    np.savez_compressed(f'resources/covariates/{name}.npz', comparisons=cov_pairs)



if __name__ == "__main__":
    
    # FOREHEAD_VISIBLE
    '''
    cov = metadata[:, 2]
    
    cov_0 = np.flatnonzero(cov == '0')
    cov_1 = np.flatnonzero(cov == '1')

    min_len = np.min([len(cov_0), len(cov_1)])

    save_pairs(cov_0, min_len, 'cov_forehead_0')
    save_pairs(cov_1, min_len, 'cov_forehead_1')
    '''
    
    # NOSE_MOUTH_VISIBLE
    '''
    cov = metadata[:, 1]
    
    cov_0 = np.flatnonzero(cov == '0')
    cov_1 = np.flatnonzero(cov == '1')

    min_len = np.min([len(cov_0), len(cov_1)])

    save_pairs(cov_0, min_len, 'cov_nose_mouth_0')
    save_pairs(cov_1, min_len, 'cov_nose_mouth_1')
    '''

    # FACIAL_HAIR
    '''
    cov = metadata[:, 3]
    
    cov_0 = np.flatnonzero(cov == '0')
    cov_1 = np.flatnonzero(cov == '1')
    cov_2 = np.flatnonzero(cov == '2')
    cov_3 = np.flatnonzero(cov == '3')

    min_len = np.min([len(cov_0), len(cov_1), len(cov_2), len(cov_3)])

    save_pairs(cov_0, min_len, 'cov_facial_hair_0')
    save_pairs(cov_1, min_len, 'cov_facial_hair_1')
    save_pairs(cov_2, min_len, 'cov_facial_hair_2')
    save_pairs(cov_3, min_len, 'cov_facial_hair_3')
    '''
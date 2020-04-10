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

def create_pairs_cont(from_identities, cov, val_from, val_to):
    pairs = combinations(from_identities, 2)
    pairs = [*pairs]
    pairs = np.array(pairs)

    t1_i = pairs[:, 0]
    t2_i = pairs[:, 1]

    diffs = abs(cov[t1_i] - cov[t2_i])
    diff_mask = np.where((diffs >= val_from) & (diffs < val_to))

    t1_i = t1_i[diff_mask]
    t2_i = t2_i[diff_mask]
    pairs = pairs[diff_mask]

    identities = faces[:, 6]

    labels = np.where(identities[t1_i] == identities[t2_i], 1, 0)

    impostor = len(np.flatnonzero(labels == 0))
    genuine = len(labels) - impostor
    print('Impostor: ', impostor)
    print('Genuine: ', genuine)

    for i, (f1, f2) in enumerate(pairs):
        pairs[i] = [faces_templates[f1], faces_templates[f2]]

    # size = len(pairs) if len(pairs) < 3000000 else 3000000

    # genuine_pairs = np.flatnonzero(labels == 1)
    # random_pairs = np.random.choice(pairs.shape[0], size - genuine) # todo delete
    # random_pairs = np.concatenate((random_pairs, genuine_pairs))
    # pairs = pairs[random_pairs]
    # labels = labels[random_pairs]
    # print('Test set: ', len(labels))
    return np.concatenate((pairs, labels[:, np.newaxis]), axis=1)


def save_pairs(cov_ids, min_len, name):
    min_len = len(cov_ids) if min_len > len(cov_ids) else min_len
    cov_ids = np.random.choice(cov_ids, min_len, replace=False)
    cov_pairs = create_pairs(cov_ids)
    np.savez_compressed(f'resources/covariates/{name}.npz', comparisons=cov_pairs)

def save_pairs_cont(cov_ids, cov, val_from, val_to, min_len, name):
    min_len = len(cov_ids) if min_len > len(cov_ids) else min_len
    cov_ids = np.random.choice(cov_ids, min_len, replace=False)
    cov_pairs = create_pairs_cont(cov_ids, cov, val_from, val_to)
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
    # '''
    
    # NOSE_MOUTH_VISIBLE
    '''
    cov = metadata[:, 1]
    
    cov_0 = np.flatnonzero(cov == '0')
    cov_1 = np.flatnonzero(cov == '1')

    min_len = np.min([len(cov_0), len(cov_1)])

    save_pairs(cov_0, min_len, 'cov_nose_mouth_0')
    save_pairs(cov_1, min_len, 'cov_nose_mouth_1')
    # '''

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
    # '''
    
    # YAW
    '''
    cov = metadata[:, 8].astype(np.float32)
    
    cov = np.abs(cov)

    with np.errstate(invalid='ignore'):
        cov_0 = np.flatnonzero((cov >= 0) & (cov < 15))
        cov_1 = np.flatnonzero((cov >= 15) & (cov < 30))
        cov_2 = np.flatnonzero((cov >= 30) & (cov < 45))
        cov_3 = np.flatnonzero((cov >= 45) & (cov < 90))

    min_len = np.min([len(cov_0), len(cov_1), len(cov_2), len(cov_3)])
    # print(min_len)
    print(len(cov_0))
    print(len(cov_1))
    print(len(cov_2))
    print(len(cov_3))
    min_len = 5500
    # raise Exception

    save_pairs_cont(cov_0, cov, 0, 15, min_len, 'cov_yaw_0_15')
    save_pairs_cont(cov_1, cov, 10, 35, min_len, 'cov_yaw_15_30')
    save_pairs_cont(cov_2, cov, 12.5, 60, min_len, 'cov_yaw_30_45')
    save_pairs_cont(cov_3, cov, 20, 90, min_len, 'cov_yaw_45_90')
    # '''

    # ROLL
    '''
    cov = metadata[:, 9].astype(np.float32)
    
    abs_cov = np.abs(cov)

    with np.errstate(invalid='ignore'):
        cov_0 = np.flatnonzero((abs_cov >= 0) & (abs_cov < 15))
        cov_1 = np.flatnonzero((abs_cov >= 15) & (abs_cov < 65))

    min_len = np.min([len(cov_0), len(cov_1)])
    print(len(cov_0))
    print(len(cov_1))
    min_len = 4000

    save_pairs_cont(cov_0, cov, 0, 15, min_len, 'cov_roll_0_15')
    save_pairs_cont(cov_1, cov, 15, 65, min_len, 'cov_roll_15_65')
    # '''
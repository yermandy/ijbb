import csv, os, json
import numpy as np
from os.path import isfile

os.makedirs("resources/features", exist_ok=True)
os.makedirs("results", exist_ok=True)

faces = np.genfromtxt(f"resources/ijbb_faces.csv", dtype=np.str, delimiter=',')
gallery_S1 = np.genfromtxt(f"protocol/ijbb_1N_gallery_S1.csv", dtype=np.str, delimiter=',', skip_header=1)
gallery_S2 = np.genfromtxt(f"protocol/ijbb_1N_gallery_S2.csv", dtype=np.str, delimiter=',', skip_header=1)
probe_img = np.genfromtxt(f"protocol/ijbb_1N_probe_img.csv", dtype=np.str, delimiter=',', skip_header=1)
probe_video = np.genfromtxt(f"protocol/ijbb_1N_probe_video.csv", dtype=np.str, delimiter=',', skip_header=1)
probe_mixed = np.genfromtxt(f"protocol/ijbb_1N_probe_mixed.csv", dtype=np.str, delimiter=',', skip_header=1)

def populate_templates(array, templates):
    for row in array:
        t, id, f = row[[0,1,2]]
        if t not in templates:
            templates[t] = [(f, id)]
        elif (f, id) not in templates[t]:
            templates[t].append((f, id))


# region ijbb_templates_subjects.json

if not isfile("resources/ijbb_templates_subjects.json"):

    templates = {}

    populate_templates(gallery_S1, templates)
    populate_templates(gallery_S2, templates)
    populate_templates(probe_img, templates)
    populate_templates(probe_video, templates)
    populate_templates(probe_mixed, templates)

    paths = faces[:, 0]
    subjects = faces[:, 6]

    for k, values in templates.items():
        new_v = []
        for v in values:
            path, subject = v
            index = np.flatnonzero((paths == path) & (subject == subjects))[0]
            new_v.append(index.item())
        templates[k] = new_v
        
    with open("resources/ijbb_templates_subjects.json", "w") as json_file:
        json.dump(templates, json_file)

else:

    with open('resources/ijbb_templates_subjects.json', 'r') as json_file:
        templates = json.load(json_file)

# endregion

# region ijbb_comparisons.npz

if not isfile("resources/ijbb_comparisons.npz"):

    comparisons = np.genfromtxt(f"protocol/ijbb_11_S1_S2_matches.csv", dtype=np.str, delimiter=',')
    subjects = faces[:, 6]

    comparisons_with_label = np.empty((comparisons.shape[0], 3), dtype=np.int32)

    for i, comparison in enumerate(comparisons):

        t1, t2 = comparison

        identity1_index = int(templates[t1][0])
        identity2_index = int(templates[t2][0])
        
        subject1 = subjects[identity1_index]
        subject2 = subjects[identity2_index]

        comparisons_with_label[i] = [t1, t2, int(subject1 == subject2)]
    
    comparisons = comparisons_with_label
    np.savez_compressed("resources/ijbb_comparisons", comparisons=comparisons)

else:

    comparisons = np.load("resources/ijbb_comparisons.npz")['comparisons']

impostor = comparisons[comparisons[:, 2] == 0].shape[0]
genuine = comparisons.shape[0] - impostor
print(f'loaded {impostor} impostor and {genuine} genuine comparisons for 1:1 Baseline Verification')

# endregion

# region ijbb_cov_templates_subjects.json

if not isfile("resources/ijbb_cov_templates_subjects.json"):

    cov = np.genfromtxt(f"protocol/ijbb_11_covariate_probe_reference_metadata.csv", dtype=np.str, delimiter=',', skip_header=1)

    templates = {}

    populate_templates(cov, templates)

    paths = faces[:, 0]
    subjects = faces[:, 6]

    for k, values in templates.items():
        new_v = []
        for v in values:
            path, subject = v
            index = np.flatnonzero((paths == path) & (subject == subjects))[0]
            new_v.append(index.item())
        templates[k] = new_v
        
    with open("resources/ijbb_cov_templates_subjects.json", "w") as json_file:
        json.dump(templates, json_file)

else:

    with open('resources/ijbb_cov_templates_subjects.json', 'r') as json_file:
        templates = json.load(json_file)

# endregion

# region ijbb_covariates.npz

if not isfile("resources/ijbb_covariates.npz"):

    comparisons = np.genfromtxt(f"protocol/ijbb_11_covariate_matches.csv", dtype=np.str, delimiter=',')
    subjects = np.genfromtxt(f"resources/ijbb_faces.csv", dtype=np.int, delimiter=',')[:, 6]

    comparisons_with_label = np.empty((comparisons.shape[0], 3), dtype=np.int32)

    for i, comparison in enumerate(comparisons):

        t1, t2 = comparison

        identity1_index = int(templates[t1][0])
        identity2_index = int(templates[t2][0])
        
        subject1 = subjects[identity1_index]
        subject2 = subjects[identity2_index]

        comparisons_with_label[i] = [t1, t2, int(subject1 == subject2)]
    
    comparisons = comparisons_with_label
    np.savez_compressed("resources/ijbb_covariates.npz", comparisons=comparisons)

else:

    comparisons = np.load("resources/ijbb_covariates.npz")['comparisons']

impostor = comparisons[comparisons[:, 2] == 0].shape[0]
genuine = comparisons.shape[0] - impostor
print(f'loaded {impostor} impostor and {genuine} genuine comparisons for 1:1 Covariate Verification')

# endregion

# region ijbb_faces_cov.csv

if not isfile("ijbb_faces_cov.csv"):

    metadata = np.genfromtxt(f"protocol/ijbb_metadata.csv", dtype=np.str, delimiter=',')

    metadata = metadata[:68196, 14:]

    np.savetxt("resources/ijbb_faces_cov.csv", metadata, delimiter=",", fmt="%s")

else:

    metadata = np.genfromtxt(f"resources/ijbb_faces_cov.csv", dtype=np.str, delimiter=',', skip_header=1)

# endregion


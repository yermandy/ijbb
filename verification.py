import numpy as np
from json import load
from time import time
from scipy.optimize import fminbound
from plot import *
from config import *
from numpy.linalg import norm

class Verification():

    def __init__(self, template_faces_map, pairs_labels, features, quality_scores=None, method=None):
        """
        1:1 Verification
        
        Parameters
        ----------
        template_faces_map : dict
            Dictionary where for each IJB-A template (key) assigned list of references (value):
            faces (features parameter) it consists of.
        pairs_labels : np.array
            Pairs of faces and labels for verification. (N, 3): N – number of pairs to verify
        features : np.array
            Extracted features. (M, 128): M – number of faces in IJB-A
        quality_scores : np.array, optional
            Extracted qualities. (M,): M – number of faces in IJB-A
        """
        self.template_faces_map = template_faces_map
        self.pairs = pairs_labels
        self.labels = pairs_labels[:, 2]
        self.features = features
        self.quality_scores = quality_scores
        self.thresholds = None
        self.method = method
    
    def compute_templates(self, method=1):
        method = self.method if self.method is not None else method
        if method == 1:
            return self.compute_templates_1()
        if method == 2:
            return self.compute_templates_2()
        if method == 3:
            return self.compute_templates_3()
        if method == 4:
            return self.compute_templates_4()
        raise Exception(f"Method {method} is not supported")

    def compute_templates_1(self):
        templates_features = {}
        for template, faces in self.template_faces_map.items():
            templates_features[template] = np.average(self.features[faces], axis=0)
        return templates_features

    def compute_templates_2(self):
        templates_features = {}
        for template, faces in self.template_faces_map.items():
            qualities = self.quality_scores[faces]
            # l = np.clip(np.log(qualities / (1 - qualities)) / 2, -5, 5)
            logit = np.log(qualities / (1 - qualities)) / 2 
            logit = np.where(logit <= 7, logit, 7)
            c = np.e ** (logit * 1)
            c /= c.sum()
            templates_features[template] = np.sum(c * self.features[faces].T, axis=1)
        return templates_features

    def compute_templates_3(self, alpha=0.5):
        templates_features = {}
        for template, faces in self.template_faces_map.items():
            qualities = self.quality_scores[faces]
            logit = (alpha / qualities) * np.log(qualities / (1 - qualities))
            c = np.exp(logit)
            c /= c.sum()
            templates_features[template] = np.sum(c * self.features[faces].T, axis=1)
        return templates_features
    
    def compute_templates_4(self):
        templates_features = {}
        for template, faces in self.template_faces_map.items():
            qualities = self.quality_scores[faces]
            qualities /= qualities.sum()
            templates_features[template] = np.sum(qualities * self.features[faces].T, axis=1)
        return templates_features

    def batches(self, iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    def find_distances(self, templates_features=None, batch_size=100000):
        """
        Calculate cosine distances between pairs of templates
        
        Parameters
        ----------
        templates_features : dict, optional
            Dictionary where for each template (key) assigned computed feature vector (value)
        
        Returns
        -------
        np.array (N, )
            N cosine distances between N pairs of templates
        """
        if templates_features is None:
            templates_features = self.compute_templates()
        distances = np.empty((self.pairs.shape[0]), dtype=np.float32)
        start, end = 0, 0
        for batch in self.batches(self.pairs, batch_size):
            t1 = np.empty((len(batch), 256), dtype=np.float32)
            t2 = np.empty((len(batch), 256), dtype=np.float32)
            start = end
            for i, pair in enumerate(batch):
                t1[i] = templates_features[pair[0]]
                t2[i] = templates_features[pair[1]]
            end += len(batch)
            distances[start:end] = 1 - np.einsum("ij,ij->i", t1, t2) / (norm(t1, axis=1) * norm(t2, axis=1) + 1e-12)
        return distances

    def calculate_tar_at_far(self, distances=None):
        """
        Evaluates ROC curve using distances and labels.

        Parameters
        -------
        distances : np.array, optional

        Returns
        -------
        tuple(np.array, np.array)
            Returns calculated tar and far
        """
        if distances is None :
            distances = self.find_distances()
        if self.thresholds is None:
            min_d = np.min(distances)
            max_d = np.max(distances) + 1e-8
            self.thresholds = np.linspace(min_d, max_d, min(200, len(distances)))
        pos = distances[self.labels == 1]
        neg = distances[self.labels == 0]
        fars = []
        tars = []
        for th in self.thresholds:
            tar = len(pos[pos < th])
            far = len(neg[neg < th])
            tars.append(tar)
            fars.append(far)
        tars = np.array(tars) / len(pos)
        fars = np.array(fars) / len(neg)
        return tars, fars


def calculate_tars(template_faces_map, features, pairs, quality_scores, method):
    tars = []
    start = time()
    verify = Verification(
        template_faces_map = template_faces_map,
        pairs_labels = pairs,
        features = features,
        quality_scores = quality_scores,
        method = method
    )

    extracted_features = verify.compute_templates()
    distances = verify.find_distances(extracted_features)
    tar, far = verify.calculate_tar_at_far(distances)

    tars.append(np.interp(mean_far, far, tar))
    print(f'dataset was processed in {time() - start:.2f} seconds')
    return tars

if __name__ == "__main__":
    ROOT = 'protocol_11'
    template_faces_map = {}

    template_faces_map = load(open('resources/ijbb_templates_subjects.json', 'r'))
    template_faces_map = {int(k) : v for k, v in template_faces_map.items()}
    features = np.load("resources/features/new_features_retina_ijbb_0.5.npy")
    pairs = np.load(f'resources/ijbb_comparisons.npz')['comparisons']

    mean_far = np.linspace(1e-5, 1, 10000)
    qualities = qualities 

    curves = []

    for quality in qualities.values():

        print(f'\n{quality["name"]}')

        features = np.load(f'resources/{quality["features"]}') if 'features' in quality else features
        method = quality['method'] if 'method' in quality else None
        quality_scores = np.load(f"resources/{quality['file']}") if 'file' in quality else None
        
        tars = calculate_tars(
            template_faces_map = template_faces_map,
            features = features,
            pairs = pairs,
            quality_scores = quality_scores,
            method = method)
        mean_tar = np.mean(tars, axis=0, dtype=np.float64)
        std_tar  = np.std(tars, axis=0, dtype=np.float64)
        
        print(f'AUC: {np.trapz(mean_tar, mean_far):.5f}')

        quality['far']      = mean_far
        quality['tar_mean'] = mean_tar
        quality['tar_std']  = std_tar

        curves.append(quality)

    np.savez_compressed("results/results.npz", *curves)

    plot_tar_at_far(curves, errorbar=False, plot_SOTA=True)
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
            Dictionary where for each IJB-B template (key) assigned list of references (value):
            faces (features parameter) it consists of.
        pairs_labels : np.array
            Pairs of faces and labels for verification. (N, 3): N – number of pairs to verify
        features : np.array
            Extracted features. (M, 256): M – number of faces in IJB-B
        quality_scores : np.array, optional
            Extracted qualities. (M,): M – number of faces in IJB-B
        """
        self.template_faces_map = template_faces_map
        self.pairs = pairs_labels
        self.labels = pairs_labels[:, 2]
        self.features = features
        self.quality_scores = quality_scores
        self.thresholds = None
        self.method = method
    
    def compute_templates(self, method=1, alpha=None):
        method = self.method if self.method is not None else method
        if method == 1:
            return self.compute_templates_1()
        if method == 2:
            return self.compute_templates_2(alpha=alpha)
        if method == 3:
            return self.compute_templates_3(alpha=alpha)
        if method == 4:
            return self.compute_templates_4()
        raise Exception(f"Method {method} is not supported")

    def compute_templates_1(self):
        """
        Averaging
        """
        templates_features = {}
        for template, faces in self.template_faces_map.items():
            templates_features[template] = np.average(self.features[faces], axis=0)
        return templates_features

    def compute_templates_2(self, alpha=None):
        """
        Quality Pooling
        Method described in https://arxiv.org/pdf/1804.01159.pdf
        
        Parameters
        ----------
        alpha : float, optional
            In the paper, it's a hyperparameter λ, by default 1.0
        
        Returns
        -------
        dict
            Dictinary with keys as templates and values are np.array of size 256
        """
        templates_features = {}
        for template, faces in self.template_faces_map.items():
            qualities = self.quality_scores[faces]
            logit = np.log((qualities + 1e-8) / ((1 - qualities) + 1e-8)) / 2 
            sevens = np.repeat(7, len(faces))
            logit = np.amin([logit, sevens], axis=0)
            c = np.e ** (logit * alpha)
            c /= c.sum()
            templates_features[template] = np.sum(c * self.features[faces].T, axis=1)
        return templates_features

    def compute_templates_3(self, alpha=None):
        templates_features = {}
        for template, faces in self.template_faces_map.items():
            qualities = self.quality_scores[faces]
            logit = (alpha / qualities) * np.log(qualities / (1 - qualities))
            c = np.exp(logit)
            c /= c.sum()
            templates_features[template] = np.sum(c * self.features[faces].T, axis=1)
        return templates_features
    
    def compute_templates_4(self):
        """
        Average Weighting
        
        Returns
        -------
        dict
            Dictinary with keys as templates and values are np.array of size 256
        """ 
        templates_features = {}
        for template, faces in self.template_faces_map.items():
            qualities = self.quality_scores[faces]
            qualities /= qualities.sum()
            templates_features[template] = np.sum(qualities * self.features[faces].T, axis=1)
        return templates_features

    def batches(self, iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    def find_distances(self, templates_features=None, batch_size=50000, th=0.2, beta=1.1):
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
            # attenuate = np.empty((len(batch)), dtype=np.bool)
            for i, pair in enumerate(batch):
                t1[i] = templates_features[pair[0]]
                t2[i] = templates_features[pair[1]]

                # lomax1 = np.max(self.quality_scores[pair[0]])
                # lomax2 = np.max(self.quality_scores[pair[1]])

                # attenuate[i] = lomax1 <= th or lomax2 <= th
                
            end += len(batch)
            distances[start:end] = 1 - np.einsum("ij,ij->i", t1, t2) / (norm(t1, axis=1) * norm(t2, axis=1) + 1e-12)

            # distances[start:end] = np.where(attenuate, distances[start:end], distances[start:end] / beta)
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


def append_calculated(quality, name, curves, file):
    if quality['name'] == name:
        npz_dict = np.load(file, allow_pickle=True)
        for arr in npz_dict:
            q = npz_dict[arr].tolist()
            if q['name'] == name:
                quality['far'] = q['far']
                quality['tar_mean'] = q['tar_mean']
                quality['tar_std'] = q['tar_std']
                print(f"AUC: {np.trapz(quality['tar_mean'], quality['far']):.5f}")
                curves.append(quality)
                return True
    return False
    

if __name__ == "__main__":
    template_faces_map = load(open('resources/ijbb_templates_subjects.json', 'r'))
    pairs = np.load(f'resources/ijbb_comparisons.npz')['comparisons']

    # template_faces_map = load(open('resources/ijbb_cov_templates_subjects.json', 'r'))
    # pairs = np.load(f'resources/ijbb_covariates.npz')['comparisons']

    template_faces_map = {int(k) : v for k, v in template_faces_map.items()}
    features = np.load("resources/features/features_retina_ijbb_0.5.npy")    

    # pairs = pairs[:20000]

    mean_far = np.linspace(1e-5, 1, 1000000)
    # qualities = cov_size
    qualities = average_weighting

    curves = []
    
    for quality in qualities.values():
        print(f'\n{quality["name"]}')
        # if append_calculated(quality, 'SE-ResNet-50', curves, f'results/results.npz'):
            # continue

        # region load values from dict

        if 'features' in quality:
            features = np.load(f'resources/{quality["features"]}')

        if 'comparisons' in quality:
            pairs = np.load(f'resources/{quality["comparisons"]}')['comparisons']

        if 'method' in quality:
            method = quality['method']
        else:
            method = None

        if 'file' in quality:
            quality_scores = np.load(f"resources/{quality['file']}")
            # Normalize qualities if not in [0, 1] range
            if len(np.flatnonzero(quality_scores > 1.0)) > 0:
                min_qs = np.min(quality_scores)
                max_qs = np.max(quality_scores)
                quality_scores = (quality_scores - min_qs) / (max_qs - min_qs)
        else:
            quality_scores = None

        if 'alpha' in quality:
            alpha = quality['alpha']
        else:
            alpha = None

        if 'color' not in quality:
            quality['color'] = None

        # endregion

        start = time()
        verify = Verification(
            template_faces_map=template_faces_map,
            pairs_labels=pairs,
            features=features,
            quality_scores=quality_scores,
            method=method
        )

        extracted_features = verify.compute_templates(alpha=alpha)
        distances = verify.find_distances(extracted_features)
        tar, far = verify.calculate_tar_at_far(distances)

        tars = [np.interp(mean_far, far, tar)]
        print(f'dataset was processed in {time() - start:.2f} seconds')

        mean_tar = np.mean(tars, axis=0, dtype=np.float64)
        std_tar  = np.std(tars, axis=0, dtype=np.float64)
        
        print(mean_tar, mean_far)
        print(f'AUC: {np.trapz(mean_tar, mean_far):.5f}')

        quality['far']      = mean_far
        quality['tar_mean'] = mean_tar
        quality['tar_std']  = std_tar

        curves.append(quality)

    np.savez_compressed("results/results.npz", *curves)

    plot_tar_at_far(curves, errorbar=False, plot_SOTA=False)
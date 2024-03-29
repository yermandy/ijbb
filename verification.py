import numpy as np
from json import load
from time import time
from plot import *
from config import *
from aggragation import Aggregation

class Verification(Aggregation):

    def __init__(self, template_faces_dict, pairs_labels, features, quality_scores=None, method=None):
        """
        1:1 Verification
        
        Parameters
        ----------
        template_faces_dict : dict
            Dictionary where for each IJB-B template (key) assigned list of references (value):
            faces (features parameter) it consists of.
        pairs_labels : np.array
            Pairs of faces and labels for verification. (N, 3): N – number of pairs to verify
        features : np.array
            Extracted features. (M, D): M – number of faces in IJB-B, D – dimensions of features
        quality_scores : np.array, optional
            Extracted qualities. (M,): M – number of faces in IJB-B
        """
        super().__init__(
            template_faces_dict,
            features,
            quality_scores,
            method
        )
        self.features_dim = features.shape[1]
        self.pairs = pairs_labels
        self.labels = pairs_labels[:, 2]

    def batches(self, iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    def calc_distances(self, templates_features=None, batch_size=50000, th=0.2, beta=1.1):
        """
        Calculate cosine distances between pairs of templates.
        It is assumed that all template descriptors have length 1.
        
        Parameters
        ----------
        templates_features : dict, optional
            Dictionary where for each template (key) assigned normalized feature vector (value)
        
        Returns
        -------
        np.array (N, )
            N cosine distances between N pairs of templates
        """
        if templates_features is None:
            templates_features = self.calc_templates()
        distances = np.empty((self.pairs.shape[0]), dtype=np.float32)
        start, end = 0, 0
        for batch in self.batches(self.pairs, batch_size):
            t1 = np.empty((len(batch), self.features_dim), dtype=np.float32)
            t2 = np.empty((len(batch), self.features_dim), dtype=np.float32)
            start = end
            end += len(batch)
            # attenuate = np.empty((len(batch)), dtype=np.bool)
            for i, pair in enumerate(batch):
                t1[i] = templates_features[pair[0]]
                t2[i] = templates_features[pair[1]]
                # lomax1 = np.max(self.quality_scores[pair[0]])
                # lomax2 = np.max(self.quality_scores[pair[1]])
                # attenuate[i] = lomax1 <= th or lomax2 <= th

            ## find cosine distance, assume template descriptors are normalized
            distances[start:end] = 1 - np.einsum("ij,ij->i", t1, t2)
            # distances[start:end] = np.where(attenuate, distances[start:end], distances[start:end] / beta)
        return distances

    def calc_roc(self, distances=None):
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
            distances = self.calc_distances()
        min_d = np.min(distances)
        max_d = np.max(distances) + 1e-8
        thresholds = np.linspace(min_d, max_d, min(200, len(distances)))
        pos_d = distances[self.labels == 1]
        neg_d = distances[self.labels == 0]
        fars = np.empty(len(thresholds))
        tars = np.empty(len(thresholds))
        for i, th in enumerate(thresholds):
            tars[i] = len(pos_d[pos_d < th])
            fars[i] = len(neg_d[neg_d < th])
        tars = tars / len(pos_d)
        fars = fars / len(neg_d)
        return tars, fars


def append_calculated(quality, name, curves, file):
    if quality['name'] == name:
        npz_dict = np.load(file, allow_pickle=True)
        for arr in npz_dict:
            q = npz_dict[arr].tolist()
            if q['name'] == name:
                quality = q
                print(f"AUC: {np.trapz(quality['tar'], quality['far']):.5f}")
                curves.append(quality)
                return True
    return False
    

if __name__ == "__main__":
    features = np.load("resources/features/features_retina_ijbb_0.5.npy")    

    covariates = False
    show_results = True

    if covariates:
        template_faces_dict = load(open('resources/ijbb_cov_templates_subjects.json', 'r'))
        pairs = np.load(f'resources/ijbb_covariates.npz')['comparisons']
        qualities = cov_yaw
    else:
        template_faces_dict = load(open('resources/ijbb_templates_subjects.json', 'r'))
        pairs = np.load(f'resources/ijbb_comparisons.npz')['comparisons']
        qualities = new_aggregation

    template_faces_dict = {int(k) : np.array(v) for k, v in template_faces_dict.items()}
        

    # pairs = pairs[:1000000]
    interp_far = np.linspace(1e-5, 1, 1000000)

    curves = []
    
    for quality in qualities.values():
        print(f'\n{quality["name"]}')
        if append_calculated(quality, 'averaging SENet-50', curves, f'results/se_vs_arc.npz'):
            continue
        if append_calculated(quality, 'averaging ArcFace', curves, f'results/se_vs_arc.npz'):
            continue
        if append_calculated(quality, 'media averaging ArcFace', curves, f'results/media_arc.npz'):
            continue
        if append_calculated(quality, 'media averaging 2 ArcFace', curves, f'results/media_arc_and_cnn_fq.npz'):
            continue
        
        

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
        verify = Verification(
            template_faces_dict=template_faces_dict,
            pairs_labels=pairs,
            features=features,
            quality_scores=quality_scores,
            method=method
        )

        extracted_features = verify.calc_templates(alpha=alpha)
        distances = verify.calc_distances(extracted_features)
        tar, far = verify.calc_roc(distances)

        interp_tar = np.interp(interp_far, far, tar)
        print(f'dataset was processed in {time() - start:.2f} seconds')
        print(f'AUC: {np.trapz(interp_tar, interp_far):.5f}')

        quality['far'] = interp_far
        quality['tar'] = interp_tar

        if show_results:
            tars_at_fars = [1e-5, 1e-4, 1e-3, 1e-2]
            for t_at_f in tars_at_fars:
                tar = interp_tar[np.searchsorted(interp_far, t_at_f)]
                print(f'{t_at_f:.0E}: {tar:.3f}')

        curves.append(quality)

    np.savez_compressed("results/verification_results.npz", *curves)

    plot_roc(curves, plot_SOTA=False)

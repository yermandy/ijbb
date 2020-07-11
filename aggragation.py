import numpy as np
from numpy.linalg import norm
from sklearn.cluster import KMeans

class Aggregation:

    def __init__(self, template_faces_dict, features, quality_scores, method=None):
        self.method = method
        self.template_faces_dict = template_faces_dict
        self.features = features
        self.quality_scores = quality_scores


    def calc_templates(self, method=1, alpha=None):
        method = method if self.method is None else self.method
        if method == 1:
            return self.calc_templates_avg()
        elif method == 2:
            return self.calc_templates_quality_pooling(alpha=alpha)
        elif method == 3:
            return self.calc_templates_3(alpha=alpha)
        elif method == 4:
            return self.calc_templates_weighted_avg()
        elif method == 5:
            return self.calc_templates_media_avg()
        elif method == 6:
            return self.calc_templates_weighted_media_avg()
        elif method == 7:
            return self.calc_templates_weighted_cluster_avg()
        raise Exception(f"Method {method} is not supported")


    def calc_templates_avg(self):
        """
        Averaging
        """
        templates_features = {}
        for template, faces in self.template_faces_dict.items():
            t = np.average(self.features[faces], axis=0)
            templates_features[template] = t / norm(t)
        return templates_features


    def calc_templates_media_avg(self):
        templates_features = {}
        is_img = np.genfromtxt("resources/ijbb_is_img.csv", delimiter=",", dtype=np.int)
        for template, faces in self.template_faces_dict.items():
            mask = is_img[faces] == 1            
            arr = []

            ## aggregate img
            img_faces = faces[mask]
            if len(img_faces) > 0:
                t = np.average(self.features[img_faces], axis=0)
                arr.append(t / norm(t))

            ## aggregate frames
            frames_faces = faces[~mask]
            if len(frames_faces) > 0:
                t = np.average(self.features[frames_faces], axis=0)
                arr.append(t / norm(t))

            t = np.mean(arr, axis=0)
            templates_features[template] = t / norm(t)
        return templates_features


    def calc_templates_weighted_media_avg(self):
        templates_features = {}
        is_img = np.genfromtxt("resources/ijbb_is_img.csv", delimiter=",", dtype=np.int)
        for template, faces in self.template_faces_dict.items():
            mask = is_img[faces] == 1            
            arr = []
            weights = []

            ## aggregate img
            img_faces = faces[mask]
            if len(img_faces) > 0:
                qualities = self.quality_scores[img_faces]                
                weights.append(np.mean(np.exp(qualities)))
                qualities /= qualities.sum()
                t = np.sum(qualities * self.features[img_faces].T, axis=1)
                t = t / norm(t)                
                arr.append(t)

            ## aggregate frames
            frames_faces = faces[~mask]
            if len(frames_faces) > 0:
                qualities = self.quality_scores[frames_faces]
                weights.append(np.mean(np.exp(qualities)))
                qualities /= qualities.sum()
                t = np.sum(qualities * self.features[frames_faces].T, axis=1)
                t = t / norm(t)
                arr.append(t)
                
            t = np.average(arr, axis=0, weights=weights)
            templates_features[template] = t / norm(t)
        return templates_features


    def calc_templates_weighted_cluster_avg(self):
        templates_features = {}
        # is_img = np.genfromtxt("resources/ijbb_is_img.csv", delimiter=",", dtype=np.int)
        for template, faces in self.template_faces_dict.items():
            # mask = is_img[faces] == 1            
            
            qualities = self.quality_scores[faces]
            features = self.features[faces]

            arr = []
            weights = []

            n_clusters = 5 if 5 < features.shape[0] else features.shape[0]
            kmeans = KMeans(n_clusters=n_clusters, n_init=4).fit(features)

            clusters = kmeans.labels_
            clusters_sorted = np.argsort(clusters)
            clusters = clusters[clusters_sorted]
            features = features[clusters_sorted]
            qualities = qualities[clusters_sorted]

            # print(features)
            # print(qualities)
            # print(clusters)

            indices_to_split = np.cumsum(np.unique(clusters, return_counts=True)[1])[:-1]
            clusters_of_features = np.split(features, indices_to_split)
            clusters_of_qualities = np.split(qualities, indices_to_split)

            for cluster_features, cluster_qualities in zip(clusters_of_features, clusters_of_qualities):
                weights.append(np.mean(cluster_qualities))
                cluster_qualities /= cluster_qualities.sum()
                t = np.sum(cluster_qualities * cluster_features.T, axis=1)
                t = t / norm(t)
                arr.append(t)
                # print(cluster_features)
                # print(cluster_qualities)


            '''
            ## aggregate img
            img_faces = faces[mask]
            if len(img_faces) > 0:
                qualities = self.quality_scores[img_faces]
                weights.append(np.mean(np.exp(qualities)))
                qualities /= qualities.sum()
                t = np.sum(qualities * self.features[img_faces].T, axis=1)
                t = t / norm(t)                
                arr.append(t)

            ## aggregate frames
            frames_faces = faces[~mask]
            if len(frames_faces) > 0:
                qualities = self.quality_scores[frames_faces]
                weights.append(np.mean(np.exp(qualities)))
                qualities /= qualities.sum()
                t = np.sum(qualities * self.features[frames_faces].T, axis=1)
                t = t / norm(t)
                arr.append(t)
            # '''
                
            t = np.average(arr, axis=0, weights=weights)
            templates_features[template] = t / norm(t)
        return templates_features


    def calc_templates_quality_pooling(self, alpha=None):
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
        for template, faces in self.template_faces_dict.items():
            qualities = self.quality_scores[faces]
            logit = np.log((qualities + 1e-8) / ((1 - qualities) + 1e-8)) / 2 
            sevens = np.repeat(7, len(faces))
            logit = np.amin([logit, sevens], axis=0)
            c = np.e ** (logit * alpha)
            c /= c.sum()
            t = np.sum(c * self.features[faces].T, axis=1)
            templates_features[template] = t / norm(t)
        return templates_features

    def calc_templates_3(self, alpha=None):
        # TODO rename this method
        templates_features = {}
        for template, faces in self.template_faces_dict.items():
            qualities = self.quality_scores[faces]
            logit = (alpha / qualities) * np.log(qualities / (1 - qualities))
            c = np.exp(logit)
            c /= c.sum()
            t = np.sum(c * self.features[faces].T, axis=1)
            templates_features[template] = t / norm(t)
        return templates_features
    

    def calc_templates_weighted_avg(self):
        """
        Weighted Averaging
        
        Returns
        -------
        dict
            Dictinary with keys as templates and values are np.array of size 256
        """ 
        templates_features = {}
        for template, faces in self.template_faces_dict.items():
            qualities = self.quality_scores[faces]
            qualities /= qualities.sum()
            t = np.sum(qualities * self.features[faces].T, axis=1)
            templates_features[template] = t / norm(t)
        return templates_features
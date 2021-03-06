from sklearn.cluster import KMeans
import numpy as np

class Auror:

    def __init__(self, k):
        self.k = k

    def score(self, grad):

        input = np.asarray(list(grad.values()))
        kmeans = KMeans(n_clusters=self.k, random_state=0).fit(input)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        score = {}
        i = 0
        for key in grad:
            cluster = labels[i]
            score[key] = np.sqrt(np.sum(np.square(np.asarray(grad[key]) - centers[cluster])))
            i += 1

        max_val = max(score.values())
        min_val = min(score.values())
        for key in score:
            score[key] = (score[key] - min_val) / (max_val - min_val)

        return score


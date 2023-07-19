import numpy as np


class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = max(n_clusters, 2)
        self.data = None

    def dist(self, a, b):
        return np.linalg.norm(a - b)

    def centroid(self, points):
        l = len(points)
        n = len(points[0])

        center = [0]*n

        for point in points:
            center += point

        return center/l

    def normalize(self, data):
        norm_data_trans = []
        for col in data.T:
            norm_data_trans.append((col - col.mean())/col.std())

        return np.array(norm_data_trans).T

    def load_data(self, data):
        self.data = data

    def fit(self):
        if self.data is None:
            print('Load data first')
            return

        data = self.normalize(self.data)

        # cause i'm too dumb to do random sampling without replacement
        data2 = data.copy()
        np.random.seed(24)
        np.random.shuffle(data2)

        centers = data2[:self.n_clusters]

        while True:
            clusters = [[] for _ in range(self.n_clusters)]

            for d in data:
                min_dist = self.dist(d, centers[0])
                min_cluster = 0

                for i in range(self.n_clusters):
                    if self.dist(d, centers[i]) < min_dist:
                        min_dist = self.dist(d, centers[i])
                        min_cluster = i

                clusters[min_cluster].append(d)

            new_centers = [[] for _ in range(self.n_clusters)]
            for i in range(self.n_clusters):
                new_centers[i] = self.centroid(clusters[i])

            diffs = []
            for i in range(self.n_clusters):
                diffs.append(sum(abs(new_centers[i] - centers[i])))

            diff = sum(diffs)
            centers = new_centers

            if diff < 0.001:
                break

        self.centers = centers
        self.clusters = clusters

        self.classes = []
        for d in data:
            min_dist = self.dist(d, self.centers[0])
            min_cluster = 0

            for i in range(self.n_clusters):
                if self.dist(d, self.centers[i]) < min_dist:
                    min_dist = self.dist(d, self.centers[i])
                    min_cluster = i

            self.classes.append(min_cluster)


if __name__ == '__main__':
    from sklearn.datasets import load_wine
    from sklearn.cluster import KMeans as sKMeans

    data_dict = load_wine()

    kmeans = KMeans(3)
    kmeans.load_data(data_dict['data'])

    kmeans.fit()

    print('True:')
    print(data_dict['target'])

    print('\nMine:')
    print(kmeans.classes)

    skmeans = sKMeans(n_clusters=3, n_init='auto', random_state=24)
    skmeans.fit(data_dict['data'])

    print('\nsklearn:')
    print(skmeans.labels_)


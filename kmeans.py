from sklearn.cluster import KMeans
import pickle
import copy


class Kmeans:

    def __init__(self, num_cluster=None, stop_rate=.15):
        self.num_cluster = num_cluster
        self.stop_rate = stop_rate

    def train(self, input_data):

        # find number of optimal cluster using elbow method (in case that num of cluster is not determined)
        if self.num_cluster is None:
            n_clusters = self.num_cluster
            scores = []
            max_k = 1
            # make sure number of clusters are not more than length of unique data
            for col in input_data.columns:
                max_k = max(len(input_data[col].unique()), max_k)

            #  find the optimal number of clusters between 1 and max(max_k,100)
            for k in range(1, min(max_k, 100)):
                tmp_model = KMeans(n_clusters=k)
                tmp_model.fit(input_data)
                score = tmp_model.score(input_data)
                if k != 1:
                    if (abs(abs(score) - abs(scores[-1])) / abs(score)) < self.stop_rate:
                        n_clusters = k
                        break
                scores.append(score)

            if n_clusters is None:
                n_clusters = min(max_k, 100)
        else:
            n_clusters = self.num_cluster

        kmeans_model = KMeans(n_clusters=n_clusters)
        kmeans_model.fit(input_data)

        clusters = kmeans_model.predict(input_data)
        output_data = input_data.copy()
        output_data['cluster'] = clusters

        # save the model
        model_path = 'models/kmeans-' + str(n_clusters) + '-cluster'
        with open(model_path, 'wb') as model_file:
            pickle.dump(kmeans_model, model_file)

        return output_data


